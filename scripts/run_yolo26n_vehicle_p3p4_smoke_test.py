"""Build a compact YOLO26n-derived vehicle detector and run it on four full frames.

This is an initialization smoke test, not a trained-model evaluation. It transfers
the surviving YOLO26n backbone layers into the P3/P4 student, then runs tiled
inference on one declared frame from A, B, C and D.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIRECTORY / "configs" / "yolo26n_vehicle_p3p4_smoke_test.yaml"
ULTRALYTICS_CONFIG_DIRECTORY = PROJECT_DIRECTORY / "My Artifacts" / "ultralytics"
VEHICLE_CLASS_ID = 0
VEHICLE_CLASS_NAME = "vehicle"
OVERLAY_COLOR = "#19c37d"
OVERLAY_TEXT_COLOR = "white"
OVERLAY_QUALITY = 95
OVERLAY_SUBSAMPLING = 0
IMAGE_HASH_CHUNK_SIZE_BYTES = 1024 * 1024
MINIMUM_SOURCE_TENSOR_DIMENSIONS = 1


def read_yaml(file_path: Path) -> dict:
    """Read a YAML file as a mapping."""
    with file_path.open("r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file)


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configuration path from the project root."""
    candidate = Path(configured_path)
    return candidate if candidate.is_absolute() else PROJECT_DIRECTORY / candidate


def read_manifest(file_path: Path) -> list[dict[str, str]]:
    """Read the canonical frame manifest in its declared order."""
    with file_path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if not rows:
        raise ValueError("Frame manifest is empty")
    return rows


def select_sample_frames(rows: list[dict[str, str]], source_videos: list[str]) -> list[dict[str, str]]:
    """Choose the first declared frame from every requested source video."""
    samples: list[dict[str, str]] = []
    for video_name in source_videos:
        sample = next(
            (
                row
                for row in rows
                if Path(row["source_video_relative_path"]).name == video_name
            ),
            None,
        )
        if sample is None:
            raise ValueError(f"No manifest frame found for {video_name}")
        samples.append(sample)
    return samples


def calculate_sha256(file_path: Path) -> str:
    """Calculate a content hash using bounded memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(IMAGE_HASH_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_tile_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    """Return tile origins that cover an axis and align the final tile to its end."""
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    origins = list(range(0, length - tile_size + 1, stride))
    final_origin = length - tile_size
    if origins[-1] != final_origin:
        origins.append(final_origin)
    return origins


def get_tile_ownership_bounds(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: float,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Return the minimum centre accepted from each tile after its overlaps."""
    horizontal_origins = get_tile_origins(image_width, tile_size, overlap)
    vertical_origins = get_tile_origins(image_height, tile_size, overlap)
    bounds: dict[tuple[int, int], tuple[float, float]] = {}
    for top in vertical_origins:
        for left in horizontal_origins:
            minimum_x = float("-inf")
            minimum_y = float("-inf")
            if left != horizontal_origins[0]:
                previous_left = max(origin for origin in horizontal_origins if origin < left)
                overlap_end = min(previous_left + tile_size, left + tile_size)
                minimum_x = (left + overlap_end) / 2.0
            if top != vertical_origins[0]:
                previous_top = max(origin for origin in vertical_origins if origin < top)
                overlap_end = min(previous_top + tile_size, top + tile_size)
                minimum_y = (top + overlap_end) / 2.0
            bounds[(left, top)] = (minimum_x, minimum_y)
    return bounds


def calculate_box_area(box: list[float]) -> float:
    """Return the positive area of an xyxy box."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def calculate_iou(first_box: list[float], second_box: list[float]) -> float:
    """Calculate IoU for two xyxy boxes."""
    left = max(first_box[0], second_box[0])
    top = max(first_box[1], second_box[1])
    right = min(first_box[2], second_box[2])
    bottom = min(first_box[3], second_box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = calculate_box_area(first_box) + calculate_box_area(second_box) - intersection
    return intersection / union if union > 0.0 else 0.0


def calculate_containment(container: list[float], candidate: list[float]) -> float:
    """Return the fraction of the smaller box enclosed by the overlap."""
    left = max(container[0], candidate[0])
    top = max(container[1], candidate[1])
    right = min(container[2], candidate[2])
    bottom = min(container[3], candidate[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    smaller_area = min(calculate_box_area(container), calculate_box_area(candidate))
    return intersection / smaller_area if smaller_area > 0.0 else 0.0


def calculate_aspect_ratio(box: list[float]) -> float:
    """Return the longer side divided by the shorter side."""
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return max(width, height) / min(width, height) if min(width, height) > 0.0 else float("inf")


def keep_prediction_owned_by_tile(prediction: dict, minimum_x: float, minimum_y: float) -> bool:
    """Keep only boxes whose centre lies in the current tile's ownership region."""
    box = prediction["box_xyxy"]
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    return center_x >= minimum_x and center_y >= minimum_y


def remove_elongated_parts(predictions: list[dict], aspect_ratio: float, containment: float, larger_area_ratio: float) -> tuple[list[dict], int]:
    """Remove an elongated proposal that is enclosed by a substantially larger proposal."""
    kept: list[dict] = []
    removed_count = 0
    for candidate in predictions:
        candidate_box = candidate["box_xyxy"]
        candidate_area = calculate_box_area(candidate_box)
        is_part = any(
            calculate_aspect_ratio(candidate_box) >= aspect_ratio
            and calculate_box_area(other["box_xyxy"]) >= candidate_area * larger_area_ratio
            and calculate_containment(other["box_xyxy"], candidate_box) >= containment
            for other in predictions
            if other is not candidate
        )
        if is_part:
            removed_count += 1
        else:
            kept.append(candidate)
    return kept, removed_count


def suppress_overlapping_boxes(predictions: list[dict], iou_threshold: float) -> tuple[list[dict], int]:
    """Keep the highest-confidence box from overlapping one-class predictions."""
    kept: list[dict] = []
    removed_count = 0
    for candidate in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        if any(calculate_iou(candidate["box_xyxy"], existing["box_xyxy"]) > iou_threshold for existing in kept):
            removed_count += 1
            continue
        kept.append(candidate)
    return kept, removed_count


def copy_tensor_prefix(source_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor | None:
    """Copy equal-shaped tensors or their leading compatible channel dimensions."""
    if source_tensor.ndim != target_tensor.ndim:
        return None
    if source_tensor.ndim < MINIMUM_SOURCE_TENSOR_DIMENSIONS:
        return None
    if any(source_size < target_size for source_size, target_size in zip(source_tensor.shape, target_tensor.shape)):
        return None
    slices = tuple(slice(0, dimension) for dimension in target_tensor.shape)
    return source_tensor[slices].to(device=target_tensor.device, dtype=target_tensor.dtype)


def transfer_backbone_weights(source_model: torch.nn.Module, student_model: torch.nn.Module) -> dict:
    """Transfer layers 0-6 from YOLO26n, slicing only narrowed channels.

    These layers retain the same topology in the student. P4 SPPF and the whole
    neck/head have changed shape or connectivity and deliberately remain newly
    initialized.
    """
    exact_tensor_count = 0
    sliced_tensor_count = 0
    initialized_tensor_count = 0
    exact_parameter_count = 0
    sliced_parameter_count = 0
    initialized_parameter_count = 0
    copied_layers: list[int] = []
    for layer_index in range(7):
        source_state = source_model.model[layer_index].state_dict()
        student_state = student_model.model[layer_index].state_dict()
        student_parameter_names = {
            parameter_name
            for parameter_name, _ in student_model.model[layer_index].named_parameters()
        }
        updated_state = dict(student_state)
        copied_layers.append(layer_index)
        for tensor_name, target_tensor in student_state.items():
            source_tensor = source_state.get(tensor_name)
            copied_tensor = None if source_tensor is None else copy_tensor_prefix(source_tensor, target_tensor)
            tensor_size = target_tensor.numel()
            if copied_tensor is None:
                initialized_tensor_count += 1
                if tensor_name in student_parameter_names:
                    initialized_parameter_count += tensor_size
                continue
            updated_state[tensor_name] = copied_tensor
            if source_tensor.shape == target_tensor.shape:
                exact_tensor_count += 1
                if tensor_name in student_parameter_names:
                    exact_parameter_count += tensor_size
            else:
                sliced_tensor_count += 1
                if tensor_name in student_parameter_names:
                    sliced_parameter_count += tensor_size
        student_model.model[layer_index].load_state_dict(updated_state, strict=True)

    untouched_parameter_count = sum(
        parameter.numel()
        for layer in student_model.model[7:]
        for parameter in layer.parameters()
    )
    return {
        "source_checkpoint_architecture": "YOLO26n",
        "copied_backbone_layers": copied_layers,
        "transfer_method": "exact_tensor_copy_or_leading_channel_slice",
        "exact_tensor_count": exact_tensor_count,
        "sliced_tensor_count": sliced_tensor_count,
        "new_tensor_count_in_copied_layers": initialized_tensor_count,
        "exact_parameter_count": exact_parameter_count,
        "sliced_parameter_count": sliced_parameter_count,
        "new_parameter_count_in_copied_layers": initialized_parameter_count,
        "new_parameter_count_in_changed_layers": untouched_parameter_count,
    }


def draw_predictions(image: Image.Image, predictions: list[dict]) -> Image.Image:
    """Draw one-class predictions on a full-resolution image."""
    overlay = image.copy()
    drawing = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    line_width = max(2, round(max(image.size) / 1000))
    for prediction in predictions:
        box = prediction["box_xyxy"]
        drawing.rectangle(box, outline=OVERLAY_COLOR, width=line_width)
        label = f"Vehicle {prediction['confidence']:.2f}"
        text_box = drawing.textbbox((box[0], box[1]), label, font=font)
        drawing.rectangle(text_box, fill=OVERLAY_COLOR)
        drawing.text((box[0], box[1]), label, fill=OVERLAY_TEXT_COLOR, font=font)
    return overlay


def write_json(file_path: Path, value) -> None:
    """Write readable deterministic JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_student(config: dict):
    """Build the compact student, transfer YOLO26n features, and prepare inference."""
    os.environ["YOLO_CONFIG_DIR"] = str(ULTRALYTICS_CONFIG_DIRECTORY)
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    LOGGER.setLevel("ERROR")
    source_checkpoint = resolve_project_path(config["source_checkpoint"])
    source = YOLO(source_checkpoint)
    student = YOLO(resolve_project_path(config["model_yaml"]), task="detect")
    student.model.names = {VEHICLE_CLASS_ID: VEHICLE_CLASS_NAME}
    transfer_report = transfer_backbone_weights(source.model, student.model)
    total_parameter_count = sum(parameter.numel() for parameter in student.model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in student.model.parameters() if parameter.requires_grad
    )
    if total_parameter_count > config["max_parameter_count"]:
        raise ValueError(
            f"Student has {total_parameter_count:,} parameters, exceeding "
            f"{config['max_parameter_count']:,}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test but is not available")
    transfer_report["total_parameter_count"] = total_parameter_count
    transfer_report["trainable_parameter_count"] = trainable_parameter_count
    transfer_report["transferred_parameter_fraction"] = round(
        (transfer_report["exact_parameter_count"] + transfer_report["sliced_parameter_count"])
        / total_parameter_count,
        6,
    )
    return student, source_checkpoint, transfer_report


def predict_tile(student, tile: Image.Image, config: dict) -> list[dict]:
    """Run one tile through the randomly initialized detection head."""
    result = student.predict(
        source=tile,
        imgsz=config["input_size_pixels"],
        conf=config["confidence_threshold"],
        iou=1.0,
        classes=[VEHICLE_CLASS_ID],
        device=config["device"],
        half=True,
        batch=1,
        nms=True,
        verbose=False,
    )[0]
    if result.boxes is None:
        return []
    return [
        {
            "box_xyxy": [float(value) for value in box],
            "confidence": float(confidence),
            "class_id": VEHICLE_CLASS_ID,
            "class_name": VEHICLE_CLASS_NAME,
        }
        for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())
    ]


def process_frame(student, row: dict[str, str], config: dict) -> tuple[list[dict], list[dict], dict, Image.Image]:
    """Tile one arbitrary-size source image and return full-image predictions."""
    image_path = resolve_project_path(row["relative_image_path"])
    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
    ownership_bounds = get_tile_ownership_bounds(
        image.width,
        image.height,
        config["tile_size_pixels"],
        config["tile_overlap"],
    )
    raw_predictions: list[dict] = []
    owned_predictions: list[dict] = []
    rejected_by_ownership = 0
    for top in sorted({origin[1] for origin in ownership_bounds}):
        for left in sorted({origin[0] for origin in ownership_bounds}):
            tile = image.crop(
                (
                    left,
                    top,
                    min(left + config["tile_size_pixels"], image.width),
                    min(top + config["tile_size_pixels"], image.height),
                )
            )
            for prediction in predict_tile(student, tile, config):
                box = prediction["box_xyxy"]
                prediction["box_xyxy"] = [box[0] + left, box[1] + top, box[2] + left, box[3] + top]
                prediction["frame_key"] = row["frame_key"]
                prediction["relative_image_path"] = row["relative_image_path"]
                prediction["source_video_relative_path"] = row["source_video_relative_path"]
                prediction["split"] = row["split"]
                prediction["source_image_width_pixels"] = image.width
                prediction["source_image_height_pixels"] = image.height
                prediction["tile_origin"] = [left, top]
                prediction["inference_mode"] = "tiled"
                raw_predictions.append(prediction)
                minimum_x, minimum_y = ownership_bounds[(left, top)]
                if keep_prediction_owned_by_tile(prediction, minimum_x, minimum_y):
                    owned_predictions.append(prediction)
                else:
                    rejected_by_ownership += 1

    without_parts, elongated_removed = remove_elongated_parts(
        owned_predictions,
        aspect_ratio=4.0,
        containment=0.65,
        larger_area_ratio=1.50,
    )
    predictions, overlap_removed = suppress_overlapping_boxes(
        without_parts, config["nms_iou_threshold"]
    )
    for prediction in predictions:
        prediction["postprocessing_rules"] = [
            "tile_ownership",
            "remove_elongated_contained_parts",
            f"class_agnostic_iou_suppression_{config['nms_iou_threshold']:.2f}",
        ]
    statistics = {
        "frame_key": row["frame_key"],
        "source_image_width_pixels": image.width,
        "source_image_height_pixels": image.height,
        "tile_count": len(ownership_bounds),
        "raw_prediction_count": len(raw_predictions),
        "ownership_rejected": rejected_by_ownership,
        "elongated_parts_removed": elongated_removed,
        "overlap_boxes_removed": overlap_removed,
        "final_prediction_count": len(predictions),
    }
    return raw_predictions, predictions, statistics, image


def main() -> None:
    """Build, initialize, and test a compact detector on four representative frames."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Project-relative YAML configuration for one compact-student smoke test.",
    )
    arguments = parser.parse_args()
    config_path = arguments.config
    config = read_yaml(config_path)
    output_directory = resolve_project_path(config["artifact_directory"]) / config["run_id"]
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_directory / "config_snapshot.yaml")
    shutil.copyfile(resolve_project_path(config["model_yaml"]), output_directory / "model_snapshot.yaml")
    torch.cuda.reset_peak_memory_stats()

    started_at = time.perf_counter()
    student, source_checkpoint, transfer_report = load_student(config)
    initialized_weights_path = output_directory / "initialized_student_state_dict.pt"
    torch.save(student.model.state_dict(), initialized_weights_path)
    student.model.to(config["device"]).half().eval()
    model_loaded_at = time.perf_counter()
    manifest_path = resolve_project_path(config["input_manifest"])
    selected_rows = select_sample_frames(read_manifest(manifest_path), config["source_videos"])

    raw_predictions: list[dict] = []
    predictions: list[dict] = []
    frame_statistics: list[dict] = []
    for frame_number, row in enumerate(selected_rows, start=1):
        frame_started_at = time.perf_counter()
        raw_frame_predictions, frame_predictions, statistics, image = process_frame(student, row, config)
        statistics["elapsed_seconds"] = time.perf_counter() - frame_started_at
        raw_predictions.extend(raw_frame_predictions)
        predictions.extend(frame_predictions)
        frame_statistics.append(statistics)
        overlay_path = output_directory / "overlays_full_resolution" / Path(row["relative_image_path"]).name
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        draw_predictions(image, frame_predictions).save(
            overlay_path,
            quality=OVERLAY_QUALITY,
            subsampling=OVERLAY_SUBSAMPLING,
        )
        print(
            f"{frame_number:02d}/{len(selected_rows)} {row['frame_key']}: "
            f"{statistics['final_prediction_count']} boxes ({statistics['elapsed_seconds']:.2f}s)",
            flush=True,
        )

    finished_at = time.perf_counter()
    summary = {
        "run_kind": "untrained_compact_student_smoke_test",
        "warning": "Predictions come from an untrained one-class neck and head; they are not model-quality metrics.",
        "source_checkpoint_path": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": calculate_sha256(source_checkpoint),
        "student_model_yaml": str(resolve_project_path(config["model_yaml"]).resolve()),
        "student_initialization_state_dict": str(initialized_weights_path.resolve()),
        "input_size_pixels": config["input_size_pixels"],
        "tile_size_pixels": config["tile_size_pixels"],
        "tile_overlap": config["tile_overlap"],
        "confidence_threshold": config["confidence_threshold"],
        "nms_iou_threshold": config["nms_iou_threshold"],
        "sample_selection": config["sample_selection"],
        "source_videos": config["source_videos"],
        "sampled_frame_keys": [row["frame_key"] for row in selected_rows],
        "prediction_count": len(predictions),
        "prediction_count_by_class": dict(Counter(item["class_name"] for item in predictions)),
        "model_load_seconds": model_loaded_at - started_at,
        "total_seconds": finished_at - started_at,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "transfer_report": transfer_report,
    }
    write_json(output_directory / "transfer_report.json", transfer_report)
    write_json(output_directory / "raw_predictions.json", raw_predictions)
    write_json(output_directory / "predictions.json", predictions)
    write_json(output_directory / "frame_statistics.json", frame_statistics)
    write_json(output_directory / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
