"""Generate tiled teacher proposals and full-resolution review overlays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIRECTORY / "configs" / "auto_label.yaml"
SOURCE_CLASSES = ("car", "motorcycle", "bus", "truck")
OUTPUT_CLASSES = ("car", "motorcycle", "truck")
CLASS_COLORS = {
    "car": "#19c37d",
    "motorcycle": "#7c3aed",
    "bus": "#f59e0b",
    "truck": "#ef4444",
}
ULTRALYTICS_CONFIG_DIRECTORY = PROJECT_DIRECTORY / "My Artifacts" / "ultralytics"
COCO_CATEGORY_ID_TO_NAME = {
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}


def parse_arguments() -> argparse.Namespace:
    """Read the teacher name and optional configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher", required=True, choices=("yolo26x", "rfdetr_large")
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--source-video",
        dest="source_videos",
        action="append",
        help=(
            "Process only this source video filename. Repeat the option to select "
            "several videos."
        ),
    )
    return parser.parse_args()


def read_yaml(file_path: Path) -> dict:
    """Read a YAML configuration file."""
    with file_path.open("r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file)


def read_manifest(file_path: Path) -> list[dict[str, str]]:
    """Read and validate the canonical frame manifest in declared order."""
    with file_path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if not rows:
        raise ValueError("Frame manifest is empty")
    if len({row["frame_key"] for row in rows}) != len(rows):
        raise ValueError("Frame manifest contains duplicate frame_key values")
    unexpected_splits = {row["split"] for row in rows} - {"train", "validation"}
    if unexpected_splits:
        raise ValueError(f"Unexpected dataset splits: {sorted(unexpected_splits)}")
    return rows


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configuration path relative to the project directory."""
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_DIRECTORY / path


def calculate_sha256(file_path: Path) -> str:
    """Calculate a file digest using bounded memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_teacher_config(config: dict, teacher_name: str) -> dict:
    """Return one named teacher configuration."""
    for teacher in config["teachers"]:
        if teacher["name"] == teacher_name:
            return teacher
    raise KeyError(f"Teacher is missing from config: {teacher_name}")


def get_tile_origins(length: int, tile_size: int, overlap: float) -> list[int]:
    """Return origins that cover an axis and align the last tile to its end."""
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1 - overlap)))
    origins = list(range(0, length - tile_size + 1, stride))
    last_origin = length - tile_size
    if origins[-1] != last_origin:
        origins.append(last_origin)
    return origins


def get_tiles(image: Image.Image, tile_size: int, overlap: float):
    """Yield overlapping image tiles and their source-image origins."""
    for top in get_tile_origins(image.height, tile_size, overlap):
        for left in get_tile_origins(image.width, tile_size, overlap):
            right = min(left + tile_size, image.width)
            bottom = min(top + tile_size, image.height)
            yield image.crop((left, top, right, bottom)), left, top


def calculate_iou(box: list[float], other_box: list[float]) -> float:
    """Calculate IoU for two xyxy boxes."""
    left = max(box[0], other_box[0])
    top = max(box[1], other_box[1])
    right = min(box[2], other_box[2])
    bottom = min(box[3], other_box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    other_area = max(0.0, other_box[2] - other_box[0]) * max(
        0.0, other_box[3] - other_box[1]
    )
    union = box_area + other_area - intersection
    return intersection / union if union > 0 else 0.0


def calculate_box_area(box: list[float]) -> float:
    """Return the positive area of an xyxy box."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def calculate_containment(box: list[float], other_box: list[float]) -> float:
    """Return intersection divided by the smaller box area."""
    left = max(box[0], other_box[0])
    top = max(box[1], other_box[1])
    right = min(box[2], other_box[2])
    bottom = min(box[3], other_box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    smaller_area = min(calculate_box_area(box), calculate_box_area(other_box))
    return intersection / smaller_area if smaller_area > 0 else 0.0


def suppress_duplicate_boxes(predictions: list[dict], iou_threshold: float) -> list[dict]:
    """Apply class-aware greedy NMS after converting tile coordinates."""
    kept_predictions: list[dict] = []
    for class_name in OUTPUT_CLASSES:
        candidates = sorted(
            (item for item in predictions if item["class_name"] == class_name),
            key=lambda item: item["confidence"],
            reverse=True,
        )
        while candidates:
            best = candidates.pop(0)
            kept_predictions.append(best)
            candidates = [
                item
                for item in candidates
                if calculate_iou(best["box_xyxy"], item["box_xyxy"])
                < iou_threshold
            ]
    return sorted(kept_predictions, key=lambda item: item["confidence"], reverse=True)


def normalize_bus_predictions(predictions: list[dict], enabled: bool) -> tuple[list[dict], int]:
    """Convert bus proposals to truck while preserving the source class."""
    normalized = []
    converted_count = 0
    for prediction in predictions:
        item = dict(prediction)
        item["source_class_name"] = item["class_name"]
        if enabled and item["class_name"] == "bus":
            item["class_name"] = "truck"
            converted_count += 1
        normalized.append(item)
    return normalized, converted_count


def resolve_cross_class_conflicts(
    predictions: list[dict], iou_threshold: float
) -> tuple[list[dict], int]:
    """For highly overlapping class conflicts keep the smaller box as car."""
    remaining = list(predictions)
    removed_count = 0
    while True:
        conflict = None
        for first_index, first in enumerate(remaining):
            for second_index in range(first_index + 1, len(remaining)):
                second = remaining[second_index]
                if first["class_name"] == second["class_name"]:
                    continue
                if calculate_iou(first["box_xyxy"], second["box_xyxy"]) > iou_threshold:
                    conflict = (first_index, second_index)
                    break
            if conflict:
                break
        if conflict is None:
            return remaining, removed_count
        first_index, second_index = conflict
        first_area = calculate_box_area(remaining[first_index]["box_xyxy"])
        second_area = calculate_box_area(remaining[second_index]["box_xyxy"])
        keep_index, remove_index = (
            (first_index, second_index)
            if first_area <= second_area
            else (second_index, first_index)
        )
        remaining[keep_index]["class_name"] = "car"
        remaining[keep_index]["postprocessing_rule"] = "cross_class_smaller_is_car"
        remaining.pop(remove_index)
        removed_count += 1


def keep_larger_overlapping_boxes(
    predictions: list[dict],
    class_name: str,
    overlap_threshold: float,
    use_containment: bool,
) -> tuple[list[dict], int]:
    """Remove duplicate proposals of one class, retaining the larger box."""
    remaining = list(predictions)
    removed_count = 0
    while True:
        duplicate = None
        for first_index, first in enumerate(remaining):
            if first["class_name"] != class_name:
                continue
            for second_index in range(first_index + 1, len(remaining)):
                second = remaining[second_index]
                if second["class_name"] != class_name:
                    continue
                overlap = (
                    calculate_containment(first["box_xyxy"], second["box_xyxy"])
                    if use_containment
                    else calculate_iou(first["box_xyxy"], second["box_xyxy"])
                )
                if overlap > overlap_threshold:
                    duplicate = (first_index, second_index)
                    break
            if duplicate:
                break
        if duplicate is None:
            return remaining, removed_count
        first_index, second_index = duplicate
        first_area = calculate_box_area(remaining[first_index]["box_xyxy"])
        second_area = calculate_box_area(remaining[second_index]["box_xyxy"])
        remove_index = second_index if first_area >= second_area else first_index
        remaining.pop(remove_index)
        removed_count += 1


def apply_annotation_rules(
    predictions: list[dict], row: dict[str, str], rules: dict
) -> tuple[list[dict], dict[str, int]]:
    """Apply the agreed scene and overlap rules to one full-size frame."""
    statistics = {
        "bus_converted_to_truck": 0,
        "video_specific_trucks_removed": 0,
        "cross_class_conflicts_removed": 0,
        "duplicate_cars_removed": 0,
        "duplicate_trucks_removed": 0,
        "tile_nms_removed": 0,
    }
    predictions, statistics["bus_converted_to_truck"] = normalize_bus_predictions(
        predictions, rules["convert_bus_to_truck"]
    )
    source_video_name = Path(row["source_video_relative_path"]).name
    if source_video_name in set(rules["remove_truck_from_source_videos"]):
        before_count = len(predictions)
        predictions = [
            item for item in predictions if item["class_name"] != "truck"
        ]
        statistics["video_specific_trucks_removed"] = before_count - len(predictions)
    predictions, statistics["cross_class_conflicts_removed"] = (
        resolve_cross_class_conflicts(
            predictions, rules["cross_class_conflict_iou"]
        )
    )
    predictions, statistics["duplicate_cars_removed"] = keep_larger_overlapping_boxes(
        predictions,
        class_name="car",
        overlap_threshold=rules["duplicate_car_iou"],
        use_containment=False,
    )
    predictions, statistics["duplicate_trucks_removed"] = keep_larger_overlapping_boxes(
        predictions,
        class_name="truck",
        overlap_threshold=rules["duplicate_truck_containment"],
        use_containment=True,
    )
    before_nms_count = len(predictions)
    predictions = suppress_duplicate_boxes(predictions, rules["tile_nms_iou"])
    statistics["tile_nms_removed"] = before_nms_count - len(predictions)
    return predictions, statistics


def load_yolo_teacher(checkpoint: str, device: str, use_fp16: bool):
    """Load a YOLO teacher and validate its class vocabulary."""
    os.environ["YOLO_CONFIG_DIR"] = str(ULTRALYTICS_CONFIG_DIRECTORY)
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    LOGGER.setLevel("ERROR")

    model = YOLO(checkpoint)
    available_classes = set(model.names.values())
    missing_classes = set(SOURCE_CLASSES) - available_classes
    if missing_classes:
        raise ValueError(f"YOLO checkpoint is missing classes: {sorted(missing_classes)}")

    class_ids = [
        class_id
        for class_id, class_name in model.names.items()
        if class_name in SOURCE_CLASSES
    ]

    def predict(image: Image.Image, input_size: int, confidence: float, iou: float):
        result = model.predict(
            source=image,
            imgsz=input_size,
            conf=confidence,
            iou=iou,
            classes=class_ids,
            device=device,
            half=use_fp16,
            batch=1,
            nms=True,
            verbose=False,
        )[0]
        predictions = []
        if result.boxes is None:
            return predictions
        for box, score, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            predictions.append(
                {
                    "box_xyxy": [float(value) for value in box],
                    "confidence": float(score),
                    "class_name": model.names[int(class_id)],
                    "source_class_id": int(class_id),
                }
            )
        return predictions

    metadata = {
        "library": "ultralytics",
        "library_version": ultralytics.__version__,
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "class_names": model.names,
    }
    return model, predict, metadata


def load_rfdetr_teacher(checkpoint: str, device: str, use_fp16: bool):
    """Load RF-DETR Large and normalize its sparse COCO category IDs."""
    import torch
    from rfdetr import RFDETRLarge

    if device != "cuda:0":
        raise ValueError("RF-DETR adapter currently expects cuda:0")
    model = RFDETRLarge(resolution=704, amp=False)
    inference_dtype = torch.float16 if use_fp16 else torch.float32
    model.inference(
        compile=False,
        batch_size=1,
        dtype=inference_dtype,
        inplace=True,
    )

    default_checkpoint_path = (
        Path.home() / ".roboflow" / "models" / "rf-detr-large-2026.pth"
    )

    def predict(image: Image.Image, input_size: int, confidence: float, iou: float):
        del input_size, iou
        detections = model.predict(image, threshold=confidence)
        predictions = []
        for box, score, class_id in zip(
            detections.xyxy.tolist(),
            detections.confidence.tolist(),
            detections.class_id.tolist(),
        ):
            class_id = int(class_id)
            class_name = COCO_CATEGORY_ID_TO_NAME.get(class_id)
            if class_name is None:
                continue
            predictions.append(
                {
                    "box_xyxy": [float(value) for value in box],
                    "confidence": float(score),
                    "class_name": class_name,
                    "source_class_id": class_id,
                }
            )
        return predictions

    metadata = {
        "library": "rfdetr",
        "library_version": importlib.metadata.version("rfdetr"),
        "checkpoint_path": (
            str(default_checkpoint_path.resolve())
            if default_checkpoint_path.is_file()
            else checkpoint
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "class_names": COCO_CATEGORY_ID_TO_NAME,
        "category_id_convention": "raw sparse COCO category IDs",
    }
    return model, predict, metadata


def offset_predictions(predictions: list[dict], left: int, top: int) -> list[dict]:
    """Move tile predictions into source-image coordinates."""
    offset_items = []
    for prediction in predictions:
        box = prediction["box_xyxy"]
        offset_item = dict(prediction)
        offset_item["box_xyxy"] = [
            box[0] + left,
            box[1] + top,
            box[2] + left,
            box[3] + top,
        ]
        offset_item["tile_origin"] = [left, top]
        offset_items.append(offset_item)
    return offset_items


def draw_predictions(image: Image.Image, predictions: list[dict]) -> Image.Image:
    """Draw labelled boxes on a copy of the source frame."""
    overlay = image.copy()
    drawing = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    line_width = max(2, round(max(image.size) / 1000))
    for prediction in predictions:
        box = prediction["box_xyxy"]
        class_name = prediction["class_name"]
        color = CLASS_COLORS[class_name]
        drawing.rectangle(box, outline=color, width=line_width)
        label = f"{class_name} {prediction['confidence']:.2f}"
        label_box = drawing.textbbox((box[0], box[1]), label, font=font)
        drawing.rectangle(label_box, fill=color)
        drawing.text((box[0], box[1]), label, fill="white", font=font)
    return overlay


def run_tiled_inference(
    image: Image.Image,
    predict,
    teacher_config: dict,
) -> list[dict]:
    """Run a teacher on tiles sized for that teacher's native input."""
    tiling = teacher_config["tiling"]
    tile_predictions: list[dict] = []
    for tile, left, top in get_tiles(
        image, tiling["tile_size_pixels"], tiling["tile_overlap"]
    ):
        predictions = predict(
            tile,
            teacher_config["input_size_pixels"],
            teacher_config["confidence_threshold"],
            1.0,
        )
        tile_predictions.extend(offset_predictions(predictions, left, top))
    return tile_predictions


def write_json(file_path: Path, value) -> None:
    """Write deterministic, readable JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run and record one teacher on every frame in the canonical manifest."""
    arguments = parse_arguments()
    config = read_yaml(arguments.config)
    teacher_config = get_teacher_config(config, arguments.teacher)
    manifest_path = resolve_project_path(config["input_manifest"])
    manifest_rows = read_manifest(manifest_path)
    selected_source_videos = None
    if arguments.source_videos:
        selected_source_videos = set(arguments.source_videos)
        manifest_rows = [
            row
            for row in manifest_rows
            if Path(row["source_video_relative_path"]).name in selected_source_videos
        ]
        if not manifest_rows:
            raise ValueError(
                "The requested source videos are absent from the frame manifest: "
                f"{sorted(selected_source_videos)}"
            )
    artifact_root = resolve_project_path(config["artifact_directory"])
    output_directory = artifact_root / arguments.teacher / config["run_id"]
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Teacher output already exists: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = output_directory / "config_snapshot.yaml"
    shutil.copyfile(arguments.config, config_snapshot_path)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    use_fp16 = teacher_config["precision"] == "fp16"
    loader = (
        load_yolo_teacher if arguments.teacher == "yolo26x" else load_rfdetr_teacher
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    model, predict, environment = loader(
        teacher_config["checkpoint"], config["device"], use_fp16
    )
    model_loaded_at = time.perf_counter()

    raw_output_records = []
    output_records = []
    frame_statistics = []
    postprocessing_totals = {
        "bus_converted_to_truck": 0,
        "video_specific_trucks_removed": 0,
        "cross_class_conflicts_removed": 0,
        "duplicate_cars_removed": 0,
        "duplicate_trucks_removed": 0,
        "tile_nms_removed": 0,
    }
    image_count = 0
    for row in manifest_rows:
        image_path = resolve_project_path(row["relative_image_path"])
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        frame_started_at = time.perf_counter()
        raw_predictions = run_tiled_inference(image, predict, teacher_config)
        for prediction in raw_predictions:
            raw_output_records.append(
                {
                    **prediction,
                    "frame_key": row["frame_key"],
                    "relative_image_path": row["relative_image_path"],
                    "source_video_relative_path": row["source_video_relative_path"],
                    "split": row["split"],
                    "source_image_width_pixels": image.width,
                    "source_image_height_pixels": image.height,
                    "inference_mode": "tiled",
                }
            )
        predictions, rule_statistics = apply_annotation_rules(
            raw_predictions, row, config["postprocessing"]
        )
        elapsed_seconds = time.perf_counter() - frame_started_at
        for key, value in rule_statistics.items():
            postprocessing_totals[key] += value
        for prediction in predictions:
            prediction.update(
                {
                    "frame_key": row["frame_key"],
                    "relative_image_path": row["relative_image_path"],
                    "source_video_relative_path": row["source_video_relative_path"],
                    "split": row["split"],
                    "source_image_width_pixels": image.width,
                    "source_image_height_pixels": image.height,
                    "inference_mode": "tiled",
                }
            )
        output_records.extend(predictions)
        frame_statistics.append(
            {
                "frame_key": row["frame_key"],
                "relative_image_path": row["relative_image_path"],
                "source_image_width_pixels": image.width,
                "source_image_height_pixels": image.height,
                "raw_prediction_count": len(raw_predictions),
                "final_prediction_count": len(predictions),
                "elapsed_seconds": elapsed_seconds,
                "postprocessing": rule_statistics,
            }
        )
        overlay_path = (
            output_directory
            / "overlays_full_resolution"
            / f"{Path(image_path).stem}.jpg"
        )
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        draw_predictions(image, predictions).save(
            overlay_path, quality=95, subsampling=0
        )
        print(
            f"{image_count + 1:03d} {row['frame_key']}: "
            f"{len(raw_predictions)} raw -> {len(predictions)} boxes "
            f"in {elapsed_seconds:.2f}s",
            flush=True,
        )
        image_count += 1

    finished_at = time.perf_counter()
    checkpoint_path = Path(
        environment.get("checkpoint_path", teacher_config["checkpoint"])
    )
    summary = {
        "teacher": arguments.teacher,
        "checkpoint": teacher_config["checkpoint"],
        "checkpoint_sha256": (
            calculate_sha256(checkpoint_path) if checkpoint_path.is_file() else None
        ),
        "config_path": str(config_snapshot_path.resolve()),
        "config_sha256": calculate_sha256(config_snapshot_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": calculate_sha256(manifest_path),
        "source_videos": sorted(
            selected_source_videos
            if selected_source_videos is not None
            else {
                Path(row["source_video_relative_path"]).name
                for row in manifest_rows
            }
        ),
        "image_count": image_count,
        "inference_mode": "tiled",
        "tiling": teacher_config["tiling"],
        "source_images_preserved_at_original_resolution": True,
        "raw_prediction_count": len(raw_output_records),
        "prediction_count": len(output_records),
        "postprocessing": postprocessing_totals,
        "load_seconds": model_loaded_at - started_at,
        "total_seconds": finished_at - started_at,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment": environment,
    }
    write_json(output_directory / "raw_predictions.json", raw_output_records)
    write_json(output_directory / "predictions.json", output_records)
    write_json(output_directory / "frame_statistics.json", frame_statistics)
    write_json(output_directory / "summary.json", summary)
    del model
    torch.cuda.empty_cache()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
