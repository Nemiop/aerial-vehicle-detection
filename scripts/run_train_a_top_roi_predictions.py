"""Add a focused RF-DETR pass for distant vehicles in the top of video A."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from run_teacher_predictions import (
    apply_annotation_rules,
    draw_predictions,
    load_rfdetr_teacher,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIRECTORY / "configs" / "rfdetr_train_a_top_roi.yaml"


def parse_arguments() -> argparse.Namespace:
    """Read the optional focused-pass configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_DIRECTORY / path


def read_yaml(file_path: Path) -> dict:
    """Read one YAML document."""
    with file_path.open("r", encoding="utf-8") as input_file:
        return yaml.safe_load(input_file)


def read_json(file_path: Path):
    """Read one JSON document."""
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json(file_path: Path, value) -> None:
    """Write deterministic, readable JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_manifest(file_path: Path) -> list[dict[str, str]]:
    """Read the canonical frame manifest in declared order."""
    with file_path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))
    if not rows:
        raise ValueError("Frame manifest is empty")
    return rows


def index_predictions(predictions: list[dict]) -> dict[str, list[dict]]:
    """Group predictions by canonical frame key."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction["frame_key"]].append(prediction)
    return grouped


def is_point_inside_polygon(
    point_x: float, point_y: float, polygon: list[list[float]]
) -> bool:
    """Return whether a point is inside a polygon using ray casting."""
    inside = False
    previous_index = len(polygon) - 1
    for current_index, (current_x, current_y) in enumerate(polygon):
        previous_x, previous_y = polygon[previous_index]
        crosses_y = (current_y > point_y) != (previous_y > point_y)
        if crosses_y:
            crossing_x = (previous_x - current_x) * (point_y - current_y) / (
                previous_y - current_y
            ) + current_x
            if point_x < crossing_x:
                inside = not inside
        previous_index = current_index
    return inside


def get_rejection_reason(prediction: dict, config: dict) -> str | None:
    """Explain why a focused prediction does not fit the known road geometry."""
    left, top, right, bottom = prediction["box_xyxy"]
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    if not is_point_inside_polygon(
        center_x, center_y, config["acceptance_polygon_pixels"]
    ):
        return "center_outside_acceptance_polygon"

    width = right - left
    height = bottom - top
    size_filter = config["box_size_filter_pixels"]
    if width < size_filter["minimum_width"]:
        return "box_too_narrow"
    if height < size_filter["minimum_height"]:
        return "box_too_short"
    if width > size_filter["maximum_width"]:
        return "box_too_wide"
    if height > size_filter["maximum_height"]:
        return "box_too_tall"
    return None


def add_frame_metadata(
    prediction: dict,
    row: dict[str, str],
    image: Image.Image,
    inference_mode: str,
) -> dict:
    """Attach source-image coordinates and provenance to one prediction."""
    item = dict(prediction)
    item.update(
        {
            "frame_key": row["frame_key"],
            "relative_image_path": row["relative_image_path"],
            "source_video_relative_path": row["source_video_relative_path"],
            "split": row["split"],
            "source_image_width_pixels": image.width,
            "source_image_height_pixels": image.height,
            "inference_mode": inference_mode,
        }
    )
    return item


def predict_focused_region(image: Image.Image, predict, config: dict) -> list[dict]:
    """Run RF-DETR on one upper-road crop and restore full-frame coordinates."""
    left, top, right, bottom = config["region_xyxy_pixels"]
    crop = image.crop((left, top, right, bottom))
    local_predictions = predict(
        crop,
        config["model_resolution_pixels"],
        config["confidence_threshold"],
        1.0,
    )
    predictions = []
    for local_prediction in local_predictions:
        item = dict(local_prediction)
        local_box = item["box_xyxy"]
        item["box_xyxy"] = [
            local_box[0] + left,
            local_box[1] + top,
            local_box[2] + left,
            local_box[3] + top,
        ]
        item["focused_region_xyxy"] = [left, top, right, bottom]
        predictions.append(item)
    return predictions


def main() -> None:
    """Create an augmented RF-DETR proposal set without changing other videos."""
    arguments = parse_arguments()
    config = read_yaml(arguments.config)
    manifest_path = resolve_project_path(config["input_manifest"])
    base_predictions_path = resolve_project_path(config["base_predictions"])
    output_directory = resolve_project_path(config["output_directory"])
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Focused-pass output already exists: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(arguments.config, output_directory / "config_snapshot.yaml")

    rows = read_manifest(manifest_path)
    base_predictions = read_json(base_predictions_path)
    base_by_frame = index_predictions(base_predictions)
    target_video_name = config["target_source_video"]
    target_rows = [
        row
        for row in rows
        if Path(row["source_video_relative_path"]).name == target_video_name
    ]
    if not target_rows:
        raise ValueError(f"No manifest frames found for {target_video_name}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    use_fp16 = config["precision"] == "fp16"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started_at = time.perf_counter()
    model, predict, environment = load_rfdetr_teacher(
        config["checkpoint"], config["device"], use_fp16
    )
    model_loaded_at = time.perf_counter()

    augmented_by_frame = {
        frame_key: [dict(item) for item in predictions]
        for frame_key, predictions in base_by_frame.items()
    }
    focused_raw_records = []
    focused_accepted_records = []
    target_frame_statistics = {}
    rejection_totals: Counter[str] = Counter()

    for frame_number, row in enumerate(target_rows, start=1):
        image_path = resolve_project_path(row["relative_image_path"])
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        frame_started_at = time.perf_counter()
        raw_predictions = predict_focused_region(image, predict, config)
        accepted_predictions = []
        frame_rejections: Counter[str] = Counter()
        for prediction in raw_predictions:
            raw_record = add_frame_metadata(
                prediction, row, image, "focused_train_A_top_roi_raw"
            )
            focused_raw_records.append(raw_record)
            rejection_reason = get_rejection_reason(prediction, config)
            if rejection_reason is None:
                accepted_predictions.append(prediction)
                focused_accepted_records.append(
                    add_frame_metadata(
                        prediction, row, image, "focused_train_A_top_roi"
                    )
                )
            else:
                frame_rejections[rejection_reason] += 1
                rejection_totals[rejection_reason] += 1

        combined_predictions = [
            dict(item) for item in base_by_frame.get(row["frame_key"], [])
        ] + accepted_predictions
        combined_predictions, rule_statistics = apply_annotation_rules(
            combined_predictions, row, config["postprocessing"]
        )
        augmented_predictions = [
            add_frame_metadata(
                prediction, row, image, "tiled_plus_focused_train_A_top_roi"
            )
            for prediction in combined_predictions
        ]
        augmented_by_frame[row["frame_key"]] = augmented_predictions

        overlay_path = (
            output_directory
            / "overlays_full_resolution"
            / f"{image_path.stem}.jpg"
        )
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        draw_predictions(image, augmented_predictions).save(
            overlay_path,
            quality=config["overlay"]["jpeg_quality"],
            subsampling=config["overlay"]["jpeg_subsampling"],
        )
        elapsed_seconds = time.perf_counter() - frame_started_at
        target_frame_statistics[row["frame_key"]] = {
                "frame_key": row["frame_key"],
                "focused_pass_applied": True,
                "base_prediction_count": len(base_by_frame.get(row["frame_key"], [])),
                "focused_raw_prediction_count": len(raw_predictions),
                "focused_accepted_prediction_count": len(accepted_predictions),
                "augmented_prediction_count": len(augmented_predictions),
                "rejections": dict(frame_rejections),
                "postprocessing": rule_statistics,
                "elapsed_seconds": elapsed_seconds,
            }
        print(
            f"{frame_number:02d}/{len(target_rows)} {row['frame_key']}: "
            f"{len(raw_predictions)} focused raw, "
            f"{len(accepted_predictions)} accepted, "
            f"{len(augmented_predictions)} combined",
            flush=True,
        )

    augmented_records = []
    frame_statistics = []
    base_overlay_directory = base_predictions_path.parent / "overlays_full_resolution"
    for row in rows:
        frame_key = row["frame_key"]
        frame_predictions = augmented_by_frame.get(frame_key, [])
        augmented_records.extend(frame_predictions)
        if frame_key in target_frame_statistics:
            frame_statistics.append(target_frame_statistics[frame_key])
            continue

        frame_statistics.append(
            {
                "frame_key": frame_key,
                "focused_pass_applied": False,
                "base_prediction_count": len(frame_predictions),
                "focused_raw_prediction_count": 0,
                "focused_accepted_prediction_count": 0,
                "augmented_prediction_count": len(frame_predictions),
                "rejections": {},
                "postprocessing": {},
                "elapsed_seconds": 0.0,
            }
        )
        overlay_name = f"{Path(row['relative_image_path']).stem}.jpg"
        source_overlay_path = base_overlay_directory / overlay_name
        destination_overlay_path = (
            output_directory / "overlays_full_resolution" / overlay_name
        )
        if not source_overlay_path.is_file():
            raise FileNotFoundError(f"Base overlay is missing: {source_overlay_path}")
        destination_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_overlay_path, destination_overlay_path)

    finished_at = time.perf_counter()
    summary = {
        "run_id": config["run_id"],
        "teacher": "rfdetr_large",
        "target_source_video": target_video_name,
        "target_frame_count": len(target_rows),
        "untouched_frame_count": len(rows) - len(target_rows),
        "base_prediction_count": len(base_predictions),
        "focused_raw_prediction_count": len(focused_raw_records),
        "focused_accepted_prediction_count": len(focused_accepted_records),
        "augmented_prediction_count": len(augmented_records),
        "rejection_totals": dict(rejection_totals),
        "load_seconds": model_loaded_at - started_at,
        "total_seconds": finished_at - started_at,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "environment": environment,
    }
    write_json(output_directory / "focused_raw_predictions.json", focused_raw_records)
    write_json(
        output_directory / "focused_accepted_predictions.json",
        focused_accepted_records,
    )
    write_json(output_directory / "predictions.json", augmented_records)
    write_json(output_directory / "frame_statistics.json", frame_statistics)
    write_json(output_directory / "summary.json", summary)
    del model
    torch.cuda.empty_cache()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
