"""Validate prediction-tree geometry, class conflicts, provenance, and overlays."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import yaml
from PIL import Image

from prediction_tree import (
    calculate_box_area,
    calculate_iou,
    should_keep_large_trucks_separate,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIRECTORY / "configs" / "merge_predictions.yaml"
FLOAT_TOLERANCE = 1e-5


def parse_arguments() -> argparse.Namespace:
    """Read the optional merge configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configuration path from the project root."""
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_DIRECTORY / path


def boxes_are_equal(first: list[float], second: list[float]) -> bool:
    """Compare boxes with a small floating-point tolerance."""
    return all(abs(left - right) <= FLOAT_TOLERANCE for left, right in zip(first, second))


def main() -> None:
    """Audit the configured prediction tree and write an audit report."""
    arguments = parse_arguments()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    output_directory = resolve_project_path(config["output_directory"])
    predictions = json.loads(
        (output_directory / "predictions.json").read_text(encoding="utf-8")
    )
    frame_statistics = json.loads(
        (output_directory / "frame_statistics.json").read_text(encoding="utf-8")
    )
    with resolve_project_path(config["input_manifest"]).open(
        "r", encoding="utf-8-sig", newline=""
    ) as input_file:
        manifest = {row["frame_key"]: row for row in csv.DictReader(input_file)}

    violations = Counter()
    prediction_ids = set()
    for prediction in predictions:
        prediction_id = prediction["prediction_id"]
        if prediction_id in prediction_ids:
            violations["duplicate_prediction_id"] += 1
        prediction_ids.add(prediction_id)
        row = manifest[prediction["frame_key"]]
        width = prediction["source_image_width_pixels"]
        height = prediction["source_image_height_pixels"]
        if width != int(row["width"]) or height != int(row["height"]):
            violations["source_size_mismatch"] += 1
        box = prediction["box_xyxy"]
        if (
            not all(math.isfinite(value) for value in box)
            or box[0] < 0
            or box[1] < 0
            or box[2] > width
            or box[3] > height
            or box[0] >= box[2]
            or box[1] >= box[3]
        ):
            violations["invalid_box"] += 1

        teachers = prediction["teacher_predictions"]
        if prediction["teacher_count"] != len(teachers):
            violations["teacher_count_mismatch"] += 1
        if prediction["teacher_count"] == 1:
            teacher_name, teacher_prediction = next(iter(teachers.items()))
            if prediction["class_name"] != teacher_prediction["class_name"]:
                violations["unmatched_class_mismatch"] += 1
            if not boxes_are_equal(box, teacher_prediction["box_xyxy"]):
                violations["unmatched_box_mismatch"] += 1
            if (
                "component_predictions" not in prediction
                and prediction["merge_status"] != f"{teacher_name}_only"
            ):
                violations["unmatched_status_mismatch"] += 1
            if "component_predictions" not in prediction:
                continue

        if prediction["teacher_count"] == 2 and set(teachers) != {
            "yolo26x",
            "rfdetr_large",
        }:
            violations["matched_teacher_names"] += 1
            continue
        if prediction["teacher_count"] == 2:
            yolo = teachers["yolo26x"]
            rfdetr = teachers["rfdetr_large"]
            matched_iou = calculate_iou(yolo["box_xyxy"], rfdetr["box_xyxy"])
            if abs(matched_iou - prediction["matched_iou"]) > FLOAT_TOLERANCE:
                violations["matched_iou_mismatch"] += 1
            class_conflict = yolo["class_name"] != rfdetr["class_name"]
            expected_class = (
                config["class_conflict"]["output_class_name"]
                if class_conflict
                else yolo["class_name"]
            )
            if (
                prediction["has_class_conflict"] != class_conflict
                or prediction["class_name"] != expected_class
            ):
                violations["class_conflict_mismatch"] += 1

            if matched_iou >= config["matching"]["near_identical_iou"]:
                larger = (
                    yolo["box_xyxy"]
                    if calculate_box_area(yolo["box_xyxy"])
                    >= calculate_box_area(rfdetr["box_xyxy"])
                    else rfdetr["box_xyxy"]
                )
                if not boxes_are_equal(box, larger):
                    violations["near_identical_not_larger"] += 1
            else:
                expected_box = [
                    (yolo_value + rfdetr_value) / 2.0
                    for yolo_value, rfdetr_value in zip(
                        yolo["box_xyxy"], rfdetr["box_xyxy"]
                    )
                ]
                if not boxes_are_equal(box, expected_box):
                    violations["moderate_overlap_not_fused"] += 1

        if "component_predictions" in prediction:
            components = prediction["component_predictions"]
            if prediction["component_count"] != len(components):
                violations["component_count_mismatch"] += 1
            recommended_box = [
                min(item["box_xyxy"][0] for item in components),
                min(item["box_xyxy"][1] for item in components),
                max(item["box_xyxy"][2] for item in components),
                max(item["box_xyxy"][3] for item in components),
            ]
            if not boxes_are_equal(
                prediction["recommended_box_xyxy"], recommended_box
            ):
                violations["recommended_box_mismatch"] += 1
            if config.get("truck_geometry", {}).get("enabled", False):
                for first_component, second_component in combinations(components, 2):
                    if should_keep_large_trucks_separate(
                        first_component["box_xyxy"],
                        second_component["box_xyxy"],
                        config["truck_geometry"],
                    ):
                        violations["distinct_large_trucks_absorbed"] += 1

    overlay_directory = output_directory / "overlays_full_resolution"
    overlays = list(overlay_directory.glob("*.jpg"))
    for row in manifest.values():
        overlay_path = overlay_directory / f"{Path(row['relative_image_path']).stem}.jpg"
        if not overlay_path.is_file():
            violations["missing_overlay"] += 1
            continue
        with Image.open(overlay_path) as overlay:
            if overlay.size != (int(row["width"]), int(row["height"])):
                violations["overlay_size_mismatch"] += 1

    result = {
        "frame_count": len(manifest),
        "frame_statistics_count": len(frame_statistics),
        "prediction_count": len(predictions),
        "unique_prediction_id_count": len(prediction_ids),
        "overlay_count": len(overlays),
        "class_counts": dict(
            sorted(Counter(item["class_name"] for item in predictions).items())
        ),
        "violations": dict(violations),
    }
    if (
        len(frame_statistics) != len(manifest)
        or len(overlays) != len(manifest)
        or violations
    ):
        raise ValueError(json.dumps(result, indent=2))
    (output_directory / "audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
