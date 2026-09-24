"""Audit completed teacher proposals against the configured annotation rules."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from run_teacher_predictions import calculate_containment, calculate_iou


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIRECTORY / "configs" / "auto_label.yaml"


def parse_arguments() -> argparse.Namespace:
    """Read the optional configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a configuration path from the project root."""
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_DIRECTORY / path


def read_manifest(file_path: Path) -> dict[str, dict[str, str]]:
    """Index the canonical manifest by stable frame key."""
    with file_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    return {row["frame_key"]: row for row in rows}


def find_rule_violations(predictions: list[dict], rules: dict) -> Counter:
    """Count overlap patterns that should have been resolved."""
    violations = Counter()
    for first_index, first in enumerate(predictions):
        for second in predictions[first_index + 1 :]:
            if first["class_name"] != second["class_name"]:
                if (
                    calculate_iou(first["box_xyxy"], second["box_xyxy"])
                    > rules["cross_class_conflict_iou"]
                ):
                    violations["cross_class_conflict"] += 1
                continue
            if first["class_name"] == "car" and (
                calculate_iou(first["box_xyxy"], second["box_xyxy"])
                > rules["duplicate_car_iou"]
            ):
                violations["duplicate_car"] += 1
            if first["class_name"] == "truck" and (
                calculate_containment(first["box_xyxy"], second["box_xyxy"])
                > rules["duplicate_truck_containment"]
            ):
                violations["duplicate_truck"] += 1
    return violations


def audit_teacher(teacher_name: str, config: dict, manifest: dict) -> dict:
    """Audit one immutable teacher run and write its audit artifact."""
    run_directory = (
        resolve_project_path(config["artifact_directory"])
        / teacher_name
        / config["run_id"]
    )
    predictions = json.loads(
        (run_directory / "predictions.json").read_text(encoding="utf-8")
    )
    frame_statistics = json.loads(
        (run_directory / "frame_statistics.json").read_text(encoding="utf-8")
    )
    predictions_by_frame = defaultdict(list)
    invalid_boxes = 0
    invalid_source_sizes = 0
    final_classes = Counter()
    source_video_classes = defaultdict(Counter)
    for prediction in predictions:
        frame_key = prediction["frame_key"]
        predictions_by_frame[frame_key].append(prediction)
        final_classes[prediction["class_name"]] += 1
        source_video_classes[Path(prediction["source_video_relative_path"]).name][
            prediction["class_name"]
        ] += 1
        width = prediction["source_image_width_pixels"]
        height = prediction["source_image_height_pixels"]
        row = manifest[frame_key]
        if width != int(row["width"]) or height != int(row["height"]):
            invalid_source_sizes += 1
        box = prediction["box_xyxy"]
        if (
            len(box) != 4
            or not all(math.isfinite(value) for value in box)
            or box[0] < 0
            or box[1] < 0
            or box[2] > width
            or box[3] > height
            or box[0] >= box[2]
            or box[1] >= box[3]
        ):
            invalid_boxes += 1

    rule_violations = Counter()
    for frame_predictions in predictions_by_frame.values():
        rule_violations.update(
            find_rule_violations(frame_predictions, config["postprocessing"])
        )

    overlay_directory = run_directory / "overlays_full_resolution"
    overlay_files = sorted(overlay_directory.glob("*.jpg"))
    overlay_size_mismatches = 0
    for row in manifest.values():
        overlay_path = overlay_directory / f"{Path(row['relative_image_path']).stem}.jpg"
        if not overlay_path.is_file():
            overlay_size_mismatches += 1
            continue
        with Image.open(overlay_path) as overlay:
            if overlay.size != (int(row["width"]), int(row["height"])):
                overlay_size_mismatches += 1

    result = {
        "teacher": teacher_name,
        "run_id": config["run_id"],
        "manifest_frame_count": len(manifest),
        "frame_statistics_count": len(frame_statistics),
        "overlay_count": len(overlay_files),
        "prediction_count": len(predictions),
        "final_class_counts": dict(sorted(final_classes.items())),
        "source_video_class_counts": {
            video: dict(sorted(counts.items()))
            for video, counts in sorted(source_video_classes.items())
        },
        "invalid_boxes": invalid_boxes,
        "invalid_source_sizes": invalid_source_sizes,
        "overlay_size_mismatches": overlay_size_mismatches,
        "rule_violations": dict(rule_violations),
        "bus_present_in_final_predictions": final_classes["bus"] > 0,
        "truck_present_in_video_b": source_video_classes["train_B.mp4"]["truck"]
        > 0,
    }
    if (
        len(frame_statistics) != len(manifest)
        or len(overlay_files) != len(manifest)
        or invalid_boxes
        or invalid_source_sizes
        or overlay_size_mismatches
        or rule_violations
        or result["bus_present_in_final_predictions"]
        or result["truck_present_in_video_b"]
    ):
        raise ValueError(json.dumps(result, indent=2))
    (run_directory / "audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    """Audit both configured teachers."""
    arguments = parse_arguments()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    manifest = read_manifest(resolve_project_path(config["input_manifest"]))
    results = [
        audit_teacher(teacher["name"], config, manifest)
        for teacher in config["teachers"]
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
