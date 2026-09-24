"""Run YOLO on train_D with ownership-aware tiled postprocessing."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from PIL import Image

try:
    from run_teacher_predictions import (
        PROJECT_DIRECTORY,
        calculate_box_area,
        calculate_containment,
        calculate_iou,
        calculate_sha256,
        draw_predictions,
        get_tile_origins,
        load_yolo_teacher,
        normalize_bus_predictions,
        read_manifest,
        resolve_project_path,
        write_json,
    )
except ModuleNotFoundError:
    from scripts.run_teacher_predictions import (
        PROJECT_DIRECTORY,
        calculate_box_area,
        calculate_containment,
        calculate_iou,
        calculate_sha256,
        draw_predictions,
        get_tile_origins,
        load_yolo_teacher,
        normalize_bus_predictions,
        read_manifest,
        resolve_project_path,
        write_json,
    )


CONFIG_PATH = PROJECT_DIRECTORY / "configs" / "yolo_train_d_deduplicated.yaml"
TARGET_VIDEO_NAME = "train_D.mp4"
TARGET_VIDEO_KEY = "train_D"


def read_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def get_target_frames(config: dict) -> list[dict[str, str]]:
    manifest_path = resolve_project_path(config["input_manifest"])
    rows = [
        row
        for row in read_manifest(manifest_path)
        if Path(row["source_video_relative_path"]).name == TARGET_VIDEO_NAME
    ]
    if not rows:
        raise ValueError(f"No frames found for {TARGET_VIDEO_NAME}")
    return rows


def get_tile_ownership_bounds(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: float,
) -> dict[tuple[int, int], tuple[float, float]]:
    """Return the first accepted center coordinate for every non-primary tile."""
    horizontal_origins = get_tile_origins(image_width, tile_size, overlap)
    vertical_origins = get_tile_origins(image_height, tile_size, overlap)
    bounds: dict[tuple[int, int], tuple[float, float]] = {}

    for top in vertical_origins:
        for left in horizontal_origins:
            minimum_x = float("-inf")
            minimum_y = float("-inf")
            if left != horizontal_origins[0]:
                previous_left = max(
                    origin for origin in horizontal_origins if origin < left
                )
                overlap_end = min(previous_left + tile_size, left + tile_size)
                minimum_x = (left + overlap_end) / 2.0
            if top != vertical_origins[0]:
                previous_top = max(origin for origin in vertical_origins if origin < top)
                overlap_end = min(previous_top + tile_size, top + tile_size)
                minimum_y = (top + overlap_end) / 2.0
            bounds[(left, top)] = (minimum_x, minimum_y)
    return bounds


def keep_prediction_owned_by_tile(
    prediction: dict,
    minimum_center_x: float,
    minimum_center_y: float,
) -> bool:
    box = prediction["box_xyxy"]
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    return center_x >= minimum_center_x and center_y >= minimum_center_y


def calculate_aspect_ratio(box: list[float]) -> float:
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    if min(width, height) <= 0.0:
        return float("inf")
    return max(width / height, height / width)


def is_elongated_part_of_larger_box(
    candidate: dict,
    other: dict,
    aspect_ratio_threshold: float,
    containment_threshold: float,
    larger_area_ratio: float,
) -> bool:
    candidate_box = candidate["box_xyxy"]
    other_box = other["box_xyxy"]
    candidate_area = calculate_box_area(candidate_box)
    other_area = calculate_box_area(other_box)
    if calculate_aspect_ratio(candidate_box) < aspect_ratio_threshold:
        return False
    if other_area < candidate_area * larger_area_ratio:
        return False
    if calculate_containment(other_box, candidate_box) < containment_threshold:
        return False
    center_x = (candidate_box[0] + candidate_box[2]) / 2.0
    center_y = (candidate_box[1] + candidate_box[3]) / 2.0
    return (
        other_box[0] <= center_x <= other_box[2]
        and other_box[1] <= center_y <= other_box[3]
    )


def remove_elongated_parts(
    predictions: list[dict], postprocessing: dict
) -> tuple[list[dict], int]:
    kept: list[dict] = []
    removed = 0
    for candidate in predictions:
        is_part = any(
            is_elongated_part_of_larger_box(
                candidate,
                other,
                postprocessing["elongated_aspect_ratio"],
                postprocessing["elongated_containment"],
                postprocessing["elongated_larger_area_ratio"],
            )
            for other in predictions
            if other is not candidate
        )
        if is_part:
            removed += 1
        else:
            kept.append(candidate)
    return kept, removed


def suppress_overlapping_boxes(
    predictions: list[dict], iou_threshold: float
) -> tuple[list[dict], int]:
    """Keep the highest-confidence box for top-view overlaps, regardless of class."""
    candidates = sorted(predictions, key=lambda item: item["confidence"], reverse=True)
    kept: list[dict] = []
    removed = 0
    for candidate in candidates:
        if any(
            calculate_iou(candidate["box_xyxy"], existing["box_xyxy"])
            > iou_threshold
            for existing in kept
        ):
            removed += 1
            continue
        kept.append(candidate)
    return kept, removed


def suppress_contained_duplicates(
    predictions: list[dict], containment_threshold: float, larger_area_ratio: float
) -> tuple[list[dict], int]:
    """Remove lower-confidence same-class boxes contained in a larger box."""
    candidates = sorted(predictions, key=lambda item: item["confidence"], reverse=True)
    kept: list[dict] = []
    removed = 0
    for candidate in candidates:
        candidate_area = calculate_box_area(candidate["box_xyxy"])
        is_duplicate = any(
            existing["class_name"] == candidate["class_name"]
            and calculate_box_area(existing["box_xyxy"])
            >= candidate_area * larger_area_ratio
            and calculate_containment(
                existing["box_xyxy"], candidate["box_xyxy"]
            )
            >= containment_threshold
            for existing in kept
        )
        if is_duplicate:
            removed += 1
            continue
        kept.append(candidate)
    return kept, removed


def main() -> None:
    config = read_config()
    teacher_config = config["teacher"]
    rows = get_target_frames(config)
    output_directory = (
        resolve_project_path(config["artifact_directory"])
        / "yolo26x"
        / config["run_id"]
    )
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Teacher output already exists: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG_PATH, output_directory / "config_snapshot.yaml")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    started_at = time.perf_counter()
    model, predict, environment = load_yolo_teacher(
        teacher_config["checkpoint"], config["device"], use_fp16=True
    )
    model_loaded_at = time.perf_counter()

    output_records: list[dict] = []
    raw_output_records: list[dict] = []
    frame_statistics: list[dict] = []
    total_statistics = Counter()
    for frame_number, row in enumerate(rows, start=1):
        image_path = resolve_project_path(row["relative_image_path"])
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        frame_started_at = time.perf_counter()
        ownership_bounds = get_tile_ownership_bounds(
            image.width,
            image.height,
            teacher_config["tile_size_pixels"],
            teacher_config["tile_overlap"],
        )
        tile_predictions: list[dict] = []
        rejected_by_ownership = 0
        for top in sorted({origin[1] for origin in ownership_bounds}):
            for left in sorted({origin[0] for origin in ownership_bounds}):
                tile = image.crop(
                    (
                        left,
                        top,
                        min(left + teacher_config["tile_size_pixels"], image.width),
                        min(top + teacher_config["tile_size_pixels"], image.height),
                    )
                )
                predictions = predict(
                    tile,
                    teacher_config["input_size_pixels"],
                    teacher_config["confidence_threshold"],
                    1.0,
                )
                for prediction in predictions:
                    prediction = dict(prediction)
                    box = prediction["box_xyxy"]
                    prediction["box_xyxy"] = [
                        box[0] + left,
                        box[1] + top,
                        box[2] + left,
                        box[3] + top,
                    ]
                    prediction["tile_origin"] = [left, top]
                    prediction["frame_key"] = row["frame_key"]
                    prediction["relative_image_path"] = row["relative_image_path"]
                    prediction["source_video_relative_path"] = row["source_video_relative_path"]
                    prediction["split"] = row["split"]
                    prediction["source_image_width_pixels"] = image.width
                    prediction["source_image_height_pixels"] = image.height
                    prediction["inference_mode"] = "tiled"
                    raw_output_records.append(prediction)
                    minimum_x, minimum_y = ownership_bounds[(left, top)]
                    if keep_prediction_owned_by_tile(prediction, minimum_x, minimum_y):
                        tile_predictions.append(prediction)
                    else:
                        rejected_by_ownership += 1

        tile_predictions, converted_buses = normalize_bus_predictions(
            tile_predictions, enabled=True
        )
        tile_predictions, elongated_removed = remove_elongated_parts(
            tile_predictions, config["postprocessing"]
        )
        tile_predictions, overlap_removed = suppress_overlapping_boxes(
            tile_predictions, config["postprocessing"]["cross_class_iou"]
        )
        for prediction in tile_predictions:
            prediction["postprocessing_rules"] = [
                "tile_ownership",
                "remove_elongated_contained_parts",
                "class_agnostic_iou_suppression_0.30",
            ]
        output_records.extend(tile_predictions)
        total_statistics.update(
            {
                "ownership_rejected": rejected_by_ownership,
                "bus_converted_to_truck": converted_buses,
                "elongated_parts_removed": elongated_removed,
                "overlap_boxes_removed": overlap_removed,
            }
        )
        elapsed = time.perf_counter() - frame_started_at
        frame_statistics.append(
            {
                "frame_key": row["frame_key"],
                "raw_prediction_count": sum(
                    1 for item in raw_output_records if item["frame_key"] == row["frame_key"]
                ),
                "final_prediction_count": len(tile_predictions),
                "ownership_rejected": rejected_by_ownership,
                "bus_converted_to_truck": converted_buses,
                "elongated_parts_removed": elongated_removed,
                "overlap_boxes_removed": overlap_removed,
                "elapsed_seconds": elapsed,
            }
        )
        overlay_path = output_directory / "overlays_full_resolution" / image_path.name
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        draw_predictions(image, tile_predictions).save(overlay_path, quality=95, subsampling=0)
        print(
            f"{frame_number:02d}/{len(rows)} {row['frame_key']}: "
            f"{len(tile_predictions)} boxes ({elapsed:.2f}s)",
            flush=True,
        )

    finished_at = time.perf_counter()
    manifest_path = resolve_project_path(config["input_manifest"])
    checkpoint_path = Path(environment["checkpoint_path"])
    summary = {
        "teacher": "yolo26x",
        "checkpoint": teacher_config["checkpoint"],
        "checkpoint_sha256": calculate_sha256(checkpoint_path)
        if checkpoint_path.is_file()
        else None,
        "config_path": str((output_directory / "config_snapshot.yaml").resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "source_videos": [TARGET_VIDEO_NAME],
        "image_count": len(rows),
        "inference_mode": "tiled",
        "tile_size_pixels": teacher_config["tile_size_pixels"],
        "tile_overlap": teacher_config["tile_overlap"],
        "confidence_threshold": teacher_config["confidence_threshold"],
        "prediction_count": len(output_records),
        "prediction_count_by_class": dict(Counter(item["class_name"] for item in output_records)),
        "postprocessing": dict(total_statistics),
        "load_seconds": model_loaded_at - started_at,
        "total_seconds": finished_at - started_at,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
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
