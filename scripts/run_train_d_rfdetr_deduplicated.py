"""Run RF-DETR on all train_D frames with tiled deduplication."""

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
        calculate_sha256,
        draw_predictions,
        load_rfdetr_teacher,
        normalize_bus_predictions,
        read_manifest,
        resolve_project_path,
        write_json,
    )
except ModuleNotFoundError:
    from scripts.run_teacher_predictions import (
        PROJECT_DIRECTORY,
        calculate_sha256,
        draw_predictions,
        load_rfdetr_teacher,
        normalize_bus_predictions,
        read_manifest,
        resolve_project_path,
        write_json,
    )

try:
    from run_train_d_yolo_deduplicated import (
        get_tile_ownership_bounds,
        keep_prediction_owned_by_tile,
        remove_elongated_parts,
        suppress_contained_duplicates,
        suppress_overlapping_boxes,
    )
except ModuleNotFoundError:
    from scripts.run_train_d_yolo_deduplicated import (
        get_tile_ownership_bounds,
        keep_prediction_owned_by_tile,
        remove_elongated_parts,
        suppress_contained_duplicates,
        suppress_overlapping_boxes,
    )


CONFIG_PATH = PROJECT_DIRECTORY / "configs" / "rfdetr_train_d_deduplicated.yaml"
TARGET_VIDEO_NAME = "train_D.mp4"


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


def run_frame_inference(
    image: Image.Image,
    row: dict[str, str],
    predict,
    teacher_config: dict,
    postprocessing: dict,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    ownership_bounds = get_tile_ownership_bounds(
        image.width,
        image.height,
        teacher_config["tile_size_pixels"],
        teacher_config["tile_overlap"],
    )
    raw_predictions: list[dict] = []
    accepted_predictions: list[dict] = []
    ownership_rejected = 0
    horizontal_origins = sorted({origin[0] for origin in ownership_bounds})
    vertical_origins = sorted({origin[1] for origin in ownership_bounds})
    for top in vertical_origins:
        for left in horizontal_origins:
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
                teacher_config["model_resolution_pixels"],
                teacher_config["confidence_threshold"],
                1.0,
            )
            minimum_x, minimum_y = ownership_bounds[(left, top)]
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
                raw_predictions.append(prediction)
                if keep_prediction_owned_by_tile(prediction, minimum_x, minimum_y):
                    accepted_predictions.append(prediction)
                else:
                    ownership_rejected += 1

    accepted_predictions, bus_converted = normalize_bus_predictions(
        accepted_predictions, enabled=True
    )
    accepted_predictions, elongated_removed = remove_elongated_parts(
        accepted_predictions, postprocessing
    )
    accepted_predictions, contained_removed = suppress_contained_duplicates(
        accepted_predictions,
        postprocessing["contained_duplicate_containment"],
        postprocessing["contained_duplicate_larger_area_ratio"],
    )
    accepted_predictions, overlap_removed = suppress_overlapping_boxes(
        accepted_predictions, postprocessing["cross_class_iou"]
    )
    for prediction in accepted_predictions:
        prediction["postprocessing_rules"] = [
            "tile_ownership",
            "remove_elongated_contained_parts",
            "remove_same_class_contained_duplicates",
            "class_agnostic_iou_suppression_0.30",
        ]
    statistics = {
        "ownership_rejected": ownership_rejected,
        "bus_converted_to_truck": bus_converted,
        "elongated_parts_removed": elongated_removed,
        "contained_duplicates_removed": contained_removed,
        "overlap_boxes_removed": overlap_removed,
    }
    return raw_predictions, accepted_predictions, statistics


def main() -> None:
    config = read_config()
    teacher_config = config["teacher"]
    rows = get_target_frames(config)
    output_directory = (
        resolve_project_path(config["artifact_directory"])
        / "rfdetr_large"
        / config["run_id"]
    )
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Teacher output already exists: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG_PATH, output_directory / "config_snapshot.yaml")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    started_at = time.perf_counter()
    model, predict, environment = load_rfdetr_teacher(
        teacher_config["checkpoint"], config["device"], use_fp16=True
    )
    model_loaded_at = time.perf_counter()
    raw_output_records: list[dict] = []
    output_records: list[dict] = []
    frame_statistics: list[dict] = []
    postprocessing_totals = Counter()

    for frame_number, row in enumerate(rows, start=1):
        image_path = resolve_project_path(row["relative_image_path"])
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        frame_started_at = time.perf_counter()
        raw_predictions, predictions, statistics = run_frame_inference(
            image, row, predict, teacher_config, config["postprocessing"]
        )
        raw_output_records.extend(raw_predictions)
        output_records.extend(predictions)
        postprocessing_totals.update(statistics)
        elapsed_seconds = time.perf_counter() - frame_started_at
        frame_statistics.append(
            {
                "frame_key": row["frame_key"],
                "relative_image_path": row["relative_image_path"],
                "raw_prediction_count": len(raw_predictions),
                "final_prediction_count": len(predictions),
                "elapsed_seconds": elapsed_seconds,
                "postprocessing": statistics,
            }
        )
        overlay_path = output_directory / "overlays_full_resolution" / image_path.name
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        draw_predictions(image, predictions).save(
            overlay_path, quality=95, subsampling=0
        )
        print(
            f"{frame_number:02d}/{len(rows)} {row['frame_key']}: "
            f"{len(raw_predictions)} raw -> {len(predictions)} boxes "
            f"in {elapsed_seconds:.2f}s",
            flush=True,
        )

    finished_at = time.perf_counter()
    manifest_path = resolve_project_path(config["input_manifest"])
    checkpoint_path = Path(environment["checkpoint_path"])
    summary = {
        "teacher": "rfdetr_large",
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
        "model_resolution_pixels": teacher_config["model_resolution_pixels"],
        "confidence_threshold": teacher_config["confidence_threshold"],
        "raw_prediction_count": len(raw_output_records),
        "prediction_count": len(output_records),
        "prediction_count_by_class": dict(
            Counter(item["class_name"] for item in output_records)
        ),
        "postprocessing": dict(postprocessing_totals),
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
