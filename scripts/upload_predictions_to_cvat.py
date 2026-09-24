from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from cvat_sdk import Client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "Data" / "Frames" / "frames_manifest.csv"
DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Labels"
    / "proposals"
    / "prediction_tree"
    / "20260921_native_tiles_v6"
    / "predictions.json"
)
DEFAULT_CVAT_URL = "http://localhost:8080"
DEFAULT_PROJECT_NAME = "Vehicle pseudo-label review - prediction_tree v6"
IMAGE_QUALITY_PERCENT = 100
COCO_FORMAT_NAME = "COCO 1.0"

CLASS_TO_CATEGORY = {
    "car": (1, "car"),
    "truck": (2, "truck"),
    "motorcycle": (3, "motorcycle"),
    "class_conflict": (4, "unknown_type"),
}

CVAT_LABELS = [
    {"name": "car", "color": "#00C853"},
    {"name": "truck", "color": "#E53935"},
    {"name": "motorcycle", "color": "#F9A825"},
    {"name": "unknown_type", "color": "#424242"},
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CVAT review tasks and upload prediction_tree boxes."
    )
    parser.add_argument("--cvat-url", default=DEFAULT_CVAT_URL)
    parser.add_argument("--username", default=os.environ.get("CVAT_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("CVAT_PASSWORD"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--project-name", default=DEFAULT_PROJECT_NAME)
    arguments = parser.parse_args()

    if not arguments.username or not arguments.password:
        parser.error("Provide CVAT_USERNAME and CVAT_PASSWORD or use the matching arguments.")

    return arguments


def read_frame_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        return list(csv.DictReader(manifest_file))


def group_frames_by_video(frames: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped_frames: dict[str, list[dict[str, str]]] = defaultdict(list)
    for frame in frames:
        video_name = frame["frame_key"].split(":", maxsplit=1)[0]
        grouped_frames[video_name].append(frame)

    for video_frames in grouped_frames.values():
        video_frames.sort(key=lambda frame: int(frame["source_frame_id"]))

    return dict(sorted(grouped_frames.items()))


def group_predictions_by_frame(predictions_path: Path) -> dict[str, list[dict]]:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    grouped_predictions: dict[str, list[dict]] = defaultdict(list)
    for prediction in predictions:
        grouped_predictions[prediction["frame_key"]].append(prediction)
    return grouped_predictions


def clamp_box_to_image(
    box_xyxy: list[float], image_width: int, image_height: int
) -> tuple[float, float, float, float] | None:
    left, top, right, bottom = box_xyxy
    left = max(0.0, min(float(left), float(image_width)))
    top = max(0.0, min(float(top), float(image_height)))
    right = max(0.0, min(float(right), float(image_width)))
    bottom = max(0.0, min(float(bottom), float(image_height)))

    width = right - left
    height = bottom - top
    if width <= 0.0 or height <= 0.0:
        return None

    return left, top, width, height


def build_coco_annotations(
    frames: list[dict[str, str]], predictions_by_frame: dict[str, list[dict]]
) -> dict:
    images = []
    annotations = []
    annotation_id = 1

    for image_id, frame in enumerate(frames, start=1):
        image_width = int(frame["width"])
        image_height = int(frame["height"])
        image_path = Path(frame["image_path"])
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": image_width,
                "height": image_height,
            }
        )

        for prediction in predictions_by_frame.get(frame["frame_key"], []):
            category_id, _ = CLASS_TO_CATEGORY[prediction["class_name"]]
            box = clamp_box_to_image(
                prediction["box_xyxy"], image_width=image_width, image_height=image_height
            )
            if box is None:
                continue

            left, top, width, height = box
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [left, top, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "info": {"description": "Merged YOLO and RF-DETR pseudo-labels for manual review"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": category_name, "supercategory": "vehicle"}
            for category_id, category_name in CLASS_TO_CATEGORY.values()
        ],
    }


def write_coco_archive(coco_annotations: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary_directory:
        annotation_path = Path(temporary_directory) / "instances_default.json"
        annotation_path.write_text(
            json.dumps(coco_annotations, ensure_ascii=False), encoding="utf-8"
        )
        with ZipFile(output_path, mode="w", compression=ZIP_DEFLATED) as archive:
            archive.write(annotation_path, arcname="annotations/instances_default.json")


def get_image_paths(frames: list[dict[str, str]]) -> list[Path]:
    image_paths = [Path(frame["image_path"]) for frame in frames]
    missing_paths = [path for path in image_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing frame: {missing_paths[0]}")
    return image_paths


def upload_review_tasks(arguments: argparse.Namespace) -> None:
    frames_by_video = group_frames_by_video(read_frame_manifest(arguments.manifest))
    predictions_by_frame = group_predictions_by_frame(arguments.predictions)
    archive_directory = PROJECT_ROOT / "My Artifacts" / "cvat_import"

    with Client(arguments.cvat_url) as client:
        client.login((arguments.username, arguments.password))
        client.config.status_check_period = 2

        project = client.projects.create(
            spec={"name": arguments.project_name, "labels": CVAT_LABELS}
        )
        print(f"Created project #{project.id}: {project.name}")

        for video_name, frames in frames_by_video.items():
            split = frames[0]["split"]
            task_name = f"{video_name} - {split} - manual review"
            annotation_archive = archive_directory / f"{video_name}_prediction_tree_v6_coco.zip"
            coco_annotations = build_coco_annotations(frames, predictions_by_frame)
            write_coco_archive(coco_annotations, annotation_archive)

            task = client.tasks.create_from_data(
                spec={
                    "name": task_name,
                    "project_id": project.id,
                    "segment_size": len(frames),
                },
                resources=get_image_paths(frames),
                data_params={
                    "image_quality": IMAGE_QUALITY_PERCENT,
                    "sorting_method": "lexicographical",
                },
                annotation_path=str(annotation_archive),
                annotation_format=COCO_FORMAT_NAME,
            )
            print(
                f"Created task #{task.id}: {task_name}; "
                f"frames={len(frames)}, boxes={len(coco_annotations['annotations'])}"
            )


if __name__ == "__main__":
    upload_review_tasks(parse_arguments())
