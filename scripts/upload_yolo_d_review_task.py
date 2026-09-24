"""Create a separate CVAT review task for the deduplicated YOLO train_D run."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvat_sdk import Client

from upload_predictions_to_cvat import (
    COCO_FORMAT_NAME,
    DEFAULT_CVAT_URL,
    DEFAULT_MANIFEST_PATH,
    build_coco_annotations,
    get_image_paths,
    group_frames_by_video,
    group_predictions_by_frame,
    read_frame_manifest,
    write_coco_archive,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Labels"
    / "proposals"
    / "yolo26x"
    / "20260923_yolo_train_d_deduplicated"
    / "predictions.json"
)
DEFAULT_ARCHIVE_PATH = (
    PROJECT_ROOT / "My Artifacts" / "cvat_import" / "train_D_yolo26_deduplicated_coco.zip"
)
PROJECT_NAME = "Vehicle pseudo-label review - prediction_tree v6"
TASK_NAME = "train_D - validation - YOLO26 deduplicated"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cvat-url", default=DEFAULT_CVAT_URL)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    frames_by_video = group_frames_by_video(read_frame_manifest(arguments.manifest))
    frames = frames_by_video["train_D"]
    predictions_by_frame = group_predictions_by_frame(arguments.predictions)
    coco_annotations = build_coco_annotations(frames, predictions_by_frame)
    write_coco_archive(coco_annotations, arguments.archive)

    with Client(arguments.cvat_url) as client:
        client.login((arguments.username, arguments.password))
        project = next(
            project for project in client.projects.list() if project.name == PROJECT_NAME
        )
        if any(task.name == TASK_NAME for task in client.tasks.list()):
            raise RuntimeError(f"CVAT task already exists: {TASK_NAME}")
        task = client.tasks.create_from_data(
            spec={
                "name": TASK_NAME,
                "project_id": project.id,
                "segment_size": len(frames),
            },
            resources=get_image_paths(frames),
            data_params={"image_quality": 100, "sorting_method": "lexicographical"},
            annotation_path=str(arguments.archive),
            annotation_format=COCO_FORMAT_NAME,
        )
        print(
            f"Created task #{task.id}: {TASK_NAME}; "
            f"frames={len(frames)}, boxes={len(coco_annotations['annotations'])}"
        )


if __name__ == "__main__":
    main()
