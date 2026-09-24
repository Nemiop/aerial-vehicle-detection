"""Replace B/C/D CVAT annotations with the existing RF-DETR 1280-tile run."""

from __future__ import annotations

import argparse
from pathlib import Path

from cvat_sdk import Client

from upload_predictions_to_cvat import (
    COCO_FORMAT_NAME,
    DEFAULT_CVAT_URL,
    DEFAULT_MANIFEST_PATH,
    build_coco_annotations,
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
    / "rfdetr_large"
    / "20260922_rfdetr_1280_tiles_bcd"
    / "predictions.json"
)
DEFAULT_ARCHIVE_DIRECTORY = PROJECT_ROOT / "My Artifacts" / "cvat_import"
TARGET_VIDEOS = ("train_B", "train_C", "train_D")
PROTECTED_VIDEO = "train_A"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace only B/C/D CVAT tasks with existing RF-DETR-only "
            "predictions from 1280 px tiles."
        )
    )
    parser.add_argument("--cvat-url", default=DEFAULT_CVAT_URL)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument(
        "--archive-directory", type=Path, default=DEFAULT_ARCHIVE_DIRECTORY
    )
    return parser.parse_args()


def get_target_frames(manifest_path: Path) -> dict[str, list[dict[str, str]]]:
    grouped_frames = group_frames_by_video(read_frame_manifest(manifest_path))
    missing_videos = [video for video in TARGET_VIDEOS if video not in grouped_frames]
    if missing_videos:
        raise ValueError(f"Missing videos in manifest: {missing_videos}")
    if PROTECTED_VIDEO in TARGET_VIDEOS:
        raise ValueError(f"Protected video unexpectedly selected: {PROTECTED_VIDEO}")
    return {video: grouped_frames[video] for video in TARGET_VIDEOS}


def create_archives(
    frames_by_video: dict[str, list[dict[str, str]]],
    predictions_path: Path,
    archive_directory: Path,
) -> dict[str, Path]:
    predictions_by_frame = group_predictions_by_frame(predictions_path)
    archive_directory.mkdir(parents=True, exist_ok=True)
    archives: dict[str, Path] = {}
    for video_name, frames in frames_by_video.items():
        archive_path = archive_directory / f"{video_name}_rfdetr_1280_tiles_coco.zip"
        coco_annotations = build_coco_annotations(frames, predictions_by_frame)
        write_coco_archive(coco_annotations, archive_path)
        archives[video_name] = archive_path
        print(
            f"Prepared {archive_path.name}: frames={len(frames)}, "
            f"boxes={len(coco_annotations['annotations'])}"
        )
    return archives


def replace_task_annotations(
    arguments: argparse.Namespace,
    frames_by_video: dict[str, list[dict[str, str]]],
    archives: dict[str, Path],
) -> None:
    with Client(arguments.cvat_url) as client:
        client.login((arguments.username, arguments.password))
        client.config.status_check_period = 2
        tasks_by_name = {task.name: task for task in client.tasks.list()}

        for video_name, frames in frames_by_video.items():
            matching_tasks = [
                task
                for task_name, task in tasks_by_name.items()
                if task_name.startswith(f"{video_name} - ")
            ]
            if len(matching_tasks) != 1:
                raise ValueError(
                    f"Expected one CVAT task for {video_name}, found "
                    f"{len(matching_tasks)}"
                )
            task = matching_tasks[0]
            task.import_annotations(COCO_FORMAT_NAME, archives[video_name])
            shape_count = len(task.get_annotations().shapes)
            print(
                f"Updated task #{task.id} ({task.name}): "
                f"frames={len(frames)}, boxes={shape_count}"
            )


def main() -> None:
    arguments = parse_arguments()
    frames_by_video = get_target_frames(arguments.manifest)
    archives = create_archives(
        frames_by_video, arguments.predictions, arguments.archive_directory
    )
    replace_task_annotations(arguments, frames_by_video, archives)


if __name__ == "__main__":
    main()
