"""Select fixed, evenly spaced train frames for teacher calibration."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST_PATH = PROJECT_DIRECTORY / "Data" / "Frames" / "frames_manifest.csv"
PILOT_MANIFEST_PATH = PROJECT_DIRECTORY / "Data" / "Frames" / "pilot_manifest.csv"
PILOT_VIDEO_NAMES = ("train_A.mp4", "train_B.mp4", "train_C.mp4")
SAMPLES_PER_VIDEO = 4


def read_train_rows() -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    """Group train rows by their source video in manifest order."""
    with SOURCE_MANIFEST_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as manifest_file:
        reader = csv.DictReader(manifest_file)
        fieldnames = list(reader.fieldnames or [])
        rows_by_video = {video_name: [] for video_name in PILOT_VIDEO_NAMES}
        for row in reader:
            video_name = Path(row["source_video_path"]).name
            if row["split"] == "train" and video_name in rows_by_video:
                rows_by_video[video_name].append(row)
    return fieldnames, rows_by_video


def get_evenly_spaced_indices(row_count: int, sample_count: int) -> list[int]:
    """Return deterministic indices spanning the first through last row."""
    if row_count < sample_count:
        raise ValueError(f"Cannot select {sample_count} samples from {row_count} rows")
    return [round(index * (row_count - 1) / (sample_count - 1)) for index in range(sample_count)]


def select_pilot_rows(rows_by_video: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    """Select the same number of temporal positions from every train video."""
    selected_rows: list[dict[str, str]] = []
    for video_name in PILOT_VIDEO_NAMES:
        video_rows = rows_by_video[video_name]
        selected_rows.extend(
            video_rows[index]
            for index in get_evenly_spaced_indices(len(video_rows), SAMPLES_PER_VIDEO)
        )
    return selected_rows


def write_pilot_manifest(
    fieldnames: list[str], selected_rows: list[dict[str, str]]
) -> None:
    """Write the frozen pilot selection without copying image files."""
    with PILOT_MANIFEST_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)


def main() -> None:
    """Create a reproducible 12-frame pilot manifest."""
    fieldnames, rows_by_video = read_train_rows()
    selected_rows = select_pilot_rows(rows_by_video)
    write_pilot_manifest(fieldnames, selected_rows)
    for row in selected_rows:
        print(row["frame_key"], row["timestamp_seconds"])


if __name__ == "__main__":
    main()
