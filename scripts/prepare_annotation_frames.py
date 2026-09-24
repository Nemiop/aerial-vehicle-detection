"""Extract training-video frames for annotation at a fixed sampling frequency.

Requires imageio-ffmpeg in addition to imageio:
    python -m pip install imageio imageio-ffmpeg
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import imageio.v2 as imageio


DEFAULT_RAW_VIDEO_DIRECTORY = Path("Data/Raw")
DEFAULT_FRAME_DIRECTORY = Path("Data/Raw/Frames")
DEFAULT_SAMPLING_FPS = 2.0
DEFAULT_JPEG_QUALITY = 95
MIN_JPEG_QUALITY = 0
MAX_JPEG_QUALITY = 100
FFMPEG_READER_FORMAT = "FFMPEG"
FRAME_ID_DIGITS = 8
MANIFEST_FILENAME = "frames_manifest.txt"

TRAINING_VIDEOS = {
    "train_A.mp4": "trainA",
    "train_B.mp4": "trainB",
    "train_C.mp4": "trainC",
    "train_D.mp4": "trainD",
}


def parse_arguments() -> argparse.Namespace:
    """Read source paths and the number of frames to save per second."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-video-directory",
        type=Path,
        default=DEFAULT_RAW_VIDEO_DIRECTORY,
        help="Directory containing train_A.mp4 through train_D.mp4.",
    )
    parser.add_argument(
        "--frame-directory",
        type=Path,
        default=DEFAULT_FRAME_DIRECTORY,
        help="Directory in which frame collections are created.",
    )
    parser.add_argument(
        "--sampling-fps",
        type=float,
        default=DEFAULT_SAMPLING_FPS,
        help=(
            "Number of frames to save per video second "
            f"(default: {DEFAULT_SAMPLING_FPS})."
        ),
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=(
            f"JPEG quality from {MIN_JPEG_QUALITY} to {MAX_JPEG_QUALITY} "
            f"(default: {DEFAULT_JPEG_QUALITY})."
        ),
    )
    arguments = parser.parse_args()
    if arguments.sampling_fps <= 0:
        parser.error("--sampling-fps must be greater than zero.")
    if not MIN_JPEG_QUALITY <= arguments.jpeg_quality <= MAX_JPEG_QUALITY:
        parser.error(
            f"--jpeg-quality must be between {MIN_JPEG_QUALITY} "
            f"and {MAX_JPEG_QUALITY}."
        )
    return arguments


def ensure_output_directories_are_empty(frame_directory: Path) -> None:
    """Stop before mixing frames from separate sampling runs."""
    for collection_name in TRAINING_VIDEOS.values():
        collection_directory = frame_directory / collection_name
        if collection_directory.exists() and any(collection_directory.iterdir()):
            raise FileExistsError(
                f"Frame directory must be empty before extraction: {collection_directory}"
            )


def get_image_filename(video_path: Path, source_frame_id: int) -> str:
    """Create an image name that retains the original video-frame ID."""
    frame_id = f"{source_frame_id:0{FRAME_ID_DIGITS}d}"
    return f"{video_path.stem}_frame_{frame_id}.jpg"


def get_sampled_frame_ids(source_fps: float, sampling_fps: float) -> Iterator[int]:
    """Yield source-frame IDs nearest to regularly spaced sampling timestamps."""
    sample_number = 0
    while True:
        yield round(sample_number * source_fps / sampling_fps)
        sample_number += 1


def save_sampled_frames(
    video_path: Path,
    collection_directory: Path,
    sampling_fps: float,
    jpeg_quality: int,
) -> list[Path]:
    """Read a video through FFmpeg and save frames selected by timestamp."""
    video_reader = imageio.get_reader(video_path, format=FFMPEG_READER_FORMAT)
    try:
        source_fps = video_reader.get_meta_data()["fps"]
        selected_frame_ids = get_sampled_frame_ids(source_fps, sampling_fps)
        next_selected_frame_id = next(selected_frame_ids)
        saved_paths: list[Path] = []
        collection_directory.mkdir(parents=True, exist_ok=True)

        for source_frame_id, image in enumerate(video_reader):
            if source_frame_id != next_selected_frame_id:
                continue

            image_path = collection_directory / get_image_filename(
                video_path, source_frame_id
            )
            imageio.imwrite(image_path, image, quality=jpeg_quality)
            saved_paths.append(image_path.resolve())
            next_selected_frame_id = next(selected_frame_ids)
    finally:
        video_reader.close()

    return saved_paths


def write_frame_manifest(frame_directory: Path, image_paths: list[Path]) -> None:
    """Write one absolute exported-frame path per line."""
    manifest_path = frame_directory / MANIFEST_FILENAME
    manifest_path.write_text(
        "\n".join(str(image_path) for image_path in image_paths) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Extract frames from the four training videos; Evaluation is excluded."""
    arguments = parse_arguments()
    ensure_output_directories_are_empty(arguments.frame_directory)

    saved_frame_paths: list[Path] = []
    for video_filename, collection_name in TRAINING_VIDEOS.items():
        video_path = arguments.raw_video_directory / video_filename
        if not video_path.is_file():
            raise FileNotFoundError(f"Training video was not found: {video_path}")

        frame_paths = save_sampled_frames(
            video_path=video_path,
            collection_directory=arguments.frame_directory / collection_name,
            sampling_fps=arguments.sampling_fps,
            jpeg_quality=arguments.jpeg_quality,
        )
        saved_frame_paths.extend(frame_paths)
        print(f"{video_filename}: {len(frame_paths)} frames saved")

    write_frame_manifest(arguments.frame_directory, saved_frame_paths)


if __name__ == "__main__":
    main()
