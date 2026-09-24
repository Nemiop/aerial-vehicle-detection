"""Build a one-class Ultralytics YOLO dataset from reviewed A/B/C and RF-DETR D."""

from __future__ import annotations

import csv
import json
import shutil
import xml.etree.ElementTree as ElementTree
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_MANIFEST_PATH = PROJECT_ROOT / "Data" / "Frames" / "frames_manifest.csv"
REVIEWED_ANNOTATIONS_PATH = (
    PROJECT_ROOT / "Data" / "Labels" / "reviewed" / "filtered" / "annotations.xml"
)
RFDETR_D_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "Data"
    / "Labels"
    / "proposals"
    / "rfdetr_large"
    / "20260923_rfdetr_train_d_deduplicated_v3_full"
    / "predictions.json"
)
OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "Data" / "Labels" / "final_vehicle" / "20260923_vehicle_yolo_v1"
)
REVIEWED_TASK_IDS = {"1": "train_A", "2": "train_B", "3": "train_C"}
RFDETR_VIDEO_NAME = "train_D.mp4"
CLASS_ID = 0
CLASS_NAME = "vehicle"
COORDINATE_DECIMAL_PLACES = 6


@dataclass(frozen=True)
class FrameInfo:
    frame_key: str
    image_name: str
    relative_image_path: str
    width: int
    height: int
    split: str


@dataclass(frozen=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float


def read_frame_manifest() -> dict[str, FrameInfo]:
    frames: dict[str, FrameInfo] = {}
    with FRAME_MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            image_name = Path(row["relative_image_path"]).name
            frame = FrameInfo(
                frame_key=row["frame_key"],
                image_name=image_name,
                relative_image_path=row["relative_image_path"],
                width=int(row["width"]),
                height=int(row["height"]),
                split="val" if row["split"] == "validation" else "train",
            )
            if frame.frame_key in frames or image_name in {
                item.image_name for item in frames.values()
            }:
                raise ValueError(f"Duplicate frame or image name: {frame.frame_key}")
            frames[frame.frame_key] = frame
    return frames


def clamp_box(box: BoundingBox, width: int, height: int) -> BoundingBox | None:
    left = max(0.0, min(box.left, float(width)))
    top = max(0.0, min(box.top, float(height)))
    right = max(0.0, min(box.right, float(width)))
    bottom = max(0.0, min(box.bottom, float(height)))
    if right <= left or bottom <= top:
        return None
    return BoundingBox(left, top, right, bottom)


def parse_reviewed_boxes(
    frames_by_key: dict[str, FrameInfo],
) -> dict[str, list[BoundingBox]]:
    root = ElementTree.parse(REVIEWED_ANNOTATIONS_PATH).getroot()
    boxes_by_frame: dict[str, list[BoundingBox]] = defaultdict(list)
    seen_frames: set[str] = set()
    for image in root.findall("image"):
        task_id = image.attrib.get("task_id")
        if task_id not in REVIEWED_TASK_IDS:
            continue
        image_name = image.attrib["name"]
        matching_frames = [
            frame for frame in frames_by_key.values() if frame.image_name == image_name
        ]
        if len(matching_frames) != 1:
            raise ValueError(f"Cannot map reviewed image to manifest: {image_name}")
        frame = matching_frames[0]
        if frame.frame_key in seen_frames:
            raise ValueError(f"Duplicate reviewed frame: {frame.frame_key}")
        seen_frames.add(frame.frame_key)
        for box in image.findall("box"):
            if box.attrib.get("outside") == "1":
                continue
            parsed = clamp_box(
                BoundingBox(
                    float(box.attrib["xtl"]),
                    float(box.attrib["ytl"]),
                    float(box.attrib["xbr"]),
                    float(box.attrib["ybr"]),
                ),
                frame.width,
                frame.height,
            )
            if parsed is not None:
                boxes_by_frame[frame.frame_key].append(parsed)
    expected = {
        frame.frame_key
        for frame in frames_by_key.values()
        if frame.image_name.startswith(("train_A_", "train_B_", "train_C_"))
    }
    if seen_frames != expected:
        raise ValueError(
            f"Reviewed A/B/C frame mismatch: missing={sorted(expected - seen_frames)}, "
            f"unexpected={sorted(seen_frames - expected)}"
        )
    return dict(boxes_by_frame)


def parse_rfdetr_boxes(
    frames_by_key: dict[str, FrameInfo],
) -> dict[str, list[BoundingBox]]:
    predictions = json.loads(RFDETR_D_PREDICTIONS_PATH.read_text(encoding="utf-8"))
    boxes_by_frame: dict[str, list[BoundingBox]] = defaultdict(list)
    for prediction in predictions:
        frame_key = prediction["frame_key"]
        frame = frames_by_key.get(frame_key)
        if frame is None:
            raise ValueError(f"RF-DETR frame missing from manifest: {frame_key}")
        if Path(prediction["source_video_relative_path"]).name != RFDETR_VIDEO_NAME:
            raise ValueError(f"Unexpected RF-DETR source video: {prediction['source_video_relative_path']}")
        box = prediction["box_xyxy"]
        parsed = clamp_box(
            BoundingBox(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            frame.width,
            frame.height,
        )
        if parsed is not None:
            boxes_by_frame[frame_key].append(parsed)
    d_frames = {
        frame.frame_key
        for frame in frames_by_key.values()
        if Path(frame.relative_image_path).parts[-2] == "trainD"
    }
    if set(boxes_by_frame) != d_frames:
        raise ValueError(
            f"RF-DETR D frame mismatch: missing={sorted(d_frames - set(boxes_by_frame))}, "
            f"unexpected={sorted(set(boxes_by_frame) - d_frames)}"
        )
    return dict(boxes_by_frame)


def format_yolo_box(box: BoundingBox, width: int, height: int) -> str:
    center_x = ((box.left + box.right) / 2.0) / width
    center_y = ((box.top + box.bottom) / 2.0) / height
    box_width = (box.right - box.left) / width
    box_height = (box.bottom - box.top) / height
    values = (center_x, center_y, box_width, box_height)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"Normalized box coordinate outside [0, 1]: {values}")
    formatted = " ".join(f"{value:.{COORDINATE_DECIMAL_PLACES}f}" for value in values)
    return f"{CLASS_ID} {formatted}"


def write_dataset(
    frames_by_key: dict[str, FrameInfo],
    boxes_by_frame: dict[str, list[BoundingBox]],
) -> dict:
    if OUTPUT_DIRECTORY.exists():
        raise FileExistsError(f"Output directory already exists: {OUTPUT_DIRECTORY}")
    for split in ("train", "val"):
        (OUTPUT_DIRECTORY / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIRECTORY / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    split_frame_counts = Counter()
    split_box_counts = Counter()
    for frame_key in sorted(frames_by_key):
        frame = frames_by_key[frame_key]
        source_path = PROJECT_ROOT / frame.relative_image_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        split = "val" if frame.image_name.startswith("train_D_") else "train"
        destination_image = OUTPUT_DIRECTORY / "images" / split / frame.image_name
        destination_label = OUTPUT_DIRECTORY / "labels" / split / f"{Path(frame.image_name).stem}.txt"
        shutil.copy2(source_path, destination_image)
        label_lines = [
            format_yolo_box(box, frame.width, frame.height)
            for box in boxes_by_frame.get(frame_key, [])
        ]
        destination_label.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        split_frame_counts[split] += 1
        split_box_counts[split] += len(label_lines)
        manifest_rows.append(
            {
                "frame_key": frame_key,
                "split": split,
                "image": f"images/{split}/{frame.image_name}",
                "label": f"labels/{split}/{Path(frame.image_name).stem}.txt",
                "source_image": frame.relative_image_path,
                "annotation_source": (
                    "CVAT reviewed annotations.xml"
                    if split == "train"
                    else "RF-DETR 20260923_rfdetr_train_d_deduplicated_v3_full"
                ),
                "box_count": len(label_lines),
                "width": frame.width,
                "height": frame.height,
            }
        )

    (OUTPUT_DIRECTORY / "data.yaml").write_text(
        f"path: {OUTPUT_DIRECTORY.relative_to(PROJECT_ROOT).as_posix()}\n"
        "train: images/train\nval: images/val\n\n"
        "names:\n  0: vehicle\n",
        encoding="utf-8",
    )
    with (OUTPUT_DIRECTORY / "dataset_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "dataset_name": OUTPUT_DIRECTORY.name,
        "class_names": {str(CLASS_ID): CLASS_NAME},
        "train_sources": ["CVAT annotations.xml tasks 1, 2, 3 (A, B, C)"],
        "validation_sources": [
            "RF-DETR 20260923_rfdetr_train_d_deduplicated_v3_full (D)"
        ],
        "image_count": dict(split_frame_counts),
        "box_count": dict(split_box_counts),
        "total_image_count": sum(split_frame_counts.values()),
        "total_box_count": sum(split_box_counts.values()),
        "coordinate_format": "YOLO normalized cx cy width height",
        "reviewed_annotations_sha256": _sha256(REVIEWED_ANNOTATIONS_PATH),
        "rfdetr_predictions_sha256": _sha256(RFDETR_D_PREDICTIONS_PATH),
    }
    (OUTPUT_DIRECTORY / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _sha256(file_path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    frames_by_key = read_frame_manifest()
    reviewed_boxes = parse_reviewed_boxes(frames_by_key)
    rfdetr_boxes = parse_rfdetr_boxes(frames_by_key)
    boxes_by_frame = {**reviewed_boxes, **rfdetr_boxes}
    metadata = write_dataset(frames_by_key, boxes_by_frame)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
