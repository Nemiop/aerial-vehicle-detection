"""Render one full-resolution ground-truth preview from each source video."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
DATASET_DIRECTORY = PROJECT_DIRECTORY / "Data" / "Labels" / "final_vehicle" / "20260923_vehicle_yolo_v1"
DATASET_MANIFEST_PATH = DATASET_DIRECTORY / "dataset_manifest.csv"
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "My Artifacts" / "dataset_quality" / "20260923_vehicle_yolo_v1_label_previews"
SAMPLE_VIDEO_IDS = ("train_A", "train_B", "train_C", "train_D")
BOX_COLOR = "#19c37d"
TEXT_COLOR = "#ffffff"
LABEL_BACKGROUND_COLOR = "#087f4f"
JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0


def read_dataset_manifest() -> list[dict[str, str]]:
    """Read the canonical manifest that joins each image to its label file."""
    with DATASET_MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as manifest_file:
        return list(csv.DictReader(manifest_file))


def select_one_frame_per_video(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select the first chronological labeled frame from A, B, C, and D."""
    selections: list[dict[str, str]] = []
    for video_id in SAMPLE_VIDEO_IDS:
        matching_rows = [row for row in rows if row["frame_key"].startswith(f"{video_id}:")]
        if not matching_rows:
            raise ValueError(f"No labeled image was found for {video_id}")
        selections.append(matching_rows[0])
    return selections


def read_yolo_boxes(label_path: Path, image_width: int, image_height: int) -> list[tuple[float, float, float, float]]:
    """Convert normalized YOLO boxes into clipped full-image xyxy coordinates."""
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        values = raw_line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number} must contain five YOLO values")
        class_id, center_x, center_y, width, height = map(float, values)
        if int(class_id) != 0:
            raise ValueError(f"{label_path}:{line_number} contains unsupported class {class_id}")
        left = max(0.0, (center_x - width / 2.0) * image_width)
        top = max(0.0, (center_y - height / 2.0) * image_height)
        right = min(float(image_width), (center_x + width / 2.0) * image_width)
        bottom = min(float(image_height), (center_y + height / 2.0) * image_height)
        if right > left and bottom > top:
            boxes.append((left, top, right, bottom))
    return boxes


def get_label_font(font_size: int) -> ImageFont.ImageFont:
    """Choose a legible font available on Windows, with a Pillow fallback."""
    for font_path in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")):
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size=font_size)
    return ImageFont.load_default(size=font_size)


def draw_ground_truth_boxes(image: Image.Image, boxes: list[tuple[float, float, float, float]]) -> Image.Image:
    """Draw full-resolution one-class annotations without resizing the source image."""
    canvas = image.copy()
    drawer = ImageDraw.Draw(canvas)
    line_width = max(2, round(max(canvas.size) / 900))
    font = get_label_font(max(14, round(max(canvas.size) / 115)))
    for left, top, right, bottom in boxes:
        drawer.rectangle((left, top, right, bottom), outline=BOX_COLOR, width=line_width)
        label = "vehicle"
        label_bounds = drawer.textbbox((0, 0), label, font=font)
        label_width = label_bounds[2] - label_bounds[0] + line_width * 2
        label_height = label_bounds[3] - label_bounds[1] + line_width * 2
        label_top = top - label_height if top >= label_height else top
        drawer.rectangle((left, label_top, left + label_width, label_top + label_height), fill=LABEL_BACKGROUND_COLOR)
        drawer.text((left + line_width, label_top + line_width), label, fill=TEXT_COLOR, font=font)
    return canvas


def main() -> None:
    """Write four JPEG previews and a manifest describing their source labels."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    preview_records: list[dict[str, object]] = []
    for row in select_one_frame_per_video(read_dataset_manifest()):
        image_path = DATASET_DIRECTORY / row["image"]
        label_path = DATASET_DIRECTORY / row["label"]
        with Image.open(image_path) as opened_image:
            source_image = opened_image.convert("RGB")
        boxes = read_yolo_boxes(label_path, *source_image.size)
        preview_image = draw_ground_truth_boxes(source_image, boxes)
        output_path = OUTPUT_DIRECTORY / f"{Path(row['image']).stem}_ground_truth.jpg"
        preview_image.save(output_path, quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
        preview_records.append(
            {
                "frame_key": row["frame_key"],
                "source_image": row["source_image"],
                "dataset_image": row["image"],
                "dataset_label": row["label"],
                "image_width": source_image.width,
                "image_height": source_image.height,
                "box_count": len(boxes),
                "preview": str(output_path.relative_to(PROJECT_DIRECTORY)),
            }
        )
    (OUTPUT_DIRECTORY / "preview_manifest.json").write_text(
        json.dumps(preview_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(preview_records)} full-resolution previews to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
