"""Summarize teacher proposals and create pilot comparison contact sheets."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
PILOT_DIRECTORY = PROJECT_DIRECTORY / "My Artifacts" / "pseudo_annotation" / "pilot_v2"
TEACHERS = ("yolo26x", "rfdetr_large")
VARIANTS = ("full_frame", "full_frame_with_tiles")
AGREEMENT_IOU_THRESHOLD = 0.50
CONTACT_SHEET_COLUMNS = 4
CONTACT_SHEET_THUMBNAIL_SIZE = (768, 432)


def read_json(file_path: Path):
    """Read a JSON artifact."""
    return json.loads(file_path.read_text(encoding="utf-8"))


def calculate_iou(first_box: list[float], second_box: list[float]) -> float:
    """Calculate intersection over union for two xyxy boxes."""
    left = max(first_box[0], second_box[0])
    top = max(first_box[1], second_box[1])
    right = min(first_box[2], second_box[2])
    bottom = min(first_box[3], second_box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first_box[2] - first_box[0]) * max(
        0.0, first_box[3] - first_box[1]
    )
    second_area = max(0.0, second_box[2] - second_box[0]) * max(
        0.0, second_box[3] - second_box[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def count_matches(
    first_predictions: list[dict],
    second_predictions: list[dict],
    require_same_class: bool,
) -> int:
    """Greedily match predictions from two sets at the configured IoU."""
    candidates = []
    for first_index, first in enumerate(first_predictions):
        for second_index, second in enumerate(second_predictions):
            if require_same_class and first["class_name"] != second["class_name"]:
                continue
            iou = calculate_iou(first["box_xyxy"], second["box_xyxy"])
            if iou >= AGREEMENT_IOU_THRESHOLD:
                candidates.append((iou, first_index, second_index))
    used_first = set()
    used_second = set()
    matches = 0
    for _, first_index, second_index in sorted(candidates, reverse=True):
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        matches += 1
    return matches


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def summarize_predictions(predictions: list[dict]) -> dict:
    """Summarize count, classes, confidence, and proposal box area."""
    confidences = [item["confidence"] for item in predictions]
    box_areas = [
        max(0.0, item["box_xyxy"][2] - item["box_xyxy"][0])
        * max(0.0, item["box_xyxy"][3] - item["box_xyxy"][1])
        for item in predictions
    ]
    return {
        "count": len(predictions),
        "class_counts": dict(sorted(Counter(item["class_name"] for item in predictions).items())),
        "confidence_p10": percentile(confidences, 0.10),
        "confidence_median": statistics.median(confidences) if confidences else None,
        "confidence_p90": percentile(confidences, 0.90),
        "box_area_median_pixels": statistics.median(box_areas) if box_areas else None,
    }


def create_contact_sheet(teacher: str, variant: str, frame_keys: list[str]) -> Path:
    """Arrange all twelve overlay images in one reproducible review sheet."""
    overlay_directory = PILOT_DIRECTORY / teacher / "overlays" / variant
    thumbnails = []
    for frame_key in frame_keys:
        image_name = frame_key.replace(":", "_frame_") + ".jpg"
        image_path = overlay_directory / image_name
        with Image.open(image_path) as opened_image:
            thumbnail = opened_image.convert("RGB")
            thumbnail.thumbnail(CONTACT_SHEET_THUMBNAIL_SIZE)
        canvas = Image.new("RGB", CONTACT_SHEET_THUMBNAIL_SIZE, "white")
        canvas.paste(
            thumbnail,
            (
                (CONTACT_SHEET_THUMBNAIL_SIZE[0] - thumbnail.width) // 2,
                (CONTACT_SHEET_THUMBNAIL_SIZE[1] - thumbnail.height) // 2,
            ),
        )
        ImageDraw.Draw(canvas).rectangle((0, 0, 300, 24), fill="black")
        ImageDraw.Draw(canvas).text((6, 5), frame_key, fill="white")
        thumbnails.append(canvas)

    row_count = (len(thumbnails) + CONTACT_SHEET_COLUMNS - 1) // CONTACT_SHEET_COLUMNS
    sheet = Image.new(
        "RGB",
        (
            CONTACT_SHEET_COLUMNS * CONTACT_SHEET_THUMBNAIL_SIZE[0],
            row_count * CONTACT_SHEET_THUMBNAIL_SIZE[1],
        ),
        "white",
    )
    for index, thumbnail in enumerate(thumbnails):
        left = (index % CONTACT_SHEET_COLUMNS) * CONTACT_SHEET_THUMBNAIL_SIZE[0]
        top = (index // CONTACT_SHEET_COLUMNS) * CONTACT_SHEET_THUMBNAIL_SIZE[1]
        sheet.paste(thumbnail, (left, top))

    output_path = PILOT_DIRECTORY / "review" / f"{teacher}_{variant}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return output_path


def main() -> None:
    """Create aggregate metrics, frame metrics, and review sheets."""
    predictions_by_teacher = {
        teacher: read_json(PILOT_DIRECTORY / teacher / "predictions.json")
        for teacher in TEACHERS
    }
    indexed = defaultdict(list)
    for teacher, predictions in predictions_by_teacher.items():
        for prediction in predictions:
            indexed[(teacher, prediction["variant"], prediction["frame_key"])].append(
                prediction
            )

    frame_keys = sorted(
        {
            prediction["frame_key"]
            for predictions in predictions_by_teacher.values()
            for prediction in predictions
        }
    )
    frame_rows = []
    agreement = {}
    for variant in VARIANTS:
        same_class_matches = 0
        any_class_matches = 0
        yolo_count = 0
        rfdetr_count = 0
        for frame_key in frame_keys:
            yolo = indexed[("yolo26x", variant, frame_key)]
            rfdetr = indexed[("rfdetr_large", variant, frame_key)]
            same_class = count_matches(yolo, rfdetr, require_same_class=True)
            any_class = count_matches(yolo, rfdetr, require_same_class=False)
            same_class_matches += same_class
            any_class_matches += any_class
            yolo_count += len(yolo)
            rfdetr_count += len(rfdetr)
            frame_rows.append(
                {
                    "frame_key": frame_key,
                    "variant": variant,
                    "yolo26x_count": len(yolo),
                    "rfdetr_large_count": len(rfdetr),
                    "same_class_matches_iou_0_5": same_class,
                    "any_class_matches_iou_0_5": any_class,
                }
            )
        agreement[variant] = {
            "same_class_matches_iou_0_5": same_class_matches,
            "any_class_matches_iou_0_5": any_class_matches,
            "yolo26x_matched_fraction_same_class": (
                same_class_matches / yolo_count if yolo_count else None
            ),
            "rfdetr_large_matched_fraction_same_class": (
                same_class_matches / rfdetr_count if rfdetr_count else None
            ),
            "class_disagreement_among_spatial_matches": any_class_matches
            - same_class_matches,
        }

    aggregate = {}
    for teacher in TEACHERS:
        aggregate[teacher] = {}
        for variant in VARIANTS:
            selected = [
                item
                for item in predictions_by_teacher[teacher]
                if item["variant"] == variant
            ]
            aggregate[teacher][variant] = summarize_predictions(selected)

    metrics = {
        "agreement_iou_threshold": AGREEMENT_IOU_THRESHOLD,
        "aggregate": aggregate,
        "cross_teacher_agreement": agreement,
    }
    metrics_path = PILOT_DIRECTORY / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    frame_metrics_path = PILOT_DIRECTORY / "frame_metrics.csv"
    with frame_metrics_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(frame_rows[0]))
        writer.writeheader()
        writer.writerows(frame_rows)

    for teacher in TEACHERS:
        for variant in VARIANTS:
            create_contact_sheet(teacher, variant, frame_keys)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
