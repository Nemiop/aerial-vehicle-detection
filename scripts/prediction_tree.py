"""Merge two teacher proposal trees into one review-oriented prediction tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIRECTORY / "configs" / "merge_predictions.yaml"
TEACHER_NAMES = ("yolo26x", "rfdetr_large")
OVERLAY_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arial.ttf",
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def parse_arguments() -> argparse.Namespace:
    """Read the optional merge configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(configured_path)
    return path if path.is_absolute() else PROJECT_DIRECTORY / path


def read_json(file_path: Path):
    """Read one JSON file."""
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json(file_path: Path, value) -> None:
    """Write deterministic readable JSON."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def calculate_sha256(file_path: Path) -> str:
    """Calculate a file digest using bounded memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_box_area(box: list[float]) -> float:
    """Return the positive area of an xyxy box."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def calculate_iou(first_box: list[float], second_box: list[float]) -> float:
    """Calculate intersection over union for two xyxy boxes."""
    left = max(first_box[0], second_box[0])
    top = max(first_box[1], second_box[1])
    right = min(first_box[2], second_box[2])
    bottom = min(first_box[3], second_box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (
        calculate_box_area(first_box)
        + calculate_box_area(second_box)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def calculate_axis_intersection(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    """Return the intersection length of two one-dimensional ranges."""
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def calculate_containment(first_box: list[float], second_box: list[float]) -> float:
    """Return intersection divided by the smaller box area."""
    x_intersection = calculate_axis_intersection(
        first_box[0], first_box[2], second_box[0], second_box[2]
    )
    y_intersection = calculate_axis_intersection(
        first_box[1], first_box[3], second_box[1], second_box[3]
    )
    smaller_area = min(calculate_box_area(first_box), calculate_box_area(second_box))
    return x_intersection * y_intersection / smaller_area if smaller_area else 0.0


def is_center_inside(inner_box: list[float], outer_box: list[float]) -> bool:
    """Return whether the center of one box lies inside another."""
    center_x = (inner_box[0] + inner_box[2]) / 2.0
    center_y = (inner_box[1] + inner_box[3]) / 2.0
    return (
        outer_box[0] <= center_x <= outer_box[2]
        and outer_box[1] <= center_y <= outer_box[3]
    )


def calculate_union_box(boxes: list[list[float]]) -> list[float]:
    """Return the smallest axis-aligned box containing every input box."""
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def should_keep_large_trucks_separate(
    first_box: list[float], second_box: list[float], settings: dict
) -> bool:
    """Keep comparable side-by-side trucks whose horizontal extents cross."""
    first_width = first_box[2] - first_box[0]
    second_width = second_box[2] - second_box[0]
    smaller_width = min(first_width, second_width)
    if smaller_width <= 0:
        return False

    first_area = calculate_box_area(first_box)
    second_area = calculate_box_area(second_box)
    larger_area = max(first_area, second_area)
    if larger_area <= 0:
        return False
    area_ratio = min(first_area, second_area) / larger_area
    if area_ratio < settings["distinct_truck_minimum_area_ratio"]:
        return False

    horizontal_intersection = calculate_axis_intersection(
        first_box[0], first_box[2], second_box[0], second_box[2]
    )
    horizontal_overlap = horizontal_intersection / smaller_width
    if horizontal_overlap < settings["distinct_truck_horizontal_overlap"]:
        return False

    if first_box[0] < second_box[0] and first_box[2] < second_box[2]:
        left_protrusion = second_box[0] - first_box[0]
        right_protrusion = second_box[2] - first_box[2]
    elif second_box[0] < first_box[0] and second_box[2] < first_box[2]:
        left_protrusion = first_box[0] - second_box[0]
        right_protrusion = first_box[2] - second_box[2]
    else:
        return False

    minimum_protrusion = settings["distinct_truck_horizontal_protrusion"]
    if (
        left_protrusion / smaller_width < minimum_protrusion
        or right_protrusion / smaller_width < minimum_protrusion
    ):
        return False

    first_center_x = (first_box[0] + first_box[2]) / 2.0
    second_center_x = (second_box[0] + second_box[2]) / 2.0
    center_separation = abs(first_center_x - second_center_x) / smaller_width
    return center_separation >= settings[
        "distinct_truck_horizontal_center_separation"
    ]


def are_aligned_truck_fragments(
    anchor_box: list[float], candidate_box: list[float], settings: dict
) -> bool:
    """Detect contained or axis-aligned partial boxes of one long truck."""
    if calculate_containment(anchor_box, candidate_box) >= settings[
        "containment_over_smaller"
    ]:
        return True
    if is_center_inside(candidate_box, anchor_box):
        return True

    anchor_width = anchor_box[2] - anchor_box[0]
    anchor_height = anchor_box[3] - anchor_box[1]
    candidate_width = candidate_box[2] - candidate_box[0]
    candidate_height = candidate_box[3] - candidate_box[1]
    anchor_is_vertical = anchor_height >= anchor_width
    candidate_is_vertical = candidate_height >= candidate_width
    if anchor_is_vertical != candidate_is_vertical:
        return False

    if anchor_is_vertical:
        transverse_intersection = calculate_axis_intersection(
            anchor_box[0], anchor_box[2], candidate_box[0], candidate_box[2]
        )
        transverse_ratio = transverse_intersection / min(
            anchor_width, candidate_width
        )
        longitudinal_intersection = calculate_axis_intersection(
            anchor_box[1], anchor_box[3], candidate_box[1], candidate_box[3]
        )
        longitudinal_ratio = longitudinal_intersection / min(
            anchor_height, candidate_height
        )
        longitudinal_gap = max(
            0.0,
            anchor_box[1] - candidate_box[3],
            candidate_box[1] - anchor_box[3],
        ) / max(anchor_height, candidate_height)
    else:
        transverse_intersection = calculate_axis_intersection(
            anchor_box[1], anchor_box[3], candidate_box[1], candidate_box[3]
        )
        transverse_ratio = transverse_intersection / min(
            anchor_height, candidate_height
        )
        longitudinal_intersection = calculate_axis_intersection(
            anchor_box[0], anchor_box[2], candidate_box[0], candidate_box[2]
        )
        longitudinal_ratio = longitudinal_intersection / min(
            anchor_width, candidate_width
        )
        longitudinal_gap = max(
            0.0,
            anchor_box[0] - candidate_box[2],
            candidate_box[0] - anchor_box[2],
        ) / max(anchor_width, candidate_width)

    return transverse_ratio >= settings["transverse_overlap"] and (
        longitudinal_ratio >= settings["longitudinal_overlap"]
        or longitudinal_gap <= settings["maximum_longitudinal_gap"]
    )


def read_manifest(file_path: Path) -> list[dict[str, str]]:
    """Read the canonical frame manifest in stable order."""
    with file_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def index_predictions(predictions: list[dict]) -> dict[str, list[dict]]:
    """Group teacher predictions by stable frame key."""
    indexed = defaultdict(list)
    for prediction in predictions:
        indexed[prediction["frame_key"]].append(prediction)
    return indexed


def match_predictions(
    yolo_predictions: list[dict],
    rfdetr_predictions: list[dict],
    minimum_iou: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Find globally optimal one-to-one cross-teacher matches by IoU."""
    if not yolo_predictions or not rfdetr_predictions:
        return [], set(), set()
    iou_matrix = np.zeros(
        (len(yolo_predictions), len(rfdetr_predictions)), dtype=np.float32
    )
    for yolo_index, yolo_prediction in enumerate(yolo_predictions):
        for rfdetr_index, rfdetr_prediction in enumerate(rfdetr_predictions):
            iou_matrix[yolo_index, rfdetr_index] = calculate_iou(
                yolo_prediction["box_xyxy"], rfdetr_prediction["box_xyxy"]
            )
    yolo_indices, rfdetr_indices = linear_sum_assignment(1.0 - iou_matrix)
    matches = []
    matched_yolo = set()
    matched_rfdetr = set()
    for yolo_index, rfdetr_index in zip(yolo_indices, rfdetr_indices):
        iou = float(iou_matrix[yolo_index, rfdetr_index])
        if iou < minimum_iou:
            continue
        matches.append((int(yolo_index), int(rfdetr_index), iou))
        matched_yolo.add(int(yolo_index))
        matched_rfdetr.add(int(rfdetr_index))
    return matches, matched_yolo, matched_rfdetr


def describe_teacher_prediction(prediction: dict) -> dict:
    """Keep the teacher evidence needed for review and provenance."""
    return {
        "class_name": prediction["class_name"],
        "source_class_name": prediction.get(
            "source_class_name", prediction["class_name"]
        ),
        "source_class_id": prediction["source_class_id"],
        "confidence": prediction["confidence"],
        "box_xyxy": prediction["box_xyxy"],
        "tile_origin": prediction.get("tile_origin"),
    }


def fuse_matched_predictions(
    yolo_prediction: dict,
    rfdetr_prediction: dict,
    matched_iou: float,
    config: dict,
) -> dict:
    """Fuse a matched pair and expose class disagreement explicitly."""
    near_identical_iou = config["matching"]["near_identical_iou"]
    if matched_iou >= near_identical_iou:
        yolo_area = calculate_box_area(yolo_prediction["box_xyxy"])
        rfdetr_area = calculate_box_area(rfdetr_prediction["box_xyxy"])
        if yolo_area >= rfdetr_area:
            box = list(yolo_prediction["box_xyxy"])
            box_source = "yolo26x"
        else:
            box = list(rfdetr_prediction["box_xyxy"])
            box_source = "rfdetr_large"
        merge_status = "matched_near_identical_larger_box"
    else:
        box = [
            (yolo_coordinate + rfdetr_coordinate) / 2.0
            for yolo_coordinate, rfdetr_coordinate in zip(
                yolo_prediction["box_xyxy"], rfdetr_prediction["box_xyxy"]
            )
        ]
        box_source = "equal_weight_box_fusion"
        merge_status = "matched_moderate_overlap_fused"

    yolo_class = yolo_prediction["class_name"]
    rfdetr_class = rfdetr_prediction["class_name"]
    has_class_conflict = yolo_class != rfdetr_class
    output_class = (
        config["class_conflict"]["output_class_name"]
        if has_class_conflict
        else yolo_class
    )
    return {
        "box_xyxy": box,
        "class_name": output_class,
        "has_class_conflict": has_class_conflict,
        "teacher_classes": {
            "yolo26x": yolo_class,
            "rfdetr_large": rfdetr_class,
        },
        "teacher_confidences": {
            "yolo26x": yolo_prediction["confidence"],
            "rfdetr_large": rfdetr_prediction["confidence"],
        },
        "teacher_predictions": {
            "yolo26x": describe_teacher_prediction(yolo_prediction),
            "rfdetr_large": describe_teacher_prediction(rfdetr_prediction),
        },
        "teacher_count": 2,
        "matched_iou": matched_iou,
        "box_source": box_source,
        "merge_status": merge_status,
        "review_reason": "class_conflict" if has_class_conflict else None,
    }


def keep_unmatched_prediction(prediction: dict, teacher_name: str) -> dict:
    """Keep a single-teacher proposal as an explicit review candidate."""
    return {
        "box_xyxy": list(prediction["box_xyxy"]),
        "class_name": prediction["class_name"],
        "has_class_conflict": False,
        "teacher_classes": {teacher_name: prediction["class_name"]},
        "teacher_confidences": {teacher_name: prediction["confidence"]},
        "teacher_predictions": {
            teacher_name: describe_teacher_prediction(prediction)
        },
        "teacher_count": 1,
        "matched_iou": None,
        "box_source": teacher_name,
        "merge_status": f"{teacher_name}_only",
        "review_reason": "single_teacher_only",
    }


def describe_component_prediction(prediction: dict) -> dict:
    """Preserve enough information to reconstruct an absorbed truck proposal."""
    return {
        "box_xyxy": prediction["box_xyxy"],
        "class_name": prediction["class_name"],
        "merge_status": prediction["merge_status"],
        "teacher_count": prediction["teacher_count"],
        "teacher_classes": prediction["teacher_classes"],
        "teacher_confidences": prediction["teacher_confidences"],
        "teacher_predictions": prediction["teacher_predictions"],
    }


def consolidate_truck_fragments(
    predictions: list[dict], settings: dict
) -> tuple[list[dict], dict[str, int]]:
    """Collapse truck fragments around a largest-box anchor without chaining."""
    truck_indices = sorted(
        (
            index
            for index, prediction in enumerate(predictions)
            if prediction["class_name"] == "truck"
        ),
        key=lambda index: calculate_box_area(predictions[index]["box_xyxy"]),
        reverse=True,
    )
    remaining_truck_indices = set(truck_indices)
    consolidated_trucks = []
    statistics = {
        "fragmented_truck_group_count": 0,
        "absorbed_truck_prediction_count": 0,
        "truck_geometry_conflict_count": 0,
        "preserved_distinct_large_truck_pair_count": 0,
    }
    for anchor_index in truck_indices:
        if anchor_index not in remaining_truck_indices:
            continue
        anchor = predictions[anchor_index]
        group_indices = [anchor_index]
        for candidate_index in truck_indices:
            if (
                candidate_index == anchor_index
                or candidate_index not in remaining_truck_indices
            ):
                continue
            candidate_box = predictions[candidate_index]["box_xyxy"]
            if any(
                should_keep_large_trucks_separate(
                    predictions[group_index]["box_xyxy"], candidate_box, settings
                )
                for group_index in group_indices
            ):
                statistics["preserved_distinct_large_truck_pair_count"] += 1
                continue
            if are_aligned_truck_fragments(
                anchor["box_xyxy"],
                predictions[candidate_index]["box_xyxy"],
                settings,
            ):
                group_indices.append(candidate_index)

        for group_index in group_indices:
            remaining_truck_indices.remove(group_index)
        if len(group_indices) == 1:
            consolidated_trucks.append(anchor)
            continue

        group = [predictions[index] for index in group_indices]
        union_box = calculate_union_box([item["box_xyxy"] for item in group])
        anchor_coverage = calculate_box_area(anchor["box_xyxy"]) / calculate_box_area(
            union_box
        )
        consolidated = dict(anchor)
        consolidated["component_predictions"] = [
            describe_component_prediction(item) for item in group
        ]
        consolidated["component_count"] = len(group)
        consolidated["component_teacher_names"] = sorted(
            {
                teacher_name
                for item in group
                for teacher_name in item["teacher_predictions"]
            }
        )
        consolidated["recommended_box_xyxy"] = union_box
        consolidated["anchor_union_coverage"] = anchor_coverage
        statistics["fragmented_truck_group_count"] += 1
        statistics["absorbed_truck_prediction_count"] += len(group) - 1
        if anchor_coverage >= settings["anchor_union_coverage"]:
            consolidated["merge_status"] = "truck_cluster_anchor_kept"
            consolidated["geometry_status"] = "anchor_kept"
        else:
            consolidated["merge_status"] = "geometry_conflict"
            consolidated["geometry_status"] = "fragmented_truck"
            consolidated["review_reason"] = "fragmented_truck"
            statistics["truck_geometry_conflict_count"] += 1
        consolidated_trucks.append(consolidated)

    non_trucks = [
        prediction for prediction in predictions if prediction["class_name"] != "truck"
    ]
    return non_trucks + consolidated_trucks, statistics


def add_frame_metadata(
    predictions: list[dict], row: dict[str, str]
) -> list[dict]:
    """Attach image provenance and deterministic per-frame IDs."""
    ordered = sorted(
        predictions,
        key=lambda item: (
            item["box_xyxy"][1],
            item["box_xyxy"][0],
            item["box_xyxy"][3],
            item["box_xyxy"][2],
            item["class_name"],
        ),
    )
    for index, prediction in enumerate(ordered):
        prediction.update(
            {
                "prediction_id": f"{row['frame_key']}:prediction_tree:{index:04d}",
                "frame_key": row["frame_key"],
                "relative_image_path": row["relative_image_path"],
                "source_video_relative_path": row["source_video_relative_path"],
                "split": row["split"],
                "source_image_width_pixels": int(row["width"]),
                "source_image_height_pixels": int(row["height"]),
            }
        )
    return ordered


def draw_dashed_rectangle(
    drawing: ImageDraw.ImageDraw,
    box: list[float],
    color: str,
    line_width: int,
    dash_length: int,
    gap_length: int,
) -> None:
    """Draw an axis-aligned dashed rectangle."""
    left, top, right, bottom = box
    step = dash_length + gap_length
    for start in np.arange(left, right, step):
        drawing.line(
            (start, top, min(start + dash_length, right), top),
            fill=color,
            width=line_width,
        )
        drawing.line(
            (start, bottom, min(start + dash_length, right), bottom),
            fill=color,
            width=line_width,
        )
    for start in np.arange(top, bottom, step):
        drawing.line(
            (left, start, left, min(start + dash_length, bottom)),
            fill=color,
            width=line_width,
        )
        drawing.line(
            (right, start, right, min(start + dash_length, bottom)),
            fill=color,
            width=line_width,
        )


def load_overlay_font(image: Image.Image, overlay_config: dict) -> ImageFont.ImageFont:
    """Load a scalable Cyrillic-capable font when one is available."""
    font_size = max(
        overlay_config["minimum_font_size_pixels"],
        round(max(image.size) / overlay_config["font_size_scale_divisor"]),
    )
    for font_candidate in OVERLAY_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def get_overlay_state(prediction: dict, overlay_config: dict) -> tuple[str, str]:
    """Return the visual class and teacher-support state for one prediction."""
    if prediction["has_class_conflict"]:
        return "unknown", "unknown"

    visual_class = overlay_config["visual_class_aliases"].get(
        prediction["class_name"], "unknown"
    )
    if visual_class == "unknown":
        return "unknown", "unknown"

    teacher_names = set(
        prediction.get(
            "component_teacher_names", prediction["teacher_predictions"].keys()
        )
    )
    if teacher_names == set(TEACHER_NAMES):
        return visual_class, "both"
    if teacher_names == {"yolo26x"}:
        return visual_class, "yolo26x"
    if teacher_names == {"rfdetr_large"}:
        return visual_class, "rfdetr_large"
    return "unknown", "unknown"


def get_overlay_label(
    visual_class: str, support_state: str, overlay_config: dict
) -> str:
    """Build the compact reviewer-facing label requested for the overlay."""
    if visual_class == "unknown":
        return overlay_config["labels"]["unknown"]
    class_label = overlay_config["labels"][visual_class]
    if support_state == "both":
        return class_label
    return f"{class_label}_{overlay_config['teacher_labels'][support_state]}"


def draw_predictions(
    image: Image.Image, predictions: list[dict], overlay_config: dict
) -> Image.Image:
    """Draw merged proposals on a full-resolution image copy."""
    overlay = image.copy()
    drawing = ImageDraw.Draw(overlay)
    font = load_overlay_font(image, overlay_config)
    line_width = max(
        overlay_config["minimum_line_width_pixels"],
        round(max(image.size) / overlay_config["line_width_scale_divisor"]),
    )
    text_padding = overlay_config["text_padding_pixels"]
    for prediction in predictions:
        box = prediction["box_xyxy"]
        visual_class, support_state = get_overlay_state(prediction, overlay_config)
        color = overlay_config["colors"][visual_class][support_state]
        label = get_overlay_label(visual_class, support_state, overlay_config)
        if visual_class == "unknown":
            draw_dashed_rectangle(
                drawing,
                box,
                color,
                line_width,
                overlay_config["dash_length_pixels"],
                overlay_config["gap_length_pixels"],
            )
        else:
            drawing.rectangle(box, outline=color, width=line_width)
        label_box = drawing.textbbox((box[0], box[1]), label, font=font)
        padded_label_box = (
            label_box[0] - text_padding,
            label_box[1] - text_padding,
            label_box[2] + text_padding,
            label_box[3] + text_padding,
        )
        drawing.rectangle(padded_label_box, fill=color)
        text_color = "white" if visual_class == "unknown" else "black"
        drawing.text((box[0], box[1]), label, fill=text_color, font=font)
    return overlay


def merge_frame_predictions(
    yolo_predictions: list[dict],
    rfdetr_predictions: list[dict],
    config: dict,
) -> tuple[list[dict], dict]:
    """Merge one frame and return review statistics."""
    matches, matched_yolo, matched_rfdetr = match_predictions(
        yolo_predictions,
        rfdetr_predictions,
        config["matching"]["minimum_iou"],
    )
    merged = [
        fuse_matched_predictions(
            yolo_predictions[yolo_index],
            rfdetr_predictions[rfdetr_index],
            matched_iou,
            config,
        )
        for yolo_index, rfdetr_index, matched_iou in matches
    ]
    if config["matching"]["keep_unmatched_predictions"]:
        merged.extend(
            keep_unmatched_prediction(prediction, "yolo26x")
            for index, prediction in enumerate(yolo_predictions)
            if index not in matched_yolo
        )
        merged.extend(
            keep_unmatched_prediction(prediction, "rfdetr_large")
            for index, prediction in enumerate(rfdetr_predictions)
            if index not in matched_rfdetr
        )
    geometry_statistics = {
        "fragmented_truck_group_count": 0,
        "absorbed_truck_prediction_count": 0,
        "truck_geometry_conflict_count": 0,
        "preserved_distinct_large_truck_pair_count": 0,
    }
    if config.get("truck_geometry", {}).get("enabled", False):
        merged, geometry_statistics = consolidate_truck_fragments(
            merged, config["truck_geometry"]
        )
    statistics = {
        "yolo26x_input_count": len(yolo_predictions),
        "rfdetr_large_input_count": len(rfdetr_predictions),
        "matched_count": len(matches),
        "near_identical_count": sum(
            matched_iou >= config["matching"]["near_identical_iou"]
            for _, _, matched_iou in matches
        ),
        "moderate_overlap_count": sum(
            matched_iou < config["matching"]["near_identical_iou"]
            for _, _, matched_iou in matches
        ),
        "class_conflict_count": sum(
            item["has_class_conflict"] for item in merged
        ),
        "yolo26x_only_count": len(yolo_predictions) - len(matched_yolo),
        "rfdetr_large_only_count": len(rfdetr_predictions) - len(matched_rfdetr),
        **geometry_statistics,
        "output_count": len(merged),
    }
    return merged, statistics


def main() -> None:
    """Build the complete prediction tree and full-resolution overlays."""
    arguments = parse_arguments()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    output_directory = resolve_project_path(config["output_directory"])
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Prediction tree already exists: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = output_directory / "config_snapshot.yaml"
    shutil.copyfile(arguments.config, config_snapshot_path)

    input_paths = {
        teacher_name: resolve_project_path(config["inputs"][teacher_name])
        for teacher_name in TEACHER_NAMES
    }
    indexed_predictions = {
        teacher_name: index_predictions(read_json(input_paths[teacher_name]))
        for teacher_name in TEACHER_NAMES
    }
    manifest_path = resolve_project_path(config["input_manifest"])
    manifest_rows = read_manifest(manifest_path)

    all_predictions = []
    frame_statistics = []
    totals = Counter()
    overlay_directory = output_directory / "overlays_full_resolution"
    overlay_directory.mkdir(parents=True, exist_ok=True)
    for frame_index, row in enumerate(manifest_rows, start=1):
        frame_key = row["frame_key"]
        merged, statistics = merge_frame_predictions(
            indexed_predictions["yolo26x"].get(frame_key, []),
            indexed_predictions["rfdetr_large"].get(frame_key, []),
            config,
        )
        merged = add_frame_metadata(merged, row)
        all_predictions.extend(merged)
        statistics = {"frame_key": frame_key, **statistics}
        frame_statistics.append(statistics)
        totals.update({key: value for key, value in statistics.items() if key != "frame_key"})

        image_path = resolve_project_path(row["relative_image_path"])
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        overlay_path = overlay_directory / f"{image_path.stem}.jpg"
        draw_predictions(image, merged, config["overlay"]).save(
            overlay_path,
            quality=config["overlay"]["jpeg_quality"],
            subsampling=config["overlay"]["jpeg_subsampling"],
        )
        print(
            f"{frame_index:03d} {frame_key}: "
            f"{statistics['matched_count']} matched, "
            f"{statistics['class_conflict_count']} conflicts, "
            f"{statistics['output_count']} output",
            flush=True,
        )

    summary = {
        "run_id": config["run_id"],
        "frame_count": len(manifest_rows),
        "prediction_count": len(all_predictions),
        "class_counts": dict(
            sorted(Counter(item["class_name"] for item in all_predictions).items())
        ),
        "totals": dict(totals),
        "matching": config["matching"],
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": calculate_sha256(manifest_path),
        "config_path": str(config_snapshot_path.resolve()),
        "config_sha256": calculate_sha256(config_snapshot_path),
        "input_predictions": {
            teacher_name: {
                "path": str(file_path.resolve()),
                "sha256": calculate_sha256(file_path),
            }
            for teacher_name, file_path in input_paths.items()
        },
        "source_images_preserved_at_original_resolution": True,
    }
    write_json(output_directory / "predictions.json", all_predictions)
    write_json(output_directory / "frame_statistics.json", frame_statistics)
    write_json(output_directory / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
