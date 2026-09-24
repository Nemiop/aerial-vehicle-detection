"""Create a bounded one-class vehicle detection dataset from COCO 2017.

Only images containing car, motorcycle, bus, or truck annotations are fetched.
All selected boxes become YOLO class 0 (``vehicle``); other COCO objects are
not labels or negatives. The result is deliberately separate from the aerial
project dataset so it can be used strictly as a pre-training stage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import yaml


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
# COCO's official download examples use this HTTP endpoint. Some Pod routes
# currently present a certificate for a different hostname at the HTTPS URL.
ANNOTATION_ARCHIVE_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/{split}2017/{file_name}"


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIRECTORY / candidate


def download(url: str, destination: Path, retries: int = 3) -> None:
    """Download one file atomically, retrying transient HTTP failures."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "vehicle-coco-pretrain/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as file:
                shutil.copyfileobj(response, file)
            temporary.replace(destination)
            return
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"Could not download {url}: {error}") from error


def ensure_annotations(root: Path) -> Path:
    annotation_directory = root / "annotations"
    expected = annotation_directory / "instances_train2017.json"
    if expected.is_file() and (annotation_directory / "instances_val2017.json").is_file():
        return annotation_directory
    archive = root / "downloads" / "annotations_trainval2017.zip"
    download(ANNOTATION_ARCHIVE_URL, archive)
    with zipfile.ZipFile(archive) as source:
        for member in ("annotations/instances_train2017.json", "annotations/instances_val2017.json"):
            source.extract(member, root)
    archive.unlink(missing_ok=True)
    return annotation_directory


def selected_records(annotation_path: Path, categories: set[str], maximum: int) -> list[tuple[dict, list[dict]]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_ids = {item["id"] for item in payload["categories"] if item["name"] in categories}
    if len(category_ids) != len(categories):
        found = {item["name"] for item in payload["categories"] if item["id"] in category_ids}
        raise ValueError(f"COCO categories not found: {sorted(categories - found)}")
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        if annotation["category_id"] in category_ids and annotation.get("iscrowd", 0) == 0:
            x, y, width, height = annotation["bbox"]
            if width > 1 and height > 1:
                annotations_by_image[annotation["image_id"]].append(annotation)
    images = sorted(
        (image for image in payload["images"] if image["id"] in annotations_by_image),
        key=lambda image: image["file_name"],
    )
    if maximum > 0:
        images = images[:maximum]
    return [(image, annotations_by_image[image["id"]]) for image in images]


def yolo_lines(image: dict, annotations: list[dict]) -> str:
    width, height = image["width"], image["height"]
    lines = []
    for annotation in annotations:
        x, y, box_width, box_height = annotation["bbox"]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
        if x2 <= x1 or y2 <= y1:
            continue
        center_x, center_y = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        normalized_width, normalized_height = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"0 {center_x:.8f} {center_y:.8f} {normalized_width:.8f} {normalized_height:.8f}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_split(root: Path, split: str, records: list[tuple[dict, list[dict]]], workers: int) -> dict:
    image_directory = root / "images" / split
    label_directory = root / "labels" / split
    image_directory.mkdir(parents=True, exist_ok=True)
    label_directory.mkdir(parents=True, exist_ok=True)

    def fetch(record: tuple[dict, list[dict]]) -> tuple[str, int]:
        image, annotations = record
        image_path = image_directory / image["file_name"]
        if not image_path.is_file() or image_path.stat().st_size == 0:
            download(IMAGE_BASE_URL.format(split="train" if split == "train" else "val", file_name=image["file_name"]), image_path)
        label_path = label_directory / f"{Path(image['file_name']).stem}.txt"
        label_path.write_text(yolo_lines(image, annotations), encoding="utf-8")
        return image["file_name"], len(annotations)

    downloaded = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(fetch, records), start=1):
            downloaded.append(result)
            if index % 250 == 0 or index == len(records):
                print(f"{split}: {index}/{len(records)} images ready", flush=True)
    return {"images": len(downloaded), "boxes": sum(count for _, count in downloaded)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a one-class COCO vehicle subset.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIRECTORY / "configs" / "coco_vehicle_pretrain.yaml")
    arguments = parser.parse_args()
    config = read_config(arguments.config)
    dataset = config["dataset"]
    root = project_path(dataset["root"])
    annotation_directory = ensure_annotations(root)
    categories = set(dataset["coco_categories"])
    summary = {"categories": sorted(categories), "output_class": dataset["output_class"], "splits": {}}
    for split, maximum in (("train", dataset["max_train_images"]), ("val", dataset["max_validation_images"])):
        records = selected_records(annotation_directory / f"instances_{split}2017.json", categories, maximum)
        summary["splits"][split] = write_split(root, split, records, dataset["download_workers"])
    (root / "data.yaml").write_text(
        f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: vehicle\n",
        encoding="utf-8",
    )
    summary["config_sha256"] = hashlib.sha256(arguments.config.read_bytes()).hexdigest()
    (root / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
