"""Render pseudo-label versus reviewed CVAT annotations for A/B/C/D frames."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "pdfs" / "cvat_pseudo_vs_reviewed.jpg"
AUDIT_OUT = ROOT / "tmp" / "pdfs" / "cvat_review_audit.json"
XML = ROOT / "Data" / "Labels" / "reviewed" / "filtered" / "annotations.xml"
FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
SMALL = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 28)
VIDEOS = ("A", "B", "C", "D")
ARCHIVES = {
    "A": "train_A_prediction_tree_v6_coco.zip",
    "B": "train_B_rfdetr_1280_tiles_coco.zip",
    "C": "train_C_rfdetr_1280_tiles_coco.zip",
    "D": "train_D_rfdetr_deduplicated_v3_coco.zip",
}
FINAL_LABELS = ROOT / "Data" / "Labels" / "final_vehicle" / "20260923_vehicle_yolo_v1" / "labels" / "val"


def source_image(letter: str) -> Path:
    folder = ROOT / "Data" / "Frames" / f"train{letter}"
    return sorted(folder.glob(f"train_{letter}_frame_*.jpg"))[0]


def pseudo_payload(letter: str) -> dict:
    archive = ROOT / "My Artifacts" / "cvat_import" / ARCHIVES[letter]
    with zipfile.ZipFile(archive) as zf:
        return json.loads(zf.read("annotations/instances_default.json"))


def pseudo_boxes(payload: dict, filename: str) -> list[list[float]]:
    image_id = next(item["id"] for item in payload["images"] if item["file_name"] == filename)
    return [item["bbox"] for item in payload["annotations"] if item["image_id"] == image_id]


def reviewed_boxes() -> dict[str, list[list[float]]]:
    tree = ET.parse(XML)
    records: dict[str, list[list[float]]] = {}
    for image in tree.findall("image"):
        name = image.attrib["name"]
        if name.endswith("_1.jpg") or name.endswith("_2.jpg"):
            continue
        boxes = []
        for box in image.findall("box"):
            boxes.append([float(box.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr")])
        records[name] = boxes
    return records


def final_d_boxes(path: Path) -> list[list[float]]:
    image = Image.open(path)
    label_path = FINAL_LABELS / f"{path.stem}.txt"
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        _, cx, cy, width, height = map(float, line.split())
        w, h = width * image.width, height * image.height
        boxes.append([cx * image.width - w / 2, cy * image.height - h / 2, cx * image.width + w / 2, cy * image.height + h / 2])
    return boxes


def render(path: Path, boxes: list[list[float]], color: tuple[int, int, int], source: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for values in boxes:
        if source == "pseudo":
            x, y, w, h = values; box = (x, y, x + w, y + h)
        else:
            box = tuple(values)
        draw.rectangle(box, outline=color, width=max(3, image.width // 450))
    image.thumbnail((440, 248), Image.Resampling.LANCZOS)
    return image


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a; bx1, by1, bx2, by2 = b
    bx, by, bw, bh = bx1, by1, bx2 - bx1, by2 - by1
    left, top, right, bottom = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def audit(pseudo: list[list[float]], manual: list[list[float]], threshold: float = .5) -> dict[str, int]:
    available = set(range(len(manual)))
    matched = 0
    for box in pseudo:
        candidate = max(available, key=lambda idx: iou(box, manual[idx]), default=None)
        if candidate is not None and iou(box, manual[candidate]) >= threshold:
            available.remove(candidate); matched += 1
    return {"proposed": len(pseudo), "kept_tp": matched, "removed_fp": len(pseudo) - matched, "added_fn": len(available), "reviewed": len(manual)}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    reviewed = reviewed_boxes()
    cells: list[tuple[str, Image.Image, Image.Image]] = []
    audit_summary = {}
    for letter in VIDEOS:
        image_path = source_image(letter)
        filename = image_path.name
        payload = pseudo_payload(letter)
        proposed = pseudo_boxes(payload, filename)
        approved = final_d_boxes(image_path) if letter == "D" else reviewed[filename]
        cells.append((letter, render(image_path, proposed, (44, 102, 205), "pseudo"), render(image_path, approved, (0, 180, 112), "reviewed")))
        cumulative = {"proposed": 0, "kept_tp": 0, "removed_fp": 0, "added_fn": 0, "reviewed": 0}
        for record in payload["images"]:
            name = record["file_name"]
            image_file = source_image(letter).parent / name
            approved = final_d_boxes(image_file) if letter == "D" else reviewed[name]
            row = audit(pseudo_boxes(payload, name), approved)
            for key, value in row.items():
                cumulative[key] += value
        cumulative["review_mode"] = "RF-DETR final validation source (no CVAT edits)" if letter == "D" else "CVAT reviewed"
        audit_summary[letter] = cumulative
    canvas = Image.new("RGB", (4 * 470, 2 * 290), "white")
    draw = ImageDraw.Draw(canvas)
    for col, (letter, pseudo, manual) in enumerate(cells):
        x = col * 470 + 10
        draw.text((x, 4), f"{letter}  pseudo-label", font=SMALL, fill=(24, 51, 79))
        canvas.paste(pseudo, (x, 38))
        label = f"{letter}  RF-DETR final" if letter == "D" else f"{letter}  CVAT reviewed"
        draw.text((x, 294), label, font=SMALL, fill=(24, 51, 79))
        canvas.paste(manual, (x, 328))
    canvas.save(OUT, quality=92)
    AUDIT_OUT.write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
