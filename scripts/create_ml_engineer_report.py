"""Build the four-page ML Engineer vehicle-detector project report."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "ml_engineer_vehicle_detector_report.pdf"
ART = ROOT / "My Artifacts"
COCO_RESULTS = ART / "student_model" / "experiments" / "overnight_20260924_coco" / "results.csv"
AERIAL_RESULTS = ART / "student_model" / "experiments" / "overnight_20260924_aerial" / "results.csv"
INFERENCE = ART / "inference" / "overnight_20260924_aerial_full_frames" / "summary.json"
RAW_FRAME = ROOT / "Data" / "Frames" / "trainD" / "train_D_frame_00000360.jpg"
PSEUDO = ART / "pseudo_annotation" / "train_A_top_roi_before_after.jpg"
REVIEW_LABELS = ROOT / "Data" / "Labels" / "reviewed" / "val_batch2_labels.jpg"
REVIEW_PREDS = ROOT / "Data" / "Labels" / "reviewed" / "val_batch2_pred.jpg"
COCO_PLOT = ART / "student_model" / "experiments" / "overnight_20260924_coco" / "results.png"
AERIAL_PLOT = ART / "student_model" / "experiments" / "overnight_20260924_aerial" / "results.png"
CVAT_GRID = ROOT / "tmp" / "pdfs" / "cvat_pseudo_vs_reviewed.jpg"
CVAT_AUDIT = ROOT / "tmp" / "pdfs" / "cvat_review_audit.json"
YOLO_PILOT = ART / "pseudo_annotation" / "pilot_v2" / "review" / "yolo26x_full_frame.jpg"
RFDETR_PILOT = ART / "pseudo_annotation" / "pilot_v2" / "review" / "rfdetr_large_full_frame.jpg"
COCO_PRED_VIEW = Path(r"C:\Users\Ruslan\Downloads\Telegram Desktop\photo_2026-09-24_11-51-31.jpg")
COCO_TARGET_VIEW = Path(r"C:\Users\Ruslan\Downloads\Telegram Desktop\photo_2026-09-24_11-51-28.jpg")
TRAIN_C_INFERENCE = ART / "inference" / "overnight_20260924_aerial_full_frames" / "train_C_frame_00000202_full.jpg"
TRUCK_FRAGMENT = Path(r"C:\Users\Ruslan\AppData\Local\Temp\codex-clipboard-ff54ff37-34d8-44c8-b110-b3dd62f1e06b.png")
BACKGROUND_FP = Path(r"C:\Users\Ruslan\AppData\Local\Temp\codex-clipboard-737f3a9f-1d84-4228-825c-de196692b1a2.png")
CAR_DUPLICATES = Path(r"C:\Users\Ruslan\AppData\Local\Temp\codex-clipboard-a0945fe1-292f-4a5b-9412-c7ecd99efaf0.png")

NAVY, BLUE, MUTED = colors.HexColor("#18334F"), colors.HexColor("#2D6FBB"), colors.HexColor("#8290A2")
LINE, PALE, TEXT = colors.HexColor("#D7DEE7"), colors.HexColor("#F3F6F9"), colors.HexColor("#263544")
PAGE_W, PAGE_H = A4
LEFT, RIGHT, TOP = 20 * mm, 20 * mm, 18 * mm
CONTENT_W = PAGE_W - LEFT - RIGHT
styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.1, leading=13.0, textColor=TEXT, spaceAfter=3)
small = ParagraphStyle("small", parent=body, fontSize=7.6, leading=10.1, textColor=MUTED)
cell = ParagraphStyle("cell", parent=body, fontSize=7.8, leading=9.7, spaceAfter=0)


def header(c: canvas.Canvas, page: int) -> None:
    c.setStrokeColor(LINE); c.setLineWidth(.55)
    c.line(LEFT, PAGE_H - TOP + 3 * mm, PAGE_W - RIGHT, PAGE_H - TOP + 3 * mm)
    c.setFillColor(MUTED); c.setFont("Helvetica", 7.5)
    c.drawString(LEFT, 10 * mm, "Aerial vehicle detection - ML Engineer test task")
    c.drawRightString(PAGE_W - RIGHT, 10 * mm, f"{page} / 4")


def title(c: canvas.Canvas, heading: str, subtitle: str) -> float:
    y = PAGE_H - TOP - 10 * mm
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 24); c.drawString(LEFT, y, heading)
    c.setStrokeColor(BLUE); c.setLineWidth(2.1); c.line(LEFT, y - 8 * mm, LEFT + 18 * mm, y - 8 * mm)
    c.setFillColor(MUTED); c.setFont("Helvetica", 9.5); c.drawString(LEFT, y - 15 * mm, subtitle)
    return y - 25 * mm


def para(c: canvas.Canvas, text: str, x: float, y: float, width: float, style: ParagraphStyle = body) -> float:
    p = Paragraph(text, style); _, h = p.wrap(width, 210 * mm); p.drawOn(c, x, y - h); return y - h


def section(c: canvas.Canvas, label: str, y: float) -> float:
    c.setFillColor(BLUE); c.setFont("Helvetica-Bold", 8.2); c.drawString(LEFT, y, label.upper())
    c.setStrokeColor(LINE); c.setLineWidth(.5); c.line(LEFT, y - 3.2 * mm, PAGE_W - RIGHT, y - 3.2 * mm)
    return y - 8 * mm


def stage(c: canvas.Canvas, number: int, name: str, y: float) -> float:
    c.setFillColor(BLUE); c.roundRect(LEFT, y - 5.8 * mm, 8 * mm, 8 * mm, 1.6 * mm, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 8); c.drawCentredString(LEFT + 4 * mm, y - 3.3 * mm, str(number))
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 10); c.drawString(LEFT + 12 * mm, y - 1.8 * mm, name)
    return y - 9 * mm


def tbl(c: canvas.Canvas, values: list[list[str]], widths: list[float], y: float) -> float:
    data = [[Paragraph(v, cell) for v in row] for row in values]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED), ("TEXTCOLOR", (0, 1), (-1, -1), TEXT), ("LINEBELOW", (0, 0), (-1, -1), .35, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    _, h = t.wrap(CONTENT_W, 210 * mm); t.drawOn(c, LEFT, y - h); return y - h - 3 * mm


def img(path: Path, max_w: float, max_h: float) -> Image:
    item = Image(str(path)); ratio = min(max_w / item.imageWidth, max_h / item.imageHeight)
    item.drawWidth, item.drawHeight = item.imageWidth * ratio, item.imageHeight * ratio
    return item


def metrics(path: Path, best: bool = False) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as file: rows = list(csv.DictReader(file))
    return max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"])) if best else rows[-1]


def augmentation_cards(c: canvas.Canvas, y: float) -> float:
    cards = [
        ("Frame scale", "0.80-2.00x"), ("Horizontal flip", "p = 0.50"), ("Mosaic", "p = 0.25"),
        ("Brightness", "+/- 0.20"), ("Contrast", "+/- 0.20"), ("Saturation", "+/- 0.25"),
        ("Hue", "+/- 0.015"), ("Gaussian blur", "p = 0.05"), ("JPEG compression", "p = 0.10"),
    ]
    gap, width, height = 3 * mm, (CONTENT_W - 2 * 3 * mm) / 3, 12 * mm
    for index, (name, value) in enumerate(cards):
        row, col = divmod(index, 3)
        x, top = LEFT + col * (width + gap), y - row * (height + gap)
        c.setFillColor(PALE); c.roundRect(x, top - height, width, height, 1.2 * mm, fill=1, stroke=0)
        c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 7.0); c.drawString(x + 2.2 * mm, top - 4.3 * mm, name)
        c.setFillColor(MUTED); c.setFont("Helvetica", 6.8); c.drawString(x + 2.2 * mm, top - 8.5 * mm, value)
    return y - 3 * height - 2 * gap


def pipeline(c: canvas.Canvas, y: float) -> float:
    labels = [("Full frame", "variable resolution"), ("Adaptive tiles", "1280 px, 20% overlap"), ("YOLO26n", "960 px per tile"), ("Global merge", "shift + NMS IoU 0.30"), ("Full overlay", "one image / video")]
    gap, width, height = 3 * mm, 31 * mm, 18 * mm
    for index, (name, detail) in enumerate(labels):
        x = LEFT + index * (width + gap)
        c.setFillColor(PALE if index not in (2, 4) else colors.HexColor("#E7F0FA"))
        c.roundRect(x, y - height, width, height, 1.5 * mm, fill=1, stroke=0)
        c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 7.0); c.drawCentredString(x + width / 2, y - 6 * mm, name)
        c.setFillColor(MUTED); c.setFont("Helvetica", 6.1); c.drawCentredString(x + width / 2, y - 10.8 * mm, detail)
        if index < len(labels) - 1:
            c.setStrokeColor(BLUE); c.setLineWidth(1.1)
            c.line(x + width + .8 * mm, y - height / 2, x + width + gap - .8 * mm, y - height / 2)
    return y - height - 4 * mm


def page_one(c: canvas.Canvas) -> None:
    header(c, 1); y = title(c, "From video to Bird View data", "Stages 1-2: inspection, frame cutting and pseudo-annotation")
    y = stage(c, 1, "Bird View inspection and frame cutting", y)
    y = para(c, "Four development videos were first inspected as aerial Bird View sources. Frames were sampled at a stable cadence and stored with source-video identity, timestamp and split metadata. The held-out <b>Evaluation</b> video was not used to create the training data.", LEFT, y, CONTENT_W)
    y = tbl(c, [["SOURCE", "FRAMES", "ROLE"], ["train A", "39", "training"], ["train B", "63", "training"], ["train C", "34", "training"], ["train D", "49", "validation"], ["total", "185", "136 train / 49 validation"]], [38 * mm, 26 * mm, CONTENT_W - 64 * mm], y)
    raw = img(RAW_FRAME, 82 * mm, 42 * mm); raw.drawOn(c, LEFT, y - raw.drawHeight)
    c.setFillColor(MUTED); c.setFont("Helvetica", 7.4); c.drawString(LEFT, y - raw.drawHeight - 3 * mm, "Bird View example: dense intersection, scale variation and small road vehicles")
    y -= raw.drawHeight + 11 * mm
    y = stage(c, 2, "Pseudo-annotation: two teachers, tiles and merge", y)
    y = para(c, "YOLO26x and RF-DETR Large were run independently, first on complete frames and then as full-frame-plus-tiles variants. The pilot deliberately measures candidate coverage and teacher agreement, not accuracy: reviewed ground truth did not yet exist at this point.", LEFT, y, CONTENT_W)
    left, right = img(YOLO_PILOT, 80 * mm, 41 * mm), img(RFDETR_PILOT, 80 * mm, 41 * mm)
    left.drawOn(c, LEFT, y - left.drawHeight); right.drawOn(c, LEFT + 94 * mm, y - right.drawHeight)
    c.setFillColor(MUTED); c.setFont("Helvetica", 7.2); c.drawString(LEFT, y - left.drawHeight - 2.7 * mm, "YOLO26x: 12 frozen pilot frames"); c.drawString(LEFT + 94 * mm, y - right.drawHeight - 2.7 * mm, "RF-DETR Large: same pilot frames")
    y -= max(left.drawHeight, right.drawHeight) + 8 * mm
    y = tbl(c, [["TEACHER / VARIANT", "CANDIDATES", "CROSS-TEACHER AGREEMENT"], ["YOLO26x, full frame", "348", "207 same-class matches at IoU 0.50"], ["YOLO26x, full + tiles", "543", "356 same-class matches at IoU 0.50"], ["RF-DETR Large, full frame", "361", "pilot comparison"], ["RF-DETR Large, full + tiles", "627", "small-object recovery; more review work"]], [60 * mm, 32 * mm, CONTENT_W - 92 * mm], y)
    c.setFillColor(PALE); c.roundRect(LEFT, y - 24 * mm, CONTENT_W, 24 * mm, 1.5 * mm, fill=1, stroke=0)
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 8.6); c.drawString(LEFT + 4 * mm, y - 5.5 * mm, "What entered CVAT")
    para(c, "A used the merged YOLO26x + RF-DETR v6 proposals, with an extra RF-DETR high-resolution top ROI at threshold 0.08 to recover missed vehicles. B and C were replaced with RF-DETR 1280-tile proposals. D used a separate RF-DETR deduplicated v3 validation source, not the merged teacher tree.", LEFT + 4 * mm, y - 8 * mm, CONTENT_W - 8 * mm, small)
    c.showPage()


def page_two(c: canvas.Canvas, audit: dict) -> None:
    header(c, 2); y = title(c, "CVAT review: predictions to labels", "Stage 3: manual correction of pseudo-labels on complete source frames")
    y = section(c, "Frequent failure modes and extra rules", y)
    y = para(c, "The pseudo-label stage deliberately retained difficult candidates, so a small rule set prevented obvious geometric failures from overwhelming CVAT review.", LEFT, y, CONTENT_W, small)
    illustrations = [(TRUCK_FRAGMENT, "Truck fragments", "anchor merge; preserve distinct adjacent trucks"), (BACKGROUND_FP, "Static-background FP", "review building / trailer-like boxes; no blind acceptance"), (CAR_DUPLICATES, "Tile duplicates", "class-aware NMS plus containment and centre rules")]
    xs = [LEFT, LEFT + 61 * mm, LEFT + 122 * mm]
    max_height = 0.0
    for x, (path, caption, rule) in zip(xs, illustrations):
        picture = img(path, 52 * mm, 29 * mm); picture.drawOn(c, x, y - picture.drawHeight); max_height = max(max_height, picture.drawHeight)
        c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 6.6); c.drawString(x, y - 32 * mm, caption)
        c.setFillColor(MUTED); c.setFont("Helvetica", 5.9); c.drawString(x, y - 35.5 * mm, rule)
    y -= max_height + 10 * mm
    y = para(c, "For elongated trucks, fragment candidates were merged only when containment, longitudinal overlap and anchor-union coverage were consistent; a distinct-neighbour safeguard kept nearby large trucks separate. The train-A high-resolution top ROI additionally used an acceptance polygon, size filters and a lower RF-DETR threshold. Truck was tested as a separate class for a possible distance proxy, but its examples were sparse and inconsistent. The final task therefore uses one <i>vehicle</i> class and no distance metric.", LEFT, y, CONTENT_W, small)
    y = stage(c, 3, "Manual correction in CVAT", y)
    y = para(c, "Pseudo-labels were only a starting point. In <b>CVAT</b>, each proposed box was reviewed on the full source image: false positives were removed, missed vehicles were added, boundaries were corrected, and all accepted objects were normalised to the one target class, <i>vehicle</i>.", LEFT, y, CONTENT_W)
    grid = img(CVAT_GRID, CONTENT_W, 124 * mm); grid.drawOn(c, LEFT, y - grid.drawHeight)
    y -= grid.drawHeight + 5 * mm
    y = section(c, "Review-derived proposal audit", y)
    rows = [["VIDEO", "PROPOSED", "TP KEPT", "FP REMOVED", "FN ADDED", "FINAL"]]
    for letter in ("A", "B", "C", "D"):
        row = audit[letter]
        rows.append([letter, f"{row['proposed']:,}", f"{row['kept_tp']:,}", f"{row['removed_fp']:,}", f"{row['added_fn']:,}", f"{row['reviewed']:,}"])
    y = tbl(c, rows, [14 * mm, 28 * mm, 28 * mm, 32 * mm, 28 * mm, CONTENT_W - 130 * mm], y)
    y = para(c, "TP = a proposal matched a final/review box at IoU >= 0.50; FP = a proposal removed by the final label set; FN = a final/review box without a matching proposal. Boundary-only edits stay matched and are not counted as a separate change.", LEFT, y, CONTENT_W, small)
    c.setFillColor(PALE); c.roundRect(LEFT, y - 22 * mm, CONTENT_W, 22 * mm, 1.5 * mm, fill=1, stroke=0)
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 8.7); c.drawString(LEFT + 5 * mm, y - 5.5 * mm, "Important split distinction")
    para(c, "A-C are CVAT-reviewed training sources. D is not a manual-CVAT correction set: it is the final RF-DETR deduplicated v3 validation source, so its audit has zero review deltas by construction. This preserves video-level validation separation.", LEFT + 5 * mm, y - 8 * mm, CONTENT_W - 10 * mm, small)
    c.showPage()


def page_three(c: canvas.Canvas, coco: dict[str, str], aerial: dict[str, str]) -> None:
    header(c, 3); y = title(c, "Model adaptation and COCO", "Stages 4-5: fit the parameter budget, then learn a vehicle spatial prior")
    y = stage(c, 4, "YOLO26n model adaptation", y)
    y = para(c, "The model was adapted step by step to fit the formal 500k-parameter limit without discarding the high-resolution evidence needed for small aerial vehicles. The final graph predicts directly from P2/4, P3/8 and P4/16.", LEFT, y, CONTENT_W)
    y = tbl(c, [["CHANGE", "PARAMETERS AFTER", "DELTA", "WHY IT HELPS"], ["YOLO26n, 80 classes", "2,572,280", "-", "pretrained starting point"], ["One vehicle class", "2,504,190", "-68,090", "match the task taxonomy"], ["Remove extra end-to-end head", "2,383,407", "-120,783", "one post-processing path is sufficient"], ["Remove P5/32, C2PSA and P5 neck/head", "574,634", "-1,808,773", "free budget from the coarsest scale"], ["Narrow new SPPF and P4 neck to 80 ch.", "492,794", "-81,840", "keep compatible backbone capacity"], ["Add direct P2/4 head", "492,535", "-259", "retain detail for tiny vehicles"]], [57 * mm, 33 * mm, 27 * mm, CONTENT_W - 117 * mm], y)
    y = stage(c, 5, "Spatial-perception pre-training on COCO", y)
    y = para(c, "COCO <i>car, motorcycle, bus</i> and <i>truck</i> were remapped to <i>vehicle</i>. This provides generic vehicle localisation before fine-tuning on the project camera geometry. The run was first attempted locally on an RTX 3060, but the training time was impractical. It was therefore executed on RunPod L40S with persistent storage; MLflow recorded parameters, metrics and artifacts.", LEFT, y, CONTENT_W)
    y = tbl(c, [["DATA", "SETUP", "FINAL VALIDATION"], ["18,175 train / 796 val", "60 epochs, 960 px, AdamW, batch 32", f"mAP50 {float(coco['metrics/mAP50(B)']):.3f}; mAP50-95 {float(coco['metrics/mAP50-95(B)']):.3f}"]], [44 * mm, 68 * mm, CONTENT_W - 112 * mm], y)
    left, right = img(COCO_PRED_VIEW, 80 * mm, 38 * mm), img(COCO_TARGET_VIEW, 80 * mm, 38 * mm)
    left.drawOn(c, LEFT, y - left.drawHeight); right.drawOn(c, LEFT + 94 * mm, y - right.drawHeight)
    c.setFillColor(MUTED); c.setFont("Helvetica", 7.2); c.drawString(LEFT, y - left.drawHeight - 2.7 * mm, "COCO one-class predictions"); c.drawString(LEFT + 94 * mm, y - right.drawHeight - 2.7 * mm, "COCO one-class targets")
    y -= max(left.drawHeight, right.drawHeight) + 9 * mm
    y = section(c, "Online Bird View augmentations", y)
    augmentation_cards(c, y)
    c.showPage()


def page_four(c: canvas.Canvas, inference: dict) -> None:
    header(c, 4); y = title(c, "Bird View training and inference", "Stages 6-7: aerial fine-tuning, adaptive tiled inference and full-frame output")
    y = stage(c, 6, "Fine-tuning on the reviewed Bird View dataset", y)
    y = para(c, "The COCO checkpoint was fine-tuned using A-C for training and D for validation. The planned 120-epoch run ended after 36 epochs; the best D checkpoint was retained. The result is honestly weak: it demonstrates the remaining aerial domain and annotation-quality gap.", LEFT, y, CONTENT_W)
    y = tbl(c, [["AERIAL SPLIT", "SELECTED D VALIDATION", "INTERPRETATION"], ["136 frames A-C / 49 frames D", "best mAP50 0.026; mAP50-95 0.005", "not deployment-ready"]], [53 * mm, 53 * mm, CONTENT_W - 106 * mm], y)
    curve, cframe = img(AERIAL_PLOT, 78 * mm, 41 * mm), img(TRAIN_C_INFERENCE, 78 * mm, 41 * mm)
    curve.drawOn(c, LEFT, y - curve.drawHeight); cframe.drawOn(c, LEFT + 94 * mm, y - cframe.drawHeight)
    c.setFillColor(MUTED); c.setFont("Helvetica", 7.2); c.drawString(LEFT, y - curve.drawHeight - 2.7 * mm, "Aerial learning curves"); c.drawString(LEFT + 94 * mm, y - cframe.drawHeight - 2.7 * mm, "Full-frame inference: train C")
    y -= max(curve.drawHeight, cframe.drawHeight) + 10 * mm
    y = stage(c, 7, "Adaptive tiled inference and global merging", y)
    y = para(c, "A 4K frame is not blindly resized before detection: that would make distant vehicles too small. Instead, the frame is divided into overlapping tiles. The tile count is automatically determined by the input resolution, so the same procedure scales to different cameras. If tiles are not desired, the input can be resized directly; scale augmentation during training makes that a supported operating choice.", LEFT, y, CONTENT_W)
    y = pipeline(c, y - 1 * mm)
    y = tbl(c, [["INFERENCE PARAMETERS", "VALUE"], ["tile / overlap", "1280 px / 20%"], ["model input", "960 px"], ["merge", "global class-agnostic NMS, IoU 0.30"], ["rendered output", "one full-resolution annotated image per video"]], [52 * mm, CONTENT_W - 52 * mm], y)
    counts = "; ".join(f"{item['video'].replace('.mp4', '').replace('train_', '').replace('Evaluation', 'Eval')}: {item['prediction_count']}" for item in inference["frames"])
    y = para(c, f"Sampled full-frame detections at confidence 0.20: {counts}. Evaluation remains held out: without reviewed Evaluation ground truth, no held-out mAP or false-alarm rate is claimed.", LEFT, y, CONTENT_W)
    y = section(c, "Held-out test-frame behaviour", y - 1 * mm)
    eval_frame = next(item for item in inference["frames"] if item["video"] == "Evaluation.mp4")
    overlay = img(Path(eval_frame["overlay"]), 36 * mm, 36 * mm); overlay.drawOn(c, LEFT, y - overlay.drawHeight)
    x, right_y, width = LEFT + 46 * mm, y, CONTENT_W - 46 * mm
    right_y = para(c, "The held-out Evaluation video differs sharply from the Bird View training distribution: nearby vehicles occupy much larger image regions. The small-object-trained detector therefore produces false positives and poorly sized boxes on this frame.", x, right_y, width)
    para(c, "This visual result is intentionally reported as a limitation, not a success metric. It motivates a more diverse reviewed training set and a separate validation protocol for large, near-camera vehicles.", x, right_y - 3 * mm, width)
    c.showPage()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
    c.setTitle("ML Engineer - Vehicle Detector Report"); c.setAuthor("Vehicle detector project")
    page_one(c); page_two(c, json.loads(CVAT_AUDIT.read_text(encoding="utf-8"))); page_three(c, metrics(COCO_RESULTS), metrics(AERIAL_RESULTS, best=True)); page_four(c, json.loads(INFERENCE.read_text(encoding="utf-8")))
    c.save(); print(OUT)


if __name__ == "__main__": main()
