"""Run tiled inference but emit one full-resolution overlay per source video."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    PROJECT_DIRECTORY
    / "My Artifacts"
    / "student_model"
    / "experiments"
    / "overnight_20260924_aerial"
    / "weights"
    / "best.pt"
)
OUTPUT_DIRECTORY = (
    PROJECT_DIRECTORY
    / "My Artifacts"
    / "inference"
    / "overnight_20260924_aerial_full_frames"
)
VIDEO_PATHS = [
    PROJECT_DIRECTORY / "Data" / "Raw" / name
    for name in ("train_A.mp4", "train_B.mp4", "train_C.mp4", "train_D.mp4", "Evaluation.mp4")
]
TILE_SIZE = 1280
TILE_OVERLAP = 0.20
MODEL_INPUT_SIZE = 960
CONFIDENCE_THRESHOLD = 0.20
GLOBAL_NMS_IOU = 0.30


def tile_origins(length: int) -> list[int]:
    if length <= TILE_SIZE:
        return [0]
    stride = max(1, round(TILE_SIZE * (1 - TILE_OVERLAP)))
    return sorted(set(list(range(0, length - TILE_SIZE + 1, stride)) + [length - TILE_SIZE]))


def iou(first: np.ndarray, second: np.ndarray) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    denominator = first_area + second_area - intersection
    return intersection / denominator if denominator else 0.0


def global_nms(predictions: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        box = np.asarray(prediction["box_xyxy"], dtype=np.float32)
        if all(iou(box, np.asarray(existing["box_xyxy"], dtype=np.float32)) <= GLOBAL_NMS_IOU for existing in kept):
            kept.append(prediction)
    return kept


def read_middle_frame(video_path: Path) -> tuple[np.ndarray, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_index = frame_count // 2
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame_bgr = capture.read()
    capture.release()
    if not success:
        raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), frame_index


def predict_full_frame(model, image: np.ndarray, device: int | str) -> list[dict]:
    height, width = image.shape[:2]
    predictions: list[dict] = []
    for top in tile_origins(height):
        for left in tile_origins(width):
            tile = image[top : min(top + TILE_SIZE, height), left : min(left + TILE_SIZE, width)]
            result = model.predict(
                tile,
                imgsz=MODEL_INPUT_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                iou=0.70,
                agnostic_nms=True,
                device=device,
                verbose=False,
            )[0]
            if result.boxes is None:
                continue
            for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                predictions.append(
                    {
                        "box_xyxy": [float(box[0] + left), float(box[1] + top), float(box[2] + left), float(box[3] + top)],
                        "confidence": float(confidence),
                    }
                )
    return global_nms(predictions)


def draw_full_frame(image: np.ndarray, predictions: list[dict]) -> Image.Image:
    rendered = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(rendered)
    for prediction in predictions:
        box = prediction["box_xyxy"]
        label = f"vehicle {prediction['confidence']:.2f}"
        draw.rectangle(box, outline="#19c37d", width=4)
        draw.text((box[0] + 4, max(0, box[1] - 20)), label, fill="#19c37d", stroke_width=1, stroke_fill="#000000")
    return rendered


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Aerial checkpoint is absent: {CHECKPOINT_PATH}")
    from ultralytics import YOLO

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(CHECKPOINT_PATH))
    summary = {"checkpoint": str(CHECKPOINT_PATH), "device": str(device), "tile_size": TILE_SIZE, "tile_overlap": TILE_OVERLAP, "frames": []}
    for video_path in VIDEO_PATHS:
        image, frame_index = read_middle_frame(video_path)
        predictions = predict_full_frame(model, image, device)
        output_path = OUTPUT_DIRECTORY / f"{video_path.stem}_frame_{frame_index:08d}_full.jpg"
        draw_full_frame(image, predictions).save(output_path, quality=95, subsampling=0)
        summary["frames"].append({"video": video_path.name, "frame_index": frame_index, "prediction_count": len(predictions), "overlay": str(output_path), "predictions": predictions})
        print(f"{video_path.name}: {len(predictions)} vehicles -> {output_path.name}", flush=True)
    (OUTPUT_DIRECTORY / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
