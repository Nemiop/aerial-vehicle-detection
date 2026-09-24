"""Online, fixed-size tiles for training the aerial vehicle detector.

Whole source frames are scaled first; the fixed 960x960 tile is then selected at
random x/y.  The tile extraction is the model input, not an extra crop augmentation.
"""

from __future__ import annotations

import csv
import io
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourceFrame:
    image_path: Path
    boxes: np.ndarray  # source pixel xyxy
    frame_key: str


def read_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_source_frames(config: dict, split: str) -> list[SourceFrame]:
    root = PROJECT_DIRECTORY / config["dataset"]["root"]
    rows = list(csv.DictReader((root / "dataset_manifest.csv").open(encoding="utf-8-sig", newline="")))
    frames: list[SourceFrame] = []
    for row in rows:
        if row["split"] != split:
            continue
        with Image.open(root / row["image"]) as image:
            width, height = image.size
        boxes = []
        for line in (root / row["label"]).read_text(encoding="utf-8").splitlines():
            _, cx, cy, box_w, box_h = map(float, line.split())
            boxes.append(((cx - box_w / 2) * width, (cy - box_h / 2) * height, (cx + box_w / 2) * width, (cy + box_h / 2) * height))
        frames.append(SourceFrame(root / row["image"], np.asarray(boxes, dtype=np.float32), row["frame_key"]))
    return frames


def quantile_linear_box_size(frames: list[SourceFrame], quantile: float) -> float:
    """Return the requested quantile of sqrt(box area), in source pixels."""
    sizes = np.concatenate([np.sqrt((f.boxes[:, 2] - f.boxes[:, 0]) * (f.boxes[:, 3] - f.boxes[:, 1])) for f in frames])
    return float(np.quantile(sizes, quantile))


def xyxy_to_normalized_xywh(boxes: np.ndarray, size: int) -> torch.Tensor:
    if len(boxes) == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    result = boxes.copy()
    result[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2 / size
    result[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2 / size
    result[:, 2] = (boxes[:, 2] - boxes[:, 0]) / size
    result[:, 3] = (boxes[:, 3] - boxes[:, 1]) / size
    if not np.all(np.isfinite(result)) or not np.all((result >= 0) & (result <= 1)):
        raise ValueError("Tile transformation produced an invalid normalized bounding box")
    return torch.from_numpy(result)


class OnlineVehicleTileDataset(Dataset):
    """Return a different transformed fixed-size tile for every data-loader call."""

    rect = False

    def __init__(self, config: dict, mode: str, seed: int = 42):
        self.config, self.mode = config, mode
        self.frames = read_source_frames(config, "train" if mode == "train" else "val")
        self.tile_size = config["training"]["sampler"]["source_tile_size_pixels"]
        self.input_size = config["training"]["sampler"]["model_input_size_pixels"]
        self.rng = random.Random(seed + (0 if mode == "train" else 10_000))
        self.minimum_linear_size = quantile_linear_box_size(
            read_source_frames(config, "train"), config["training"]["sampler"]["minimum_scaled_vehicle_size_quantile"]
        )
        self.im_files = [str(frame.image_path) for frame in self.frames]
        self.labels = [{} for _ in self.frames]
        if mode == "train":
            self.length = len(self.frames) * config["training"]["sampler"]["samples_per_source_image_per_epoch"]
        else:
            self.tile_records = self._build_validation_tiles()
            self.length = len(self.tile_records)

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        from ultralytics.data.dataset import YOLODataset
        return YOLODataset.collate_fn(batch)

    def close_mosaic(self, *args, **kwargs) -> None:
        """Compatibility hook required by the Ultralytics trainer."""

    def _random_train_tile(self, frame: SourceFrame, output_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Scale only the selected source window, avoiding a 4K-frame allocation."""
        tile_size = output_size or self.tile_size
        min_scale, max_scale = self.config["training"]["sampler"]["frame_scale_range"]
        with Image.open(frame.image_path) as image:
            source_image = np.asarray(image.convert("RGB"))
        for _ in range(64):
            scale = self.rng.uniform(min_scale, max_scale)
            source_side = max(1, round(tile_size / scale))
            canvas = np.pad(source_image, ((0, max(0, source_side - source_image.shape[0])), (0, max(0, source_side - source_image.shape[1])), (0, 0)), mode="edge")
            x = self.rng.randint(0, canvas.shape[1] - source_side)
            y = self.rng.randint(0, canvas.shape[0] - source_side)
            centres = (frame.boxes[:, :2] + frame.boxes[:, 2:]) / 2
            selected = (centres[:, 0] >= x) & (centres[:, 0] < x + source_side) & (centres[:, 1] >= y) & (centres[:, 1] < y + source_side)
            if not selected.any():
                continue
            chosen = frame.boxes[selected].copy()
            chosen[:, [0, 2]] = (chosen[:, [0, 2]] - x) * scale
            chosen[:, [1, 3]] = (chosen[:, [1, 3]] - y) * scale
            original_area = (chosen[:, 2] - chosen[:, 0]) * (chosen[:, 3] - chosen[:, 1])
            chosen = np.clip(chosen, 0, tile_size)
            visible_area = (chosen[:, 2] - chosen[:, 0]) * (chosen[:, 3] - chosen[:, 1])
            chosen = chosen[visible_area >= original_area * self.config["training"]["geometry_verification"]["minimum_visible_box_fraction"]]
            if len(chosen) == 0:
                continue
            linear_sizes = np.sqrt((chosen[:, 2] - chosen[:, 0]) * (chosen[:, 3] - chosen[:, 1]))
            if scale < 1 and np.any(linear_sizes < self.minimum_linear_size):
                continue
            tile = Image.fromarray(canvas[y : y + source_side, x : x + source_side]).resize((tile_size, tile_size), Image.Resampling.BILINEAR)
            return np.asarray(tile), chosen
        # Sparse frames may not hit a vehicle in 64 uniform draws. Keep x/y random,
        # but draw them from the valid interval that contains one randomly chosen box.
        scale = self.rng.uniform(1.0, max_scale)
        source_side = max(1, round(tile_size / scale))
        canvas = np.pad(source_image, ((0, max(0, source_side - source_image.shape[0])), (0, max(0, source_side - source_image.shape[1])), (0, 0)), mode="edge")
        box = frame.boxes[self.rng.randrange(len(frame.boxes))]
        center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        x = self.rng.randint(max(0, math.ceil(center_x - source_side + 1)), min(canvas.shape[1] - source_side, math.floor(center_x)))
        y = self.rng.randint(max(0, math.ceil(center_y - source_side + 1)), min(canvas.shape[0] - source_side, math.floor(center_y)))
        selected = ((frame.boxes[:, 0] + frame.boxes[:, 2]) / 2 >= x) & ((frame.boxes[:, 0] + frame.boxes[:, 2]) / 2 < x + source_side) & ((frame.boxes[:, 1] + frame.boxes[:, 3]) / 2 >= y) & ((frame.boxes[:, 1] + frame.boxes[:, 3]) / 2 < y + source_side)
        chosen = frame.boxes[selected].copy()
        chosen[:, [0, 2]] = (chosen[:, [0, 2]] - x) * scale; chosen[:, [1, 3]] = (chosen[:, [1, 3]] - y) * scale
        chosen = np.clip(chosen, 0, tile_size)
        tile = Image.fromarray(canvas[y : y + source_side, x : x + source_side]).resize((tile_size, tile_size), Image.Resampling.BILINEAR)
        return np.asarray(tile), chosen

    def _build_validation_tiles(self) -> list[tuple[SourceFrame, int, int]]:
        tile_size, overlap = self.config["training"]["validation"]["source_tile_size_pixels"], self.config["training"]["validation"]["tile_overlap_fraction"]
        records = []
        for frame in self.frames:
            with Image.open(frame.image_path) as image:
                width, height = image.size
            stride = round(tile_size * (1 - overlap))
            xs = list(range(0, max(1, width - tile_size + 1), stride)) + [max(0, width - tile_size)]
            ys = list(range(0, max(1, height - tile_size + 1), stride)) + [max(0, height - tile_size)]
            for y in sorted(set(ys)):
                for x in sorted(set(xs)):
                    records.append((frame, x, y))
        return records

    def _validation_tile(self, frame: SourceFrame, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
        source_size = self.config["training"]["validation"]["source_tile_size_pixels"]
        with Image.open(frame.image_path) as image:
            image_array = np.asarray(image.convert("RGB"))
        canvas = np.pad(image_array, ((0, max(0, y + source_size - image_array.shape[0])), (0, max(0, x + source_size - image_array.shape[1])), (0, 0)), mode="edge")
        centres = (frame.boxes[:, :2] + frame.boxes[:, 2:]) / 2
        selected = (centres[:, 0] >= x) & (centres[:, 0] < x + source_size) & (centres[:, 1] >= y) & (centres[:, 1] < y + source_size)
        boxes = frame.boxes[selected].copy()
        boxes[:, [0, 2]] -= x; boxes[:, [1, 3]] -= y
        boxes = np.clip(boxes, 0, source_size) * (self.input_size / source_size)
        return canvas[y:y + source_size, x:x + source_size], boxes

    def _photometric(self, image: Image.Image) -> Image.Image:
        augmentation = self.config["training"]["augmentations"]
        image = ImageEnhance.Brightness(image).enhance(1 + self.rng.uniform(-augmentation["brightness_delta"], augmentation["brightness_delta"]))
        image = ImageEnhance.Contrast(image).enhance(1 + self.rng.uniform(-augmentation["contrast_delta"], augmentation["contrast_delta"]))
        image = ImageEnhance.Color(image).enhance(1 + self.rng.uniform(-augmentation["saturation_delta"], augmentation["saturation_delta"]))
        if self.rng.random() < augmentation["gaussian_blur_probability"]:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        if self.rng.random() < augmentation["jpeg_compression_probability"]:
            buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=self.rng.randint(60, 90)); image = Image.open(buffer).convert("RGB")
        return image

    def __getitem__(self, index: int) -> dict:
        if self.mode == "train":
            if self.rng.random() < self.config["training"]["sampler"]["mosaic_probability"]:
                half = self.tile_size // 2
                image = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                mosaic_boxes = []
                for quadrant, (offset_x, offset_y) in enumerate(((0, 0), (half, 0), (0, half), (half, half))):
                    tile, tile_boxes = self._random_train_tile(self.frames[(index + quadrant + self.rng.randrange(len(self.frames))) % len(self.frames)], half)
                    image[offset_y:offset_y + half, offset_x:offset_x + half] = tile
                    tile_boxes[:, [0, 2]] += offset_x; tile_boxes[:, [1, 3]] += offset_y
                    mosaic_boxes.append(tile_boxes)
                boxes = np.concatenate(mosaic_boxes)
            else:
                image, boxes = self._random_train_tile(self.frames[index % len(self.frames)])
        else:
            image, boxes = self._validation_tile(*self.tile_records[index])
        image = Image.fromarray(image).resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        if self.mode == "train":
            image = self._photometric(image)
            array = np.asarray(image)
            aug = self.config["training"]["augmentations"]
            if self.rng.random() < aug["random_horizontal_flip_probability"]:
                array = np.ascontiguousarray(array[:, ::-1]); boxes[:, [0, 2]] = self.input_size - boxes[:, [2, 0]]
            if self.rng.random() < aug["random_vertical_flip_probability"]:
                array = np.ascontiguousarray(array[::-1, :]); boxes[:, [1, 3]] = self.input_size - boxes[:, [3, 1]]
        else:
            array = np.asarray(image)
        normalized = xyxy_to_normalized_xywh(boxes, self.input_size)
        return {"img": torch.from_numpy(np.ascontiguousarray(array[:, :, ::-1]).transpose(2, 0, 1)), "cls": torch.zeros((len(boxes), 1)), "bboxes": normalized, "batch_idx": torch.zeros((len(boxes),)), "im_file": "tile", "ori_shape": (self.input_size, self.input_size), "resized_shape": (self.input_size, self.input_size), "ratio_pad": ((1.0, 1.0), (0, 0))}
