"""Capture the software, hardware, and checkpoint provenance for the pilot."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import psutil
import torch


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "My Artifacts" / "environment"
YOLO_CHECKPOINT = PROJECT_DIRECTORY / "yolo26x.pt"
RFDETR_CHECKPOINT = Path.home() / ".roboflow" / "models" / "rf-detr-large-2026.pth"
CHECKPOINT_SOURCES = {
    "yolo26x": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt",
    "rfdetr_large": "https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth",
}


def calculate_sha256(file_path: Path) -> str:
    """Calculate a checkpoint digest using bounded memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_checkpoint(name: str, file_path: Path) -> dict:
    """Describe one immutable teacher checkpoint."""
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return {
        "name": name,
        "path": str(file_path.resolve()),
        "size_bytes": file_path.stat().st_size,
        "sha256": calculate_sha256(file_path),
        "source": CHECKPOINT_SOURCES[name],
    }


def main() -> None:
    """Write a compact JSON snapshot and a complete dependency lock."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the pilot environment")
    memory = psutil.virtual_memory()
    snapshot = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_total_bytes": memory.total,
        "memory_available_bytes_at_capture": memory.available,
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "ultralytics": importlib.metadata.version("ultralytics"),
        "rfdetr": importlib.metadata.version("rfdetr"),
        "checkpoints": [
            describe_checkpoint("yolo26x", YOLO_CHECKPOINT),
            describe_checkpoint("rfdetr_large", RFDETR_CHECKPOINT),
        ],
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "pilot_environment.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (OUTPUT_DIRECTORY / "pilot_requirements.txt").write_text(
        freeze, encoding="utf-8"
    )
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
