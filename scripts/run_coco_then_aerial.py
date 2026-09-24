"""Run COCO transport pre-training and then aerial vehicle fine-tuning.

The pipeline never changes either source experiment YAML.  It records a derived
aerial YAML that points at the state dict extracted from the best COCO model,
so the second stage is auditable and can be resumed independently.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_DIRECTORY / candidate


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)


def extract_model_state(checkpoint_path: Path, destination: Path) -> None:
    """Extract the pure state dict expected by train_online_vehicle_tiles.py."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise ValueError(f"Ultralytics checkpoint has no model: {checkpoint_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.float().state_dict(), destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train on COCO, then fine-tune on the aerial vehicle dataset.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIRECTORY / "configs" / "coco_then_aerial.yaml")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_coco_then_aerial"))
    parser.add_argument("--coco-epochs", type=int, default=None)
    parser.add_argument("--aerial-epochs", type=int, default=None)
    arguments = parser.parse_args()
    pipeline_config = load_yaml(arguments.config)
    pipeline = pipeline_config["pipeline"]
    coco_config = project_path(pipeline["coco_config"])
    aerial_source_config = project_path(pipeline["aerial_config"])
    coco_epochs = arguments.coco_epochs or pipeline["coco_epochs"]
    aerial_epochs = arguments.aerial_epochs or pipeline["aerial_epochs"]

    artifact_directory = PROJECT_DIRECTORY / "My Artifacts" / "student_model" / "pipeline_runs" / arguments.run_id
    artifact_directory.mkdir(parents=True, exist_ok=False)
    manifest_path = artifact_directory / "manifest.json"
    manifest = {
        "run_id": arguments.run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "coco_epochs": coco_epochs,
        "aerial_epochs": aerial_epochs,
        "status": "coco_training",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    coco_output_name = f"{arguments.run_id}_coco"
    coco_output_directory = PROJECT_DIRECTORY / "My Artifacts" / "student_model" / "experiments" / coco_output_name
    if coco_output_directory.exists():
        raise FileExistsError(
            f"Refusing to reuse an existing COCO output directory: {coco_output_directory}. "
            "Choose a new --run-id so its checkpoints are unambiguous."
        )
    subprocess.run(
        [sys.executable, str(PROJECT_DIRECTORY / "scripts" / "train_coco_vehicle.py"), "--config", str(coco_config), "--epochs", str(coco_epochs), "--output-name", coco_output_name],
        check=True,
        cwd=PROJECT_DIRECTORY,
    )
    coco_best = coco_output_directory / "weights" / "best.pt"
    if not coco_best.is_file():
        raise FileNotFoundError(f"COCO stage produced no best checkpoint: {coco_best}")
    state_dict_path = artifact_directory / "coco_best_state_dict.pt"
    extract_model_state(coco_best, state_dict_path)

    aerial_config = load_yaml(aerial_source_config)
    aerial_output_name = f"{arguments.run_id}_aerial"
    aerial_output_directory = PROJECT_DIRECTORY / "My Artifacts" / "student_model" / "experiments" / aerial_output_name
    if aerial_output_directory.exists():
        raise FileExistsError(
            f"Refusing to reuse an existing aerial output directory: {aerial_output_directory}. "
            "Choose a new --run-id so its checkpoints are unambiguous."
        )
    aerial_config["experiment"]["run_name"] = aerial_output_name
    aerial_config["experiment"]["description"] = (
        "Aerial fine-tuning from the preceding COCO transport pre-training stage "
        f"({coco_output_name})."
    )
    aerial_config["model"]["initialized_state_dict"] = str(state_dict_path.relative_to(PROJECT_DIRECTORY)).replace("\\", "/")
    aerial_config["training"]["batch_size"] = pipeline["aerial_batch_size"]
    aerial_config["training"]["workers"] = pipeline["aerial_workers"]
    aerial_config["training"]["amp"] = pipeline["aerial_amp"]
    derived_aerial_config = artifact_directory / "aerial_config.yaml"
    save_yaml(derived_aerial_config, aerial_config)
    manifest.update({
        "status": "aerial_training",
        "coco_best_checkpoint": str(coco_best),
        "coco_state_dict": str(state_dict_path),
        "derived_aerial_config": str(derived_aerial_config),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, str(PROJECT_DIRECTORY / "scripts" / "train_online_vehicle_tiles.py"), "--config", str(derived_aerial_config), "--epochs", str(aerial_epochs), "--output-name", aerial_output_name],
        check=True,
        cwd=PROJECT_DIRECTORY,
    )
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
