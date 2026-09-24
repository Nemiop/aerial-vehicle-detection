"""Shared MLflow setup and artifact logging for vehicle-detector experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import mlflow
import yaml
from mlflow.tracking import MlflowClient


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]


def resolve_project_path(configured_path: str) -> Path:
    """Resolve a project-relative path declared in an experiment YAML file."""
    candidate = Path(configured_path)
    return candidate if candidate.is_absolute() else PROJECT_DIRECTORY / candidate


def read_experiment_config(config_path: Path) -> dict[str, Any]:
    """Read one experiment configuration YAML file."""
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def get_tracking_uri(database_path: Path) -> str:
    """Return the absolute SQLite tracking URI required by MLflow."""
    return f"sqlite:///{database_path.resolve().as_posix()}"


def configure_tracking(config: dict[str, Any]) -> tuple[str, Path]:
    """Configure a local SQLite MLflow store and return its URI and artifact root."""
    database_path = resolve_project_path(config["mlflow"]["database_path"])
    artifact_root = resolve_project_path(config["mlflow"]["artifact_root"])
    database_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = get_tracking_uri(database_path)
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri, artifact_root


def ensure_experiment(config: dict[str, Any], artifact_root: Path) -> str:
    """Create or reuse the named local MLflow experiment."""
    experiment_name = config["experiment"]["name"]
    client = MlflowClient()
    existing_experiment = client.get_experiment_by_name(experiment_name)
    if existing_experiment is not None:
        return existing_experiment.experiment_id
    return client.create_experiment(
        name=experiment_name,
        artifact_location=artifact_root.resolve().as_uri(),
        tags={"project": "aerial-vehicle-detection", "storage": "local"},
    )


def log_experiment_inputs(config: dict[str, Any], config_path: Path) -> None:
    """Store all immutable inputs needed to reproduce a future training run."""
    dataset = config["dataset"]
    model = config["model"]
    mlflow.log_artifact(str(config_path), "configuration")
    for configured_path, artifact_subdirectory in (
        (dataset["data_yaml"], "dataset"),
        (dataset["manifest"], "dataset"),
        (dataset["metadata"], "dataset"),
        (model["architecture_yaml"], "model_definition"),
        (model["source_checkpoint"], "model_initialization"),
        (model["initialized_state_dict"], "model_initialization"),
        (model["transfer_report"], "model_initialization"),
    ):
        source_path = resolve_project_path(configured_path)
        if source_path.is_file():
            mlflow.log_artifact(str(source_path), artifact_subdirectory)


def log_checkpoint_files(checkpoint_paths: Iterable[Path], artifact_subdirectory: str = "checkpoints") -> None:
    """Log final and resumable weight files as retrievable MLflow artifacts."""
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path.is_file():
            mlflow.log_artifact(str(checkpoint_path), artifact_subdirectory)


def write_run_reference(output_path: Path, payload: dict[str, Any]) -> None:
    """Write a compact, human-readable reference to an MLflow run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
