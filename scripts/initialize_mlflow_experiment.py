"""Create the local MLflow experiment and record the first run configuration.

This script does not train a model.  It records the exact inputs and planned
hyperparameters, so the later training run has an auditable parent record.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from mlflow_vehicle_experiment import (
    PROJECT_DIRECTORY,
    configure_tracking,
    ensure_experiment,
    log_experiment_inputs,
    read_experiment_config,
    write_run_reference,
)


DEFAULT_CONFIG_PATH = PROJECT_DIRECTORY / "configs" / "experiment_001_yolo26n_vehicle.yaml"


def parse_arguments() -> argparse.Namespace:
    """Parse the optional experiment configuration path."""
    parser = argparse.ArgumentParser(description="Initialize one local MLflow experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    """Create a pending MLflow run with data and initialization snapshots."""
    arguments = parse_arguments()
    config_path = arguments.config if arguments.config.is_absolute() else PROJECT_DIRECTORY / arguments.config
    config = read_experiment_config(config_path)
    tracking_uri, artifact_root = configure_tracking(config)
    experiment_id = ensure_experiment(config, artifact_root)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run(run_name=config["experiment"]["run_name"]) as active_run:
        mlflow.set_tags(
            {
                "run.state": config["experiment"]["state"],
                "run.kind": "configuration_before_training",
                "model.architecture": "YOLO26n-derived P3/P4 vehicle detector",
                "dataset.split_policy": config["dataset"]["split_policy"],
            }
        )
        mlflow.log_params(
            {
                "model.parameter_count": config["model"]["parameter_count"],
                "model.input_size_pixels": config["model"]["input_size_pixels"],
                "training.epochs": config["training"]["epochs"],
                "training.batch_size": config["training"]["batch_size"],
                "training.optimizer": config["training"]["optimizer"],
                "training.initial_learning_rate": config["training"]["initial_learning_rate"],
                "training.random_seed": config["training"]["random_seed"],
            }
        )
        log_experiment_inputs(config, config_path)
        run_reference = {
            "experiment_name": config["experiment"]["name"],
            "run_name": config["experiment"]["run_name"],
            "run_id": active_run.info.run_id,
            "experiment_id": experiment_id,
            "tracking_uri": tracking_uri,
            "artifact_uri": active_run.info.artifact_uri,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "state": config["experiment"]["state"],
        }
        run_reference_path = artifact_root.parent / "experiment_001_setup_run.json"
        write_run_reference(run_reference_path, run_reference)
        mlflow.log_artifact(str(run_reference_path), "configuration")

    print(f"MLflow experiment: {config['experiment']['name']}")
    print(f"Run ID: {run_reference['run_id']}")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Artifacts: {artifact_root}")
    print(f"Run reference: {run_reference_path}")


if __name__ == "__main__":
    main()
