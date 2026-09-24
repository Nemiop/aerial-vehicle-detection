"""Pre-train the custom one-class P2/P3/P4 student on filtered COCO vehicles."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from mlflow_vehicle_experiment import configure_tracking, ensure_experiment, log_checkpoint_files, log_experiment_inputs


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIRECTORY / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-train the compact vehicle student on filtered COCO.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIRECTORY / "configs" / "coco_vehicle_pretrain.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-name", default=None)
    arguments = parser.parse_args()
    with arguments.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_yaml = project_path(config["dataset"]["data_yaml"])
    if not data_yaml.is_file():
        raise FileNotFoundError(f"COCO vehicle dataset is absent: run prepare_coco_vehicle_dataset.py first ({data_yaml})")

    from ultralytics import YOLO

    model = YOLO(str(project_path(config["model"]["architecture_yaml"])), task="detect")
    initialized_state = project_path(config["model"]["initialized_state_dict"])
    model.model.load_state_dict(torch.load(initialized_state, map_location="cpu", weights_only=True), strict=True)
    training = config["training"]

    tracking_uri, artifact_root = configure_tracking(config)
    import mlflow

    experiment_id = ensure_experiment(config, artifact_root)
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_name=(arguments.output_name or config["experiment"]["run_name"])):
        mlflow.set_tags({"run.kind": "coco_transport_pretrain", "dataset.classes": "car,motorcycle,bus,truck->vehicle"})
        log_experiment_inputs(config, arguments.config)

        def log_epoch_checkpoint(trainer) -> None:
            """Keep recoverable epoch snapshots in MLflow during long runs."""
            checkpoint_directory = Path(trainer.save_dir) / "weights"
            for checkpoint_name in ("last.pt", "best.pt"):
                checkpoint = checkpoint_directory / checkpoint_name
                if checkpoint.is_file():
                    mlflow.log_artifact(
                        str(checkpoint),
                        f"live_checkpoints/epoch_{trainer.epoch + 1:03d}",
                    )

        model.add_callback("on_model_save", log_epoch_checkpoint)
        model.train(
            data=str(data_yaml),
            epochs=arguments.epochs or training["epochs"],
            patience=training["early_stopping_patience_epochs"],
            imgsz=training["input_size_pixels"],
            batch=training["batch_size"],
            workers=training["workers"],
            optimizer=training["optimizer"],
            lr0=training["initial_learning_rate"],
            lrf=training["final_learning_rate_fraction"],
            warmup_epochs=training["warmup_epochs"],
            seed=training["random_seed"],
            deterministic=training["deterministic"],
            amp=training["amp"],
            project=str(PROJECT_DIRECTORY / "My Artifacts" / "student_model" / "experiments"),
            name=arguments.output_name or config["experiment"]["run_name"],
            # Reusing a directory can make a previous run's best.pt look like
            # the result of the current run. Fail rather than silently mix them.
            exist_ok=False,
            device=0,
            save_period=1,
        )
        save_directory = Path(model.trainer.save_dir)
        log_checkpoint_files([save_directory / "weights" / "best.pt", save_directory / "weights" / "last.pt"])
        mlflow.log_artifacts(str(save_directory), "ultralytics_run")
        print(f"Run directory: {save_directory}", flush=True)


if __name__ == "__main__":
    main()
