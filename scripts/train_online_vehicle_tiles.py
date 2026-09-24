"""Run experiment 001 with online random tiles and MLflow tracking."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import torch
from PIL import Image, ImageDraw

from online_vehicle_tiles import OnlineVehicleTileDataset, PROJECT_DIRECTORY, read_config
from mlflow_vehicle_experiment import configure_tracking, ensure_experiment, log_checkpoint_files, log_experiment_inputs


CONFIG_PATH = PROJECT_DIRECTORY / "configs" / "experiment_001_yolo26n_vehicle.yaml"


def save_preflight_previews(dataset: OnlineVehicleTileDataset, output_directory: Path) -> None:
    """Check transformed labels and write auditable samples before the first optimizer step."""
    output_directory.mkdir(parents=True, exist_ok=True)
    for index in range(min(100, len(dataset))):
        sample = dataset[index]
        boxes = sample["bboxes"].numpy()
        if len(boxes) == 0 or not torch.isfinite(sample["bboxes"]).all() or not ((sample["bboxes"] >= 0) & (sample["bboxes"] <= 1)).all():
            raise RuntimeError(f"Invalid online tile labels at sample {index}")
        if index < 25:
            image = sample["img"].permute(1, 2, 0).numpy()[:, :, ::-1]
            preview = Image.fromarray(image)
            draw = ImageDraw.Draw(preview)
            for cx, cy, width, height in boxes:
                draw.rectangle(((cx-width/2)*960, (cy-height/2)*960, (cx+width/2)*960, (cy+height/2)*960), outline="#19c37d", width=3)
            preview.save(output_directory / f"online_tile_{index:03d}.jpg", quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output directory name. Defaults to the declared MLflow run name.",
    )
    arguments = parser.parse_args()
    config = read_config(arguments.config)
    if config["training"]["status"] != "pending_implementation_and_ground_truth_preview_approval":
        raise RuntimeError("Unexpected experiment state")
    os.environ["YOLO_CONFIG_DIR"] = str(PROJECT_DIRECTORY / "My Artifacts" / "ultralytics")
    from ultralytics import YOLO
    from ultralytics.data.build import build_dataloader
    from ultralytics.models.yolo.detect import DetectionTrainer

    class OnlineTileTrainer(DetectionTrainer):
        experiment_config = config
        def build_dataset(self, img_path, mode="train", batch=None):
            return OnlineVehicleTileDataset(self.experiment_config, mode=mode)
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
            return build_dataloader(dataset, batch=batch_size, workers=self.args.workers if mode == "train" else self.args.workers, shuffle=mode == "train", rank=rank)
        def plot_training_labels(self):
            """Online tiles have no fixed label distribution to plot before an epoch."""

    preflight_dataset = OnlineVehicleTileDataset(config, "train")
    preflight_directory = (
        PROJECT_DIRECTORY
        / "My Artifacts"
        / "dataset_quality"
        / f"{config['experiment']['run_name']}_online_tile_preflight"
    )
    save_preflight_previews(preflight_dataset, preflight_directory)

    tracking_uri, artifact_root = configure_tracking(config)
    experiment_id = ensure_experiment(config, artifact_root)
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_name=config["experiment"]["run_name"] + "_training") as run:
        mlflow.set_tags({"run.kind": "training", "augmentation.mode": "online_per_sample", "validation.unit": "deterministic_tiles"})
        mlflow.log_params({"scale_range": "0.8-2.0", "minimum_size_quantile": 0.05, "tiles_per_source_per_epoch": 4})
        log_experiment_inputs(config, arguments.config)
        mlflow.log_artifacts(str(preflight_directory), "geometry_preflight")
        model = YOLO(str(PROJECT_DIRECTORY / config["model"]["architecture_yaml"]), task="detect")
        state = torch.load(PROJECT_DIRECTORY / config["model"]["initialized_state_dict"], map_location="cpu", weights_only=True)
        model.model.load_state_dict(state, strict=True)

        def log_epoch_checkpoint(trainer) -> None:
            """Upload an epoch snapshot before a long run can be interrupted."""
            checkpoint_directory = Path(trainer.save_dir) / "weights"
            for checkpoint_name in ("last.pt", "best.pt"):
                checkpoint = checkpoint_directory / checkpoint_name
                if checkpoint.is_file():
                    mlflow.log_artifact(
                        str(checkpoint),
                        f"live_checkpoints/epoch_{trainer.epoch + 1:03d}",
                    )

        model.add_callback("on_model_save", log_epoch_checkpoint)
        result = model.train(
            trainer=OnlineTileTrainer, data=str(PROJECT_DIRECTORY / config["dataset"]["data_yaml"]), epochs=arguments.epochs or config["training"]["epochs"],
            patience=config["training"]["early_stopping_patience_epochs"], imgsz=960, batch=config["training"]["batch_size"], workers=config["training"]["workers"],
            optimizer=config["training"]["optimizer"], lr0=config["training"]["initial_learning_rate"], lrf=config["training"]["final_learning_rate_fraction"], warmup_epochs=config["training"]["warmup_epochs"],
            seed=config["training"]["random_seed"], deterministic=config["training"]["deterministic"], amp=config["training"]["amp"], mosaic=0.0, fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            project=str(PROJECT_DIRECTORY / "My Artifacts" / "student_model" / "experiments"),
            name=arguments.output_name or config["experiment"]["run_name"],
            # Never mix checkpoints from runs that happen to share a name.
            exist_ok=False,
            device=0,
            save_period=1,
        )
        save_directory = Path(model.trainer.save_dir)
        log_checkpoint_files([save_directory / "weights" / "best.pt", save_directory / "weights" / "last.pt"])
        mlflow.log_artifacts(str(save_directory), "ultralytics_run")
        final_metrics = result.results_dict if hasattr(result, "results_dict") else result
        mlflow.log_dict({key: float(value) for key, value in final_metrics.items()}, "final_metrics.json")
        print(json.dumps({"run_id": run.info.run_id, "save_directory": str(save_directory)}, indent=2))


if __name__ == "__main__":
    main()
