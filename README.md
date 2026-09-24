# Aerial vehicle detection: ML Engineer test task

End-to-end experimentation code for detecting vehicles in Bird View video frames. The repository contains code, configurations, experiment documentation and the final project report; it intentionally excludes datasets, extracted frames, pseudo-label archives, model checkpoints, MLflow databases and rendered media.

## Repository layout

- `configs/` - YAML configurations for teacher inference, pseudo-label merging, COCO pre-training, aerial fine-tuning and evaluation.
- `scripts/` - preparation, annotation, training, inference, MLflow and report-generation utilities.
- `docs/` - annotation policy, pseudo-labelling plan, prediction-tree notes and training workflow.
- `README_ML_ENGINEER_VEHICLE_DETECTOR_REPORT.pdf` - final four-page project report.
- `tools/cvat/` - instructions for the external CVAT dependency.

## Pipeline

1. Inspect and sample Bird View source videos.
2. Produce candidate annotations with YOLO26x and RF-DETR Large; use tiles to recover small objects and merge candidates.
3. Review A-C proposals in CVAT; reserve D as a video-level validation source.
4. Adapt YOLO26n to a one-class, P2/P3/P4, sub-500k-parameter detector.
5. Remap COCO transport classes to `vehicle`, pre-train on RunPod L40S and track experiments in MLflow.
6. Fine-tune on the reviewed Bird View split and run adaptive tiled full-frame inference.

## Environment

Python 3.12 was used for the cloud training environment. Install the project-specific MLflow package with:

```powershell
pip install -r requirements-mlflow.txt
```

The training and inference scripts expect the data directories described in their YAML files. Create local `Data/` and `My Artifacts/` directories, then adapt absolute paths and paths in the configuration files to your environment. These directories are deliberately ignored by Git.

## External dependencies

- Ultralytics / PyTorch with CUDA for YOLO training and inference.
- RF-DETR for the second pseudo-labelling teacher.
- CVAT, cloned separately into `tools/cvat/` when manual review is required.
- MLflow for local experiment tracking; see `docs/mlflow_experiment_workflow.md`.

## Reproducibility note

The report describes a local RTX 3060 attempt followed by the practical RunPod L40S execution, with persistent storage and MLflow tracking. Exact raw data, checkpoints and runs are intentionally not published in this repository.
