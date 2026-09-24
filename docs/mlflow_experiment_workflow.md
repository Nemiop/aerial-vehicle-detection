# Local MLflow workflow

MLflow is used here as the experiment ledger, not as the training framework. It
stores the immutable experiment YAML, data manifest, model architecture, transfer
report, metrics, plots, and checkpoint files in `My Artifacts/mlflow`. The
tracking database is local SQLite, and the browser UI is a local service at
`http://127.0.0.1:5000`.

`experiment_001_yolo26n_vehicle.yaml` defines the first run: A/B/C are training
videos, and D is the untouched validation video. There is no frame-level random
split, so adjacent frames cannot leak between the two partitions.

Training samples are generated online. Every time the dataloader requests an
item, it chooses a new vehicle-aware crop and applies the transforms immediately;
the second epoch therefore differs materially from the first even when it begins
from the same source frame. Four synthetic samples are drawn per source image per
epoch. A 16:9 source window of 960×540 pixels is the unit scale: its window is
sampled from 480×270 to 1920×1080, then resized and letterboxed to the 960×960
model input. This makes vehicles appear from 2× larger to 0.8× smaller while
preserving their aspect ratio.

Each box starts as source-image pixel `xyxy` coordinates. Cropping translates and
clips it, resizing and letterboxing use the same affine transform, mosaic adds its
quadrant offset, and flips transform its centre coordinate. Photometric changes
do not change boxes. Boxes with less than 50% visible area are discarded. Before
training, a deterministic geometry suite checks boundary cases, crop/resize/pad,
both flips, and mosaic; for non-clipped boxes the inverse transform must recover
the original box at IoU >= 0.999. It also writes 25 labelled tile previews for
each video. Any failure blocks training.

Validation uses a fixed 1280-pixel tile grid resized to the 960-pixel model
input. It has no augmentation; an object belongs to the one tile that owns its
box centre. Thus a metric is computed over all validation tiles without duplicate
ground-truth boxes.

A **checkpoint** is a saved state of training. `last.pt` is the newest state and
permits resuming a stopped run. `best.pt` is the state from the epoch with the
best validation `mAP50-95`; it is the model used for later inference. At the end
of each run both files will be uploaded to that run's MLflow artifacts, along
with the final metrics and training curves.

Initialize the experiment once:

```powershell
& 'C:/Users/Ruslan/anaconda3/envs/yolo-gpu/python.exe' scripts/initialize_mlflow_experiment.py
```

The tracking dependency is pinned in `requirements-mlflow.txt`.

Start the local UI when needed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_mlflow_ui.ps1
```

The sampling and augmentation module must be implemented before this configuration
is allowed to launch training. This keeps the planned random-tile augmentation
from being accidentally replaced by Ultralytics' default full-frame resize.
