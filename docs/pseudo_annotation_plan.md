# Pseudo-annotation plan for Terra - draft for discussion

Status as of 2026-09-20: points 1 and 2 completed; the automatic part of point 3 is completed; both teacher models from point 4 are launched at 185 frames in tiled-only mode. All that remains is human verification and calculation of metrics using the corrected markup.

## Context and restrictions

- Source of requirements: `ml_engineer_test_task_v2.pdf`. The general guideline for the task is 8 hours. Priority: reproducibility, proven markup, fair metrics and explanation of solutions.
- Local GPU tested: NVIDIA RTX 3060, 12,288 MiB VRAM. Working environment: Python 3.10.20, PyTorch 2.11.0+cu128, CUDA 12.8; both teacher models passed FP32/FP16 smoke test and pilot.
- The user confirmed desktop RTX 3060 12 GB, Intel i7-6700, 32 GB RAM (currently about 15 GB free), SSD. Do not keep both teacher, all decoded frames and CVAT in memory at the same time. Read the images gradually; select the number of CPU workers in the pilot, do not enable aggressive prefetch by default.
- Ready data: `Data/Frames/frames_manifest.csv`, 185 frames at 2 fps. A: 39, B: 63, C: 34, D: 49. A–C — train (136), D — validation (49).
- `Evaluation` is completely excluded from this stage: do not read frames, do not run predictions, do not select settings based on it.
- Video division is the accepted decision of the project. PDF itself does not require D for validation. This scene is different from train, so its result is characterized by transfer to a specific new scene, rather than a universal quality.
- Teacher: YOLO26x and RF-DETR Large with common pretrained weights. You cannot use ready-made aerial datasets and checkpoints specially trained on them.
- The markup starts with model predictions, then is corrected by a human. The task itself requires estimating the number of added, deleted and corrected boxes.
- Working classes: `car`, `motorcycle`, `truck`. Scenes are considered not to contain buses, so the teacher output `bus` is converted to `truck`, and the original class is saved in metadata. In `vehicle`, classes are converted only in the derived export for student.
- The limit strictly less than 500,000 parameters applies to the future student, not teacher.
- Implementation: simple functions with verbs in their names, small dictionaries, without unnecessary classes and abstractions. Pipeline settings - in YAML; fixed technical constants - at the beginning of the modules.

## 1. Check inputs and environment

**Done.** Result: `My Artifacts/input_audit.json`, `My Artifacts/environment/`, enriched `Data/Frames/frames_manifest.csv` and reproducible smoke/pilot runs.

1. Read the existing manifest, check the uniqueness of the frames, the presence of files, sizes, original frame IDs, timestamps and split. Do not extract or move frames repeatedly.
2. Select inputs only from manifest, do not scan recursively `Data/Raw`: there are unselected frames and Evaluation.
3. Add portable frame ID/relative path and SHA-256 next to existing absolute paths. On another machine, resolve the path through the data root. Do not consider an absolute Windows path as a dataset identifier.
4. Capture versions of Python, CUDA, PyTorch, teacher libraries and specific checkpoints with SHA-256, source and training information. Check the availability of the corresponding release and API; past chat replies are not a verification of the installed version.
5. Check each model on one train frame: loading, classes, coordinates, final confidence, VRAM consumption. Batch size 1, models sequentially and preferably in separate processes.
6. Check CUDA and FP32 first; Enable FP16 after checking support and the absence of corrupted results. OOM should not silently change configuration mid-launch.

The current `prepare_annotation_frames.py` does not read YAML, writes a different manifest and has the old destination path. Don't change it or run it now; This inconsistency should be noted for a separate consistent reproducibility step.

## 2. Define the markup policy

**Completed.** Committed to `docs/annotation_policy.md`.

Before the full launch, write down short rules with examples:

- What to count `car`, `truck`, `motorcycle`; how to convert teacher-output `bus`; how to treat vans, pickups, tractor-trailers and motorcyclists.
- How to mark partial overlap and cropping by an image border. A bbox of the visible part of the object is proposed, without drawing invisible boundaries; fix this the same for all scenes.
- For heavily cut/overlapped objects, bbox does not provide a reliable physical length. Separately note suitability for distance assessment; exceptions and their number are clearly reflected in the report.
- Don't set an arbitrary minimum size that will automatically throw out distant cars. Send ambiguous small objects for verification under magnification.
- Don't force a person to guess subtype. For confidently visible transport with an unclear subtype, provide an uncertainty flag in additional metadata. The rules for exporting it and estimating the distance must be agreed upon in advance.

## 3. Short pilot on train

**The automatic part is done.** Fixed selection, predictions, overlays, contact sheets and metrics are in `My Artifacts/pseudo_annotation/pilot_v2/`. Human review and the precision/recall and edit statistics that depend on it have not yet been completed.

It is proposed to select 12 frames: four evenly over time from A, B, C. Fix the list before comparing the results.

1. Obtain and store predictions of both models separately on identical frames.
2. Manually check each pilot frame as a whole, starting with pre-layout. Also look for objects missed by both models. D should not be used to select teacher settings.
3. Use the resulting verified markup to set up auto labeling. This is a calibration sample and not an independent test; That’s how quality should be labeled on it.
4. Compare no more than two processing options: full frame and full frame with overlapping tiles. The tile size in original pixels and the network input size are set separately.
5. Initial hypothesis: tiles 1280×1280, overlap 20%; Do not consider it optimal until you look at small cars. If necessary, compare a smaller tile on the same pilot. Full frame helps preserve entire large buses/trucks.
6. Use standard preprocessing of each model and reverse transformation of coordinates into the original frame. Do not impose RF-DETR preprocessing YOLO.
7. Select separate confidence thresholds for models: the same number does not mean the same reliability. The priority is recall with an acceptable number of extra boxes for manual checking.
8. Record precision/recall at IoU 0.5, omissions, subtype errors, number of edits, time per frame and VRAM peak. Look at small objects separately; if there are no examples of a class, show no data rather than a null metric.
9. Based on the pilot, estimate the time of a full run, including decoding, preprocessing, all tiles, merging and recording the results. Do not display the published GPU latency of the network during the processing of one initial 4K frame.

Criterion for selecting the initial pre-layout: fewer missed objects and fewer manual corrections with similar recall. Don't choose by average confidence or just the number of boxes.

## 4. Run both models at 185 frames

**Done for raw and automatically cleared proposals.** Run ID: `20260920_tiled_v1`; the results are in `Data/Labels/proposals/<teacher>/20260920_tiled_v1/`. Human review has not yet been completed.

- After the pilot, fix the settings and apply them equally to train and validation.
- Each teacher creates his own immutable result in `Data/Labels/proposals/<teacher>/<run_id>/`.
- Normalize output: stable frame ID, bbox `xyxy` in original pixels, original width/height, canonical class name, original class name/ID, confidence, teacher/checkpoint, tile information.
- Compare classes according to the table of a specific model/adapter. YOLO and different RF-DETR COCO ID APIs may have different conventions; you cannot silently use one list of numeric IDs for both.
- Save frames with zero predictions.
- Remove duplicates between tiles separately within the results of each teacher. Interclass conflicts on one object should be noted and not automatically transformed into two independent objects.
- Save intermediate results by frames and errors. Restart skips completed frames only if the input, model, and configuration hashes match.
- Completion with errors should not be considered successful marking. Record expected/processed number of frames and resulting status.

## 5. Compare the results and prepare a manual check

Recommended start: choose one basic pre-training for the pilot; use the second to search for candidates and conflicts. Save the raw results of both models.

- Compare boxes of two models one-to-one in geometry; record the subtype difference separately.
- Separate cases: agreement, object only for YOLO, only for RF-DETR, class conflict, significantly different boundaries.
- Do not average the confidence of different models without calibration. Do not use WBF or automatic union as an obvious improvement.
- View priority: empty predictions, conflicting frames, small objects, tile boundaries, dense traffic. All other frames also require viewing.
- Model agreement is an indicator of agreement, not accuracy: errors can be common.

## 6. Check in CVAT and save corrections

Local CVAT with COCO JSON import is offered. At the first stage, GPU services are not needed inside CVAT: both models work as separate local scripts.

1. Check import/export on several pilot frames: names and sizes of images, three working classes, coordinates and stable connection with the source objects.
2. Import the selected base pre-layout. Keep the second one in a separate set/visual report so as not to get duplicates in the working markup.
3. View all 185 frames. Check Validation D with the same care, do not leave it as unverified pseudo-tags.
4. Explicitly mark the verification status of each frame, including frames without objects.
5. Export the three-class verified version to `Data/Labels/reviewed/<version>/`. Re-autolabeling should not overwrite manual work.
6. Compare the basic pre-layout before/after checking: added, removed, geometry changed, class changed; separately by video/split. Use persistent IDs if the editor saves them. If a geometric comparison is used, call the statistics approximate and state the rules.
7. Save the markup policy, CVAT export and connection of the dataset version with teacher run.

## 7. Audit and export for students

Check: all images are taken into account, classes are allowed, bboxes are finite and have a positive area, coordinates are inside the original image, there are no duplicates or split intersections, all frames have a check status.

Construct distributions of subtype, bbox sizes and objects by video. If, for example, motorcycle is present predominantly in D, it is obvious to note the lack of train examples: combining the classes into vehicle does not eliminate the absence of such an appearance in training. Do not change split based on quality results without a new, clearly designated protocol.

Derived export to `Data/Labels/final_vehicle/<version>/`: three worker classes become `vehicle`, bbox normalized to original width/height. Test the reverse conversion to pixels on several frames. Do not delete the original subtype and additional metadata.

At the end, compare both frozen teachers with one proven version of the markup: separately for videos and classes. Designate the results on pilot train frames as used for tuning. The results on D should not be used for further selection of teacher settings; D remains validation of the future student.

## 8. MLflow and scaling

- MLflow: parameters, metrics, artifacts and launch connections. Local SQLite and `My Artifacts`; The GPU is allocated to scripts, not to the tracking server.
- Parent run autolabeling and separate teacher runs. Store the version of the reviewed markup and its audit in a separate run with links to the source, so as not to change the meaning of the old run.
- Log manifest/hash, configuration, checkpoint/hash, environment, runtime/VRAM, number of frames/boxes, errors, COCO JSON and overlay examples. Do not duplicate all original videos for each run.
- Before manual checking, log statistics of predictions and agreement; precision/recall require validated markup.
- Transfer to the server: the same commands and configs, different data root and tracking URI. Fix dependencies; in case of teacher-library conflicts, use separate environments with a common result format.
- Orchestrator, Kubernetes and separate online endpoints are not needed for the first result. Scalability is shown by idempotency of stages, portability of inputs, understandable formats and the ability to restore the launch.

## Significant risks

| Problem | Solution/Check |
|---|---|
| Cars disappear when zoomed out 4K | Tiles; fix crop size and input size separately; visually check the smallest bbox |
| Tiles are cut by large machines | Overlapping, full frame pass, combining takes, viewing boundaries |
| Both models made the same mistake | View the entire frame, not just mismatched bboxes |
| Confused truck/car | Separate subtype check; uniform rules for vans and tractors; source `bus` is saved in metadata and converted to `truck` |
| Class ID or bbox formats are mixed up | Explicit adapters, one canonical format, conversion tests and CVAT round trip |
| Adjacent frames ended up in different splits | Split by video from manifest; checking paths/hashes before each step |
| GPU OOM or FP16 incompatibility | Sequential processes, batch 1, smoke test, explicit new configuration when changing |
| Launch aborted | Results by frames, resume with fingerprint check, separate list of errors |
| Null predictions are taken as the absence of objects | Human frame viewing and separate review status |
| Marking after CVAT has lost provenance | Verified import/export and separate metadata for stable IDs |
| A large number of similar bboxes creates the illusion of dataset size | Report on videos and scenes; do not consider adjacent frames as independent observations |

## Distances: a limitation that cannot be hidden

The four subtypes allow you to specify different reference lengths, but do not convert the bbox size into a measured range. What remains is the unknown FOV, perspective, orientation, length variation inside `truck`, partial overlap and diagonal axis-aligned bboxes.

For future estimation of TP/FN ranges, the corrected GT subtype and GT bbox can be used after one-to-one matching. Student returns only `vehicle`: the unmatched FP does not have a GT subtype. Before final evaluation, a separate rule for FP must be selected and explicitly described; You cannot imperceptibly substitute the type from teacher and claim that the distance was given by student. Possible baseline - fixed reference size for FP with sensitivity analysis. This solution is still open.

Range limits: `[0, 200)` and `[200, 400]`, so that 200 m is not counted twice. Don't select FOV/reference dimensions to fill ranges.

The current 2 FPS sampling is for annotation/train/validation. The formula for the `FP × 60 / N_frames` task assumes 1 FPS; For other frequencies, an actual time denominator is needed. Time to first detection will require a separate time protocol with track ID and timestamps. Evaluation is not run by this plan.

## Terra implementation readiness criteria

- The same manifest is processed by both teachers, all 185 frames are taken into account, errors are clearly resolved.
- Raw teacher results, three-class working proposals and a comparison report based on verified data are available.
- All 185 frames are manually checked; Terra does not claim human verification as performed without user action.
- Corrected markup, approximate/exact statistics of edits and dataset version are saved.
- Export student contains one class and passes audit; The original teacher classes and three working classes have been retained.
- Launches are linked via manifest/config/checkpoint hashes; resume is checked and reviewed data is not overwritten.
- Playback commands and restrictions are described briefly; auxiliary logs are separate from the README.

## Solutions for the next conversation

1. How much time should I allocate for manual checking? Full viewing is required; The budget affects the usability of the tool and the volume of comparison, and not the fairness of the reviewed status.
2. Should we adopt a starting approach: one basic pre-design + a second teacher for conflicts, without automatic ensemble?
3. Confirm the policy on controversial subtypes, cut-off objects and tractor-trailers.
4. Select a distance rule for unmatched FPs from the same-class student.

## Primary sources

- Task: `ml_engineer_test_task_v2.pdf` in the project root.
- MLflow Tracking: https://mlflow.org/docs/latest/ml/tracking
- CVAT COCO import/export: https://docs.cvat.ai/docs/manual/advanced/formats/format-coco/
- YOLO26: https://docs.ultralytics.com/models/yolo26/
- RF-DETR inference: https://github.com/roboflow/rf-detr/blob/develop/docs/learn/run/detection.md
- RF-DETR Large API: https://rfdetr.roboflow.com/latest/reference/large/
