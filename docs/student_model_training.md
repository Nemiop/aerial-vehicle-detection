# `vehicle` Compact Detector Training Plan

## Limitation and decision made
> **Update 09.23.2026.** The P3/P4 model with 492,794 parameters remains the control baseline. The selected candidate for the next training is `configs/yolo26n_vehicle_p2p3p4_492k.yaml`: **492,535 parameters**, direct outputs `Detect(P2/4, P3/8, P4/16)`, 309,600 accurately transferred backbone weights (62.86%). The new neck has 48/48/56 channels on P2/P3/P4. Initialization and tiled smoke test A–D have already been completed; the model was not trained.

The final model must contain strictly less than 500,000 parameters, including frozen ones. FP16, INT8, zeroing weights and normal unstructured pruning do not reduce the number of parameters and are therefore not used to satisfy this limitation.

The control P3/P4 baseline is constructed by reducing the pretrained **YOLO26n** with the transfer of compatible weights. YOLO26x remains a standalone candidate for the role of teacher for distillation. P3/P4 baseline architecture:

- one final class `vehicle`;
- regular one-to-many Detect with NMS, `end2end: false`;
- two detection levels: P3/stride 8 and P4/stride 16;
- backbone ends at P4; branch P5/stride 32 removed;
- SPPF on P4 is preserved for spatial context;
- `depth=0.5`, `width=0.25`, and P3/P4-backbone with the same tensor shapes as YOLO26n;
- the new SPPF on P4 and the new P4 neck block each have 80 channels instead of 128;
- exact size: **492,794 parameters**;
- margin up to the limit of 500,000: **7,205 parameters**.

Number 492794 verified locally via `sum(parameter.numel() for parameter in model.parameters())` in Ultralytics 8.4.127. Before training, you also need forward, loss and backward smoke tests. The option with `width=0.225` and 433,604 parameters is a reserve if the actual implementation does not meet the limit or the gain in speed justifies the loss of capacity.

The finished one-class YOLO26n contains 2,504,190 parameters and does not pass the limitation. The reduction changes the network graph: some of its weights can be copied directly, some - only after a coordinated selection of channels, new connections will have to be initialized again. This is not equivalent to disabling unnecessary layers without additional training.

## Architectures considered

| Architecture | Number of parameters | Passes `<500k` | Solution |
|---|---:|:---:|---|
| YOLO26n, one class | 2,504,190 locally | No | Source of pretrained weights |
| YOLOX-Nano | 0.91 million | No | Requires almost twofold structural reduction |
| NanoDet-Plus-m | 1.17 million | No | Compact but still well over the limit |
| NanoDet-m-0.5x | 0.28 million | Yes | Additional ready-made baseline; officially 13.5 COCO AP at 320 px |
| YOLO-FastestV2 | 0.25 million | Yes | Additional speed baseline; low resolution and outdated architecture are worse for long-range vehicles |
| **YOLO26n → P3/P4, compact new neck** | **492 794** | **Yes** | Control baseline without direct P2 output |
| YOLO26n → P3/P4, width 0.225 | 433 604 | Yes | Reserve additional narrowing |

The main option retains the P3/P4-backbone YOLO26n without narrowing the channels. The budget is freed up by removing the P5 branch and reducing only the new SPPF/neck blocks, which still cannot receive weights directly. Ready-made NanoDet-m-0.5x and YOLO-FastestV2 remain optional baseline, if time permits.

## Progressive reduction in architecture

The table compares different architectures after each stage. The numbers of parameters were obtained from the previous local construction of the model; After saving the final YAML, they need to be rechecked. Old numerical predictions of mAP changes have been deleted: they have not been confirmed before the experiment and cannot be added.

| Stage | Options after stage | Removed at stage | Main risk |
|---|---:|---:|---|
| Standard YOLO26n, 80 classes | 2 572 280 | — | Starting point |
| One `vehicle` | 2 504 190 | 68,090 | The separation of transport types disappears |
| Without additional end-to-end head | 2 383 407 | 120 783 | NMS required |
| Without P5/32, C2PSA and P5 neck/head branches; leave P3/P4 | 574 634 | 1 808 773 | Less context for very large objects |
| Reduce only new SPPF and P4 neck block from 128 to 80 channels | **492 794** | **81,840** | Less capacity of new feature-fusion blocks; **main option** |
| Additionally `width=0.225` | 433 604 | 59 190 | Less capacity and fewer compatible scales; reserve |

The original `yolo26.yaml` builds the backbone sequentially up to P5, applies SPPF and C2PSA, then neck connects P5, P4 and P3, and Detect produces boxes at three scales. `P3/8`, `P4/16`, `P5/32` denote the step of the feature map in input pixels. With a 640x640 input, the cards have approximately 80x80, 40x40, and 20x20 cells. The proposed circuit ends the backbone at P4, puts SPPF on P4 and leaves the connection P4↔P3 with Detect(P3, P4).

**Why exactly these stages are acceptable:**

1. **One class.** The classification output of the head is reduced, but the ability of the backbone to find car boundaries is not removed. A single-class output is different in form from an 80-class output and cannot be completely copied from a checkpoint.
2. **Without additional end-to-end path.** With Ultralytics installed, `end2end=True` creates additional `one2one_cv2` and `one2one_cv3` in parallel with the usual `cv2` and `cv3`. We keep the usual one-to-many Detect and use NMS. This allows you to remove the duplicate predictive path, since NMS-free inference is not required by the job.
3. **Without P5 and C2PSA.** P5 is the roughest map and wide deep blocks. For small long-range vehicles, P3 is more important; P4 is saved for larger ones and context. Together with P5, its downsample blocks, C2PSA, the reverse P5 neck branch and the third Detect output are removed. This changes the routes of signs and can worsen large trucks - check the quality separately by box size, rather than declaring removal free.
4. **SPPF on P4.** It leaves the extended spatial context without P5. This is no longer the same SPPF after P5: due to the changed input and number of channels, its weights are copied only if compatibility is proven, otherwise the block is initialized again.
5. **Narrow only new blocks.** P4-backbone is saved on 128 channels, so its tensors coincide with YOLO26n and are copied entirely. SPPF was moved from P5 to P4, and the P4 neck branch was reassembled, that is, these blocks already do not have semantically compatible weights. Their outputs have been reduced from 128 to 80 channels - this frees up the parameter budget without cutting off the surviving backbone weights.

Removing only the last P5 head is not enough: the backbone and neck that calculate it still exist. It is also impossible to simply cut out the P5-backbone without reconfiguring the inputs of the subsequent `Concat`, SPPF and Detect - the graph will become incorrect. The table describes the **architecture rebuild**. Without P5, large machines may suffer; When you reduce an entire 4K frame, distant ones may disappear. Therefore, tiles and estimates based on object sizes are required.

### Transferring weights from YOLO26n

1. Load the source `yolo26n.pt` and create a one-class student P3/P4. Match blocks by role and relationships, rather than by numerical index: indexes change after deleting branches.
2. Layers 0–6, that is, the entire saved P3/P4-backbone, have the same shape and input semantics: convolutions and BatchNorm are copied without changes. Channel cutting is not used in the basic version.
3. Re-initialize the new neck connections, the transferred SPPF and the one-class output. Copy the compatible part of the regression head if its inputs and shapes actually match. We do not plan to randomly initialize the entire model as the main run.
4. Save a report on the percentage of parameters transferred as a whole, transferred as slices, and reinitialized. Check the content after downloading rather than relying on the message `model.load()`: Ultralytics automatically downloads only weights that match the shape. After such a reduction, additional training is required; Transfer alone does not guarantee quality.

## Possibilities for further optimization

| Method | Parameters student | Why consider |
|---|---:|---|
| Distillation by YOLO26x | 492 794 | An attempt to restore quality after supervised training; winnings must be measured |
| Further structural narrowing to width 0.225 | 433,604 according to the previous calculation | Reserve for exceeding the limit or confirmed need for acceleration |
| Stronger structural thinning, depthwise blocks, SPPF removal | Will require a new count | Not necessary yet: all methods can worsen the context or recall |
| Unstructured pruning | `numel()` does not change | Doesn't help fulfill formal constraint |
| FP16 and INT8 | `numel()` does not change | Check speed, file size and degradation after selecting model |
| ONNX/TensorRT, Conv-BN fusion | Logical architecture does not change | Measure actual latency on target device |
| Entry 960 instead of 640 | 492 794 | Check the recall of distant vehicles; the number of pixels and calculations grows by approximately 2.25 times per tile |
| Tiled inference | 492 794 | Maintains the size of objects relative to the full 4K frame, but increases the time per frame |
| Straight head P2/P3/P4 | **492 535** | Selected candidate: P2 connected to `Detect`, P5 still removed |

We do not pass off numerical forecasts of mAP changes for these methods as measurements. Since the main student should already fit into the limit, we first check the transfer of weights, the selection of tiles and the quality; new thinning is not a goal in itself.

## Tiles and coordinated augmentation

The base source is the full frames A–C and their boxes. During training, take **random areas** of images, rather than one fixed set of mesh crops. The 1280x1280 base area is driven to a 640x640 input; Smaller parts of the original image are padded to an area without distorting the proportions. The area selection should cover different sized cars, random locations, and a moderate amount of empty or complex backgrounds. Sample dynamically at each epoch, fixing the seed and configuration.

User-selected scale mixture for the first series of runs:

| Share | Mode | Entry effect 640×640 |
|---:|---|---|
| 35% | Regular random tile 1280x1280 | Basic scale: objects become half the size of the original |
| 35% | Zoom-in: nested window, object magnification ×1.2…2 relative to the base | At maximum, a 640x640 window is rendered as 640x640; the area of ​​the object relative to the base reaches ×4 |
| 15% | Zoom-out: object scale ×0.75…1 relative to the base | Adds distant zooms if cars remain distinguishable |
| 15% | Mosaic of four areas **different frames A–C** | Changes the environment and usually makes objects smaller; turn off at the end of training |

The modes are mutually exclusive: do not add reduction with mosaic or with built-in strong `scale`. Simply increasing the entire tile to 2560 px before then decreasing it to 640 **does not increase the object at the input of the model**; What is needed is a tighter crop. For zoom-out, check the sizes of the boxes **after** casting to the input and cancel the transformation if the cars become almost indistinguishable; You can't just remove their labels and leave them on the image. When cropping, recalculate and trim boxes along the visible part, rejecting severely damaged examples.

Additionally, horizontal or vertical flip (not both), rotation **no more than ±20°**, moderate shift, brightness/contrast/HSV and low blur, noise or JPEG compression are allowed. 90° rotations, strong perspective/shear, MixUp, CutMix and Copy-Paste are currently disabled: they can change the geometry of the scene, hide small objects or create controversial boxes. Standard Copy-Paste Ultralytics requires segmentation masks, which are not available in this bbox dataset. Before full training, save a set of visual examples of augmentation and check the images along with the boxes.

On D and future `Evaluation`, apply **deterministic** 1280x1280 tiles with 20% overlap, return boxes to full frame coordinates, and remove inter-tile duplicates. Check for gaps in seams; Do not use a hard property-area cutoff without this check. Random sampling and mosaic work only on A–C. Augmentation creates variants of existing scenes, but does not make adjacent frames independent observations.

## Training scheme

1. Use a ready-made one-class dataset `Data/Labels/final_vehicle/20260923_vehicle_yolo_v1`: A–C — train, D — trusted validation ground truth. Fix the path and name of split in `configs/train.yaml`: the general directory and `validation` are currently indicated there, while the finished dataset uses `val`.
2. Use YAML for the main student 492 794 parameters, check the limit, forward/loss/backward and report of the full transfer of P3/P4-backbone from YOLO26n. If the actual number turns out to be ≥500,000, correct the model before training.
3. Train the **pre-trained initialized** student first with regular random tiles, then with a mixture of scales 35/35/15/15. Starting settings for both runs: seed 42, up to 100 epochs, early stopping 20, AdamW, batch 8 (if there is not enough VRAM - 4 with change recording), 2 CPU workers. Compare with the same validation scheme.
4. For the best supervised option, separately check the 960 px input, together with recall, measuring the processing time of a full frame.
5. Only after this, carry out a separate distillation experiment from YOLO26x. The standard Ultralytics implementation counts on three neck levels, while student has P3/P4; requires adaptation and testing rather than simply enabling `distill_model`. Leave distillation only if improvement is confirmed.
6. Evaluate several best checkpoints on **full** D frames with the same tiled inference that the final model will use. Fix the architecture, checkpoint, input, confidence, NMS and inter-tile merge before opening `Evaluation`.
7. After committing, evaluate `Evaluation` once, then export the selected model to ONNX/FP16. INT8 PTQ/QAT check separately if acceleration is required; they do not reduce the number of logical parameters.

Local MLflow with SQLite can store configurations, seeds, environment versions, dataset hashes, migration report, metrics, checkpoints and error examples without a constantly running server.

## Distillation loss

For a single class, passing class logits is less important than geometry and foreground detection. Priority of components in a separate **experiment**, not necessarily in the first run:

1. standard supervised detection loss based on manually verified labels;
2. box distillation for agreed teacher and GT facilities;
3. objectness distillation with masking of background areas;
4. feature distillation at levels P3/P4 near GT boxes;
5. increased weight of small objects corresponding to the far range.

Teacher only works during distillation and is not included in the parameters of the final student. Loss coefficients are selected using the trusted validation markup D, starting with a small weight, so that teacher errors do not overpower the ground truth. The built-in distillation-wrapper Ultralytics binds teacher to student layer indices and is designed for three neck levels; graph P3/P4 requires separate implementation/adaptation and smoke test.

## Minimum experiment matrix

| Run | Architecture | Initialization | Distillation | Resolution | Destination |
|---|---|---|---|---:|---|
| A | YOLO26n → P3/P4, 493k | Transfer from YOLO26n | No | 640 | Control run with random tiles |
| B | Same model | Transfer from YOLO26n | No | 640 | Selected large-scale augmentations 35/35/15/15 |
| C | Best checkpoint A/B | Same | No | 960 | Checking small objects and delays |
| D | Best A-C configuration | Same | YOLO26x, adapted P3/P4 | 640 or 960 | Optional distillation after supervised base |

If time budget is limited, A and B are required; C is carried out for problems with small objects, D - only after a stable supervised result.

## Selection metrics

On D as a trusted validation ground truth the following are stored:

- general `mAP@0.5` and, if possible, `mAP@0.5:0.95`;
- precision and recall for `[0, 200)` m;
- precision and recall for `[200, 400]` m;
- false alarms/min;
- number FN of small objects;
- latency for batch 1 at the target runtime;
- the exact number of all and trainable parameters.

We consider marking D to be ground truth according to the adopted project decision. There are no training examples of motorcycles in A–C, although there are some in D; diagnose this group separately. Check the quality also by the size of the box, especially for the smallest and largest objects. Confidence threshold should not be lowered for the sake of recall without simultaneously monitoring false alarms/min.

The camera parameters are still unknown. The `[0, 200)` and `[200, 400]` m ranges are **estimates** of distance based on box size under explicitly written assumptions about the FOV and physical length of the vehicle, with a sensitivity analysis to these assumptions. A student of the same class does not know whether it is a car, a truck, or a motorcycle, so the distance estimates for different types have a systematic error. Do not pass off these meters as instrumentally measured.

## Automatic limit check

Before starting each train-run, the following check must be performed:

```python
MAX_PARAMETER_COUNT = 499_999

parameter_count = sum(parameter.numel() for parameter in model.parameters())
trainable_parameter_count = sum(
    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
)

if parameter_count > MAX_PARAMETER_COUNT:
    raise ValueError(
        f"Model has {parameter_count:,} parameters; "
        f"limit is {MAX_PARAMETER_COUNT:,}."
    )
```

Both numbers are recorded in the manifest of each experiment. The constraint is checked against `parameter_count`, and not just against trainable parameters.

## Criteria for accepting the final model

The final student is accepted if the following conditions are simultaneously met:

- `parameter_count ≤ 499 999`;
- there are no forward, loss and backward errors;
- the selected checkpoint was obtained without using Evaluation;
- metrics of both ranges and false alarms/min are calculated according to the recorded protocol;
- the exported model reproduces FP32 metrics within a pre-recorded tolerance;
- confidence threshold, NMS and resolution are fixed until Evaluation.
- YOLO26n weight transfer report and visual check of augmentations are saved along with the launch configuration.

## Sources

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26)
- [Ultralytics Model YAML Configuration Guide](https://docs.ultralytics.com/guides/model-yaml-config)
- [Ultralytics Data Augmentation Guide](https://docs.ultralytics.com/guides/yolo-data-augmentation)
- [Ultralytics Knowledge Distillation Guide](https://docs.ultralytics.com/guides/knowledge-distillation)
- [NanoDet model zoo](https://github.com/RangiLyu/nanodet)
- [YOLOX model zoo](https://github.com/Megvii-BaseDetection/YOLOX)
- [YOLO-FastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2)
- [Focal and Global Knowledge Distillation for Detectors](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html)
- [ScaleKD: Distilling Scale-Aware Knowledge in Small Object Detector](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.html)
- [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial)
- [TorchAO quantization example](https://docs.pytorch.org/ao/stable/eager_tutorials/first_quantization_example.html)



