# Prediction tree: combining two teacher models

`prediction_tree` is a review-oriented combination of ready-made proposals YOLO26x and RF-DETR Large. It is not considered ground truth and is not used for training until manual verification.

The current configuration is in `configs/merge_predictions_v3.yaml`, and the result is in `Data/Labels/proposals/prediction_tree/20260920_tiled_v3/`. Previous runs are kept unchanged for comparison and reproducibility.

## Algorithm

1. Predictions are grouped by `frame_key`. Comparison between different frames is not possible.
2. For all YOLO–RF-DETR pairs, the IoU matrix is ​​constructed.
3. The Hungarian algorithm finds a one-to-one bipartite matching with the maximum total IoU. One bbox of one model cannot be automatically combined with several bboxes of another model at once.
4. Pairs with IoU < 0.50 are not considered one object.
5. For a pair with IoU ≥ 0.85, the boundaries practically coincide. The larger area bbox is retained as defined for the project.
6. For a pair with `0.50 ≤ IoU < 0.85`, pairwise equal-weight box fusion is applied: each coordinate of the final `xyxy` is equal to the average of the coordinates of two teachers.
7. Unmapped bboxes of both models are saved with `teacher_count: 1` and `review_reason: single_teacher_only`. They cannot be deleted automatically: among them there are both false positives and real objects found by only one model.

## Fragments of long trucks

After pairwise pooling, a separate geometry layer is applied to class `truck` only. It eliminates the case where one long truck is represented by a common bbox and several cab, body or parts bboxes coming from different tiles.

1. Truck candidates are sorted by area, the largest becomes the anchor of the group.
2. The anchor includes nested candidates and longitudinally aligned fragments with sufficient lateral overlap. Each candidate is compared to an anchor; There is no transitive union through a chain of neighboring objects.
3. Before merging, an exception is checked for two large neighboring trucks. If the areas are comparable, the horizontal overlap is large, the centers are sufficiently spaced, and each bbox protrudes noticeably on its side along the X axis, both bboxes are preserved. Y coordinates are not used in this test.
4. Thanks to X-only checking, separate cab and body located on the same width can be combined even if there is a gap or strong overlap in frame height.
5. If the anchor covers at least 70% of the bounding union of the group, the anchor with `merge_status: truck_cluster_anchor_kept` remains.
6. If the coverage is less than 70%, the anchor remains, but the entry receives `merge_status: geometry_conflict`, `geometry_status: fragmented_truck` and `review_reason: fragmented_truck`.
7. In all groups, `component_predictions`, `component_count`, a list of teacher models, anchor coverage and `recommended_box_xyxy` are saved. Therefore, the absorbed original predictions are not lost and the solution can be double-checked.

For `geometry_conflict`, the worker bbox deliberately does not automatically expand to union: neighboring trucks in heavy traffic can also be longitudinally aligned. `recommended_box_xyxy` serves as a hint to the marker, and not as a ready-made truth.

Equal-weight fusion was chosen intentionally. Confidence YOLO and RF-DETR are obtained with different models and at different thresholds, so without calibration they are not a common scale. To use them directly as weights would be to attribute unproven reliability to one teacher. Both confidences are saved separately.

## Class conflict

Geometric matching is performed independently of the class to detect when models see the same object but call it differently.

For such a pair:

- the final `class_name` is equal to `class_conflict`;
- `has_class_conflict` is equal to `true`;
- `teacher_classes` contains separate classes YOLO26x and RF-DETR Large;
- `teacher_predictions` saves the original bbox, confidence, source class ID and tile origin of both models;
- on the overlay, the conflict is drawn with a dark gray dotted line and signed `Unknown Type`.

`class_conflict` is a verification utility class. Before exporting train labels, a person must replace it with `car`, `truck` or `motorcycle` or remove the false bbox.

## Why not a regular NMS

A regular NMS chooses the bbox with the highest confidence. Confidence of two different architectures is not calibrated here, and NMS does not show class disagreement. Hungarian matching maintains a one-to-one relationship, after which the boundary rule and the class rule are applied independently and remain verifiable.

## Manual check priority

1. `geometry_conflict`;
2. `class_conflict`, displayed as `Unknown Type`;
3. `single_teacher_only`;
4. matched pairs with moderate IoU;
5. almost identical pairs of the same class.

## Captions and seven visual states

| Class | Teacher support | Signature | Decoration |
|---|---|---|---|
| car | YOLO26x | `Car_YOLO` | pale green solid |
| car | RF-DETR Large | `Car_RF-DETR` | medium green solid |
| car | both models | `Car` | deep green solid |
| truck | YOLO26x | `Truck_YOLO` | pale red solid |
| truck | RF-DETR Large | `Truck_RF-DETR` | medium red solid |
| truck | both models | `Truck` | deep red solid |
| type conflict | both models | `Unknown Type` | dark gray dotted |

`motorcycle` is saved in JSON as a separate source class, but in the review overlay it uses the green group `Car` to keep the palette of exactly seven states. For a truck cluster, teacher support is determined by all `component_predictions`, and not just by anchor.

All thresholds, input files, signatures and v3 colors are in `configs/merge_predictions_v3.yaml`. The result stores hashes of both input JSON, manifest and snapshot configuration.

## Possible next improvements

- **Temporal check on adjacent frames.** Link truck-boxes with a tracker and do not combine candidates that continue as two independent tracks. This is the strongest next step for video.
- **Linking to lanes.** Estimate the position of the lower center point bbox relative to the road lanes. Objects in adjacent stripes are saved separately even with large overlap.
- **Oriented boxes.** Compare the longitudinal and transverse axis of the rotated bbox, and not the axes of the entire frame. It is more stable on road bends.
- **Mask evidence.** For controversial large objects, use segmentation teacher: two unrelated masks mean two trucks, a single mask supports the unification of the cab and body.
- **Checking size stability.** When merging, reject the union if its width or aspect ratio goes beyond the distribution of truck-boxes of the given video and perspective zone.
