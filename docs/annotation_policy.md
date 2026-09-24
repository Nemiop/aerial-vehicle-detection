# Vehicle annotation policy

Status: pilot policy v0.2. Apply the same rules to train and validation. Revised from the train pilot on 2026-09-20.

## Labels

| Label | Include | Boundary cases |
|---|---|---|
| `car` | Passenger cars, SUVs, minivans and passenger-oriented pickups | A cargo van or visibly commercial pickup is `truck` |
| `motorcycle` | Motorcycles, scooters and mopeds, including the rider as one object | Bicycles are excluded |
| `truck` | Box trucks, cargo vans, lorries, tractors, articulated trucks and visibly commercial cargo pickups | A tractor and attached trailer form one vehicle box |

The scenes are treated as containing no buses. A teacher `bus` proposal is preserved as its source class in metadata and converted to `truck` in the working proposal.

Include parked and moving vehicles. Exclude bicycles, trains, construction machines that are not road vehicles, vehicle-shaped signs, reflections and shadows.

## Boxes

- Draw one tight axis-aligned box around the visible pixels of each vehicle. Do not include its cast shadow.
- Clip boxes at the image boundary. Mark `truncated=true` when the image boundary cuts the vehicle.
- Mark `occluded=true` when another object hides a meaningful part of the vehicle.
- A tractor with an attached trailer is one `truck`; disconnected trailers are not labelled as vehicles.
- Label every object that is recognisable as a vehicle at high zoom. Do not impose a pixel-size cutoff merely to simplify the task.
- If a speck cannot be distinguished from road texture even after checking adjacent train-video frames, do not label it as a vehicle.

## Class uncertainty and distance eligibility

- If the object is certainly a vehicle but its subtype cannot be resolved, retain the most plausible working class and set `subtype_uncertain=true`. These cases must be reviewed explicitly before distance evaluation.
- Set `distance_eligible=false` for strongly occluded, strongly truncated or subtype-uncertain objects. The detection label remains valid; the object is excluded only from size-based distance assignment under the documented evaluation rule.
- Set `distance_eligible=true` only when the visible box and subtype provide a defensible size proxy.
- Preserve these attributes in the reviewed three-class dataset. The one-class student export may merge class names, but must not destroy the source metadata.

## Review procedure

For every frame, inspect the full image at fit-to-screen and then the road regions at high zoom. Confirm missed objects, duplicate boxes, class, boundary, truncation, occlusion and distance eligibility. A frame with no boxes still needs an explicit reviewed status.

## Automated proposal cleanup

- Run tiled inference only: 1280×1280 source-image tiles with 20% overlap.
- If predictions with different classes have IoU greater than 0.85, retain the smaller box and assign `car`.
- If two `car` proposals have IoU greater than 0.85, retain the larger box.
- If one `truck` proposal substantially contains another, with intersection divided by the smaller area greater than 0.80, retain the larger box.
- Convert every teacher `bus` proposal to `truck`.
- Remove every `truck` proposal from source video B, where this class is absent by scene-level knowledge.
- After these domain rules, apply class-aware tile NMS at IoU 0.60.
- Preserve raw predictions separately so every automatic removal and class conversion remains auditable.
