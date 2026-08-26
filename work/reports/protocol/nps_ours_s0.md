# Protocol sweep -- OURS temporal NPS e100 seed 0

One fixed set of detections, re-scored under each protocol choice. Nothing about the model changes between rows; only the counting rules do.

| configuration | AP |
|---|---|
| **our evaluator** (frames=all, ap_style=all-point, matcher=iou, conf=0.001, agg=pooled) | **0.4819** |
| **their published protocol** (frames=annotated, ap_style=101pt, matcher=iou, conf=0.001, agg=pooled) | **0.4938** |

Total gap attributable to protocol: **+0.0118**

## One axis at a time, moving from our protocol toward theirs

| axis | our value | their value | AP | delta |
|---|---|---|---|---|
| frames | all | annotated | 0.4931 | +0.0112 |
| ap_style | all-point | 101pt | 0.4824 | +0.0005 |

## Full grid

| frames | ap_style | matcher | conf | agg | AP |
|---|---|---|---|---|---|
| all | all-point | iou | 0.001 | pooled | 0.4819 |
| all | all-point | iou | 0.001 | per-video | 0.5246 |
| all | all-point | iou | 0.25 | pooled | 0.4692 |
| all | all-point | iou | 0.25 | per-video | 0.5090 |
| all | all-point | centre | 0.001 | pooled | 0.4905 |
| all | all-point | centre | 0.001 | per-video | 0.5311 |
| all | all-point | centre | 0.25 | pooled | 0.4772 |
| all | all-point | centre | 0.25 | per-video | 0.5148 |
| all | 101pt | iou | 0.001 | pooled | 0.4824 |
| all | 101pt | iou | 0.001 | per-video | 0.5258 |
| all | 101pt | iou | 0.25 | pooled | 0.4870 |
| all | 101pt | iou | 0.25 | per-video | 0.5286 |
| all | 101pt | centre | 0.001 | pooled | 0.4910 |
| all | 101pt | centre | 0.001 | per-video | 0.5315 |
| all | 101pt | centre | 0.25 | pooled | 0.4943 |
| all | 101pt | centre | 0.25 | per-video | 0.5334 |
| all | 11pt | iou | 0.001 | pooled | 0.4740 |
| all | 11pt | iou | 0.001 | per-video | 0.5110 |
| all | 11pt | iou | 0.25 | pooled | 0.4526 |
| all | 11pt | iou | 0.25 | per-video | 0.4993 |
| all | 11pt | centre | 0.001 | pooled | 0.4853 |
| all | 11pt | centre | 0.001 | per-video | 0.5295 |
| all | 11pt | centre | 0.25 | pooled | 0.4619 |
| all | 11pt | centre | 0.25 | per-video | 0.5125 |
| annotated | all-point | iou | 0.001 | pooled | 0.4931 |
| annotated | all-point | iou | 0.001 | per-video | 0.5406 |
| annotated | all-point | iou | 0.25 | pooled | 0.4779 |
| annotated | all-point | iou | 0.25 | per-video | 0.5211 |
| annotated | all-point | centre | 0.001 | pooled | 0.5019 |
| annotated | all-point | centre | 0.001 | per-video | 0.5475 |
| annotated | all-point | centre | 0.25 | pooled | 0.4861 |
| annotated | all-point | centre | 0.25 | per-video | 0.5270 |
| annotated | 101pt | iou | 0.001 | pooled | 0.4938 |
| annotated | 101pt | iou | 0.001 | per-video | 0.5422 |
| annotated | 101pt | iou | 0.25 | pooled | 0.4980 |
| annotated | 101pt | iou | 0.25 | per-video | 0.5438 |
| annotated | 101pt | centre | 0.001 | pooled | 0.5025 |
| annotated | 101pt | centre | 0.001 | per-video | 0.5480 |
| annotated | 101pt | centre | 0.25 | pooled | 0.5054 |
| annotated | 101pt | centre | 0.25 | per-video | 0.5485 |
| annotated | 11pt | iou | 0.001 | pooled | 0.4844 |
| annotated | 11pt | iou | 0.001 | per-video | 0.5245 |
| annotated | 11pt | iou | 0.25 | pooled | 0.4594 |
| annotated | 11pt | iou | 0.25 | per-video | 0.5096 |
| annotated | 11pt | centre | 0.001 | pooled | 0.4958 |
| annotated | 11pt | centre | 0.001 | per-video | 0.5451 |
| annotated | 11pt | centre | 0.25 | pooled | 0.4687 |
| annotated | 11pt | centre | 0.25 | per-video | 0.5237 |
