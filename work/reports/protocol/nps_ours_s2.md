# Protocol sweep -- OURS temporal NPS e100 seed 2

One fixed set of detections, re-scored under each protocol choice. Nothing about the model changes between rows; only the counting rules do.

| configuration | AP |
|---|---|
| **our evaluator** (frames=all, ap_style=all-point, matcher=iou, conf=0.001, agg=pooled) | **0.4347** |
| **their published protocol** (frames=annotated, ap_style=101pt, matcher=iou, conf=0.001, agg=pooled) | **0.4492** |

Total gap attributable to protocol: **+0.0145**

## One axis at a time, moving from our protocol toward theirs

| axis | our value | their value | AP | delta |
|---|---|---|---|---|
| frames | all | annotated | 0.4472 | +0.0125 |
| ap_style | all-point | 101pt | 0.4365 | +0.0018 |

## Full grid

| frames | ap_style | matcher | conf | agg | AP |
|---|---|---|---|---|---|
| all | all-point | iou | 0.001 | pooled | 0.4347 |
| all | all-point | iou | 0.001 | per-video | 0.4772 |
| all | all-point | iou | 0.25 | pooled | 0.4223 |
| all | all-point | iou | 0.25 | per-video | 0.4632 |
| all | all-point | centre | 0.001 | pooled | 0.4438 |
| all | all-point | centre | 0.001 | per-video | 0.4869 |
| all | all-point | centre | 0.25 | pooled | 0.4308 |
| all | all-point | centre | 0.25 | per-video | 0.4722 |
| all | 101pt | iou | 0.001 | pooled | 0.4365 |
| all | 101pt | iou | 0.001 | per-video | 0.4786 |
| all | 101pt | iou | 0.25 | pooled | 0.4412 |
| all | 101pt | iou | 0.25 | per-video | 0.4829 |
| all | 101pt | centre | 0.001 | pooled | 0.4454 |
| all | 101pt | centre | 0.001 | per-video | 0.4872 |
| all | 101pt | centre | 0.25 | pooled | 0.4484 |
| all | 101pt | centre | 0.25 | per-video | 0.4900 |
| all | 11pt | iou | 0.001 | pooled | 0.4438 |
| all | 11pt | iou | 0.001 | per-video | 0.4645 |
| all | 11pt | iou | 0.25 | pooled | 0.4191 |
| all | 11pt | iou | 0.25 | per-video | 0.4534 |
| all | 11pt | centre | 0.001 | pooled | 0.4539 |
| all | 11pt | centre | 0.001 | per-video | 0.4769 |
| all | 11pt | centre | 0.25 | pooled | 0.4255 |
| all | 11pt | centre | 0.25 | per-video | 0.4632 |
| annotated | all-point | iou | 0.001 | pooled | 0.4472 |
| annotated | all-point | iou | 0.001 | per-video | 0.5023 |
| annotated | all-point | iou | 0.25 | pooled | 0.4325 |
| annotated | all-point | iou | 0.25 | per-video | 0.4848 |
| annotated | all-point | centre | 0.001 | pooled | 0.4566 |
| annotated | all-point | centre | 0.001 | per-video | 0.5128 |
| annotated | all-point | centre | 0.25 | pooled | 0.4412 |
| annotated | all-point | centre | 0.25 | per-video | 0.4945 |
| annotated | 101pt | iou | 0.001 | pooled | 0.4492 |
| annotated | 101pt | iou | 0.001 | per-video | 0.5041 |
| annotated | 101pt | iou | 0.25 | pooled | 0.4534 |
| annotated | 101pt | iou | 0.25 | per-video | 0.5075 |
| annotated | 101pt | centre | 0.001 | pooled | 0.4583 |
| annotated | 101pt | centre | 0.001 | per-video | 0.5133 |
| annotated | 101pt | centre | 0.25 | pooled | 0.4606 |
| annotated | 101pt | centre | 0.25 | per-video | 0.5151 |
| annotated | 11pt | iou | 0.001 | pooled | 0.4554 |
| annotated | 11pt | iou | 0.001 | per-video | 0.4863 |
| annotated | 11pt | iou | 0.25 | pooled | 0.4273 |
| annotated | 11pt | iou | 0.25 | per-video | 0.4720 |
| annotated | 11pt | centre | 0.001 | pooled | 0.4653 |
| annotated | 11pt | centre | 0.001 | per-video | 0.5008 |
| annotated | 11pt | centre | 0.25 | pooled | 0.4337 |
| annotated | 11pt | centre | 0.25 | per-video | 0.4837 |
