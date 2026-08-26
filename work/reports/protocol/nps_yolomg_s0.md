# Protocol sweep -- YOLOMG NPS seed 0

One fixed set of detections, re-scored under each protocol choice. Nothing about the model changes between rows; only the counting rules do.

| configuration | AP |
|---|---|
| **our evaluator** (frames=all, ap_style=all-point, matcher=iou, conf=0.001, agg=pooled) | **0.4965** |
| **their published protocol** (frames=annotated, ap_style=101pt, matcher=iou, conf=0.001, agg=pooled) | **0.5047** |

Total gap attributable to protocol: **+0.0082**

## One axis at a time, moving from our protocol toward theirs

| axis | our value | their value | AP | delta |
|---|---|---|---|---|
| frames | all | annotated | 0.5043 | +0.0078 |
| ap_style | all-point | 101pt | 0.4969 | +0.0003 |

## Full grid

| frames | ap_style | matcher | conf | agg | AP |
|---|---|---|---|---|---|
| all | all-point | iou | 0.001 | pooled | 0.4965 |
| all | all-point | iou | 0.001 | per-video | 0.5913 |
| all | all-point | iou | 0.25 | pooled | 0.4646 |
| all | all-point | iou | 0.25 | per-video | 0.5511 |
| all | all-point | centre | 0.001 | pooled | 0.5038 |
| all | all-point | centre | 0.001 | per-video | 0.5992 |
| all | all-point | centre | 0.25 | pooled | 0.4700 |
| all | all-point | centre | 0.25 | per-video | 0.5565 |
| all | 101pt | iou | 0.001 | pooled | 0.4969 |
| all | 101pt | iou | 0.001 | per-video | 0.5912 |
| all | 101pt | iou | 0.25 | pooled | 0.4957 |
| all | 101pt | iou | 0.25 | per-video | 0.5895 |
| all | 101pt | centre | 0.001 | pooled | 0.5041 |
| all | 101pt | centre | 0.001 | per-video | 0.5989 |
| all | 101pt | centre | 0.25 | pooled | 0.4997 |
| all | 101pt | centre | 0.25 | per-video | 0.5939 |
| all | 11pt | iou | 0.001 | pooled | 0.5108 |
| all | 11pt | iou | 0.001 | per-video | 0.5704 |
| all | 11pt | iou | 0.25 | pooled | 0.4742 |
| all | 11pt | iou | 0.25 | per-video | 0.5329 |
| all | 11pt | centre | 0.001 | pooled | 0.5143 |
| all | 11pt | centre | 0.001 | per-video | 0.5755 |
| all | 11pt | centre | 0.25 | pooled | 0.4759 |
| all | 11pt | centre | 0.25 | per-video | 0.5390 |
| annotated | all-point | iou | 0.001 | pooled | 0.5043 |
| annotated | all-point | iou | 0.001 | per-video | 0.6234 |
| annotated | all-point | iou | 0.25 | pooled | 0.4687 |
| annotated | all-point | iou | 0.25 | per-video | 0.5724 |
| annotated | all-point | centre | 0.001 | pooled | 0.5119 |
| annotated | all-point | centre | 0.001 | per-video | 0.6323 |
| annotated | all-point | centre | 0.25 | pooled | 0.4742 |
| annotated | all-point | centre | 0.25 | per-video | 0.5783 |
| annotated | 101pt | iou | 0.001 | pooled | 0.5047 |
| annotated | 101pt | iou | 0.001 | per-video | 0.6232 |
| annotated | 101pt | iou | 0.25 | pooled | 0.5013 |
| annotated | 101pt | iou | 0.25 | per-video | 0.6157 |
| annotated | 101pt | centre | 0.001 | pooled | 0.5122 |
| annotated | 101pt | centre | 0.001 | per-video | 0.6319 |
| annotated | 101pt | centre | 0.25 | pooled | 0.5053 |
| annotated | 101pt | centre | 0.25 | per-video | 0.6204 |
| annotated | 11pt | iou | 0.001 | pooled | 0.5173 |
| annotated | 11pt | iou | 0.001 | per-video | 0.5982 |
| annotated | 11pt | iou | 0.25 | pooled | 0.4774 |
| annotated | 11pt | iou | 0.25 | per-video | 0.5515 |
| annotated | 11pt | centre | 0.001 | pooled | 0.5205 |
| annotated | 11pt | centre | 0.001 | per-video | 0.6040 |
| annotated | 11pt | centre | 0.25 | pooled | 0.4791 |
| annotated | 11pt | centre | 0.25 | per-video | 0.5575 |
