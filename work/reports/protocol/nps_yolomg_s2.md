# Protocol sweep -- YOLOMG NPS seed 2

One fixed set of detections, re-scored under each protocol choice. Nothing about the model changes between rows; only the counting rules do.

| configuration | AP |
|---|---|
| **our evaluator** (frames=all, ap_style=all-point, matcher=iou, conf=0.001, agg=pooled) | **0.5485** |
| **their published protocol** (frames=annotated, ap_style=101pt, matcher=iou, conf=0.001, agg=pooled) | **0.5536** |

Total gap attributable to protocol: **+0.0052**

## One axis at a time, moving from our protocol toward theirs

| axis | our value | their value | AP | delta |
|---|---|---|---|---|
| frames | all | annotated | 0.5532 | +0.0047 |
| ap_style | all-point | 101pt | 0.5489 | +0.0004 |

## Full grid

| frames | ap_style | matcher | conf | agg | AP |
|---|---|---|---|---|---|
| all | all-point | iou | 0.001 | pooled | 0.5485 |
| all | all-point | iou | 0.001 | per-video | 0.6159 |
| all | all-point | iou | 0.25 | pooled | 0.5268 |
| all | all-point | iou | 0.25 | per-video | 0.5885 |
| all | all-point | centre | 0.001 | pooled | 0.5559 |
| all | all-point | centre | 0.001 | per-video | 0.6258 |
| all | all-point | centre | 0.25 | pooled | 0.5334 |
| all | all-point | centre | 0.25 | per-video | 0.5971 |
| all | 101pt | iou | 0.001 | pooled | 0.5489 |
| all | 101pt | iou | 0.001 | per-video | 0.6160 |
| all | 101pt | iou | 0.25 | pooled | 0.5515 |
| all | 101pt | iou | 0.25 | per-video | 0.6181 |
| all | 101pt | centre | 0.001 | pooled | 0.5564 |
| all | 101pt | centre | 0.001 | per-video | 0.6256 |
| all | 101pt | centre | 0.25 | pooled | 0.5566 |
| all | 101pt | centre | 0.25 | per-video | 0.6246 |
| all | 11pt | iou | 0.001 | pooled | 0.5458 |
| all | 11pt | iou | 0.001 | per-video | 0.5884 |
| all | 11pt | iou | 0.25 | pooled | 0.5073 |
| all | 11pt | iou | 0.25 | per-video | 0.5670 |
| all | 11pt | centre | 0.001 | pooled | 0.5494 |
| all | 11pt | centre | 0.001 | per-video | 0.5950 |
| all | 11pt | centre | 0.25 | pooled | 0.5091 |
| all | 11pt | centre | 0.25 | per-video | 0.5700 |
| annotated | all-point | iou | 0.001 | pooled | 0.5532 |
| annotated | all-point | iou | 0.001 | per-video | 0.6305 |
| annotated | all-point | iou | 0.25 | pooled | 0.5293 |
| annotated | all-point | iou | 0.25 | per-video | 0.5979 |
| annotated | all-point | centre | 0.001 | pooled | 0.5608 |
| annotated | all-point | centre | 0.001 | per-video | 0.6411 |
| annotated | all-point | centre | 0.25 | pooled | 0.5359 |
| annotated | all-point | centre | 0.25 | per-video | 0.6074 |
| annotated | 101pt | iou | 0.001 | pooled | 0.5536 |
| annotated | 101pt | iou | 0.001 | per-video | 0.6306 |
| annotated | 101pt | iou | 0.25 | pooled | 0.5551 |
| annotated | 101pt | iou | 0.25 | per-video | 0.6302 |
| annotated | 101pt | centre | 0.001 | pooled | 0.5613 |
| annotated | 101pt | centre | 0.001 | per-video | 0.6409 |
| annotated | 101pt | centre | 0.25 | pooled | 0.5602 |
| annotated | 101pt | centre | 0.25 | per-video | 0.6372 |
| annotated | 11pt | iou | 0.001 | pooled | 0.5495 |
| annotated | 11pt | iou | 0.001 | per-video | 0.6004 |
| annotated | 11pt | iou | 0.25 | pooled | 0.5088 |
| annotated | 11pt | iou | 0.25 | per-video | 0.5756 |
| annotated | 11pt | centre | 0.001 | pooled | 0.5529 |
| annotated | 11pt | centre | 0.001 | per-video | 0.6077 |
| annotated | 11pt | centre | 0.25 | pooled | 0.5106 |
| annotated | 11pt | centre | 0.25 | per-video | 0.5787 |
