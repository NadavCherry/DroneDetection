# Protocol sweep -- YOLOMG ARD-MAV seed 0

One fixed set of detections, re-scored under each protocol choice. Nothing about the model changes between rows; only the counting rules do.

| configuration | AP |
|---|---|
| **our evaluator** (frames=all, ap_style=all-point, matcher=iou, conf=0.001, agg=pooled) | **0.8425** |
| **their published protocol** (frames=annotated, ap_style=101pt, matcher=iou, conf=0.001, agg=pooled) | **0.8431** |

Total gap attributable to protocol: **+0.0007**

## One axis at a time, moving from our protocol toward theirs

| axis | our value | their value | AP | delta |
|---|---|---|---|---|
| frames | all | annotated | 0.8426 | +0.0002 |
| ap_style | all-point | 101pt | 0.8430 | +0.0005 |

## Full grid

| frames | ap_style | matcher | conf | agg | AP |
|---|---|---|---|---|---|
| all | all-point | iou | 0.001 | pooled | 0.8425 |
| all | all-point | iou | 0.001 | per-video | 0.8234 |
| all | all-point | iou | 0.25 | pooled | 0.8094 |
| all | all-point | iou | 0.25 | per-video | 0.7923 |
| all | all-point | centre | 0.001 | pooled | 0.8836 |
| all | all-point | centre | 0.001 | per-video | 0.8699 |
| all | all-point | centre | 0.25 | pooled | 0.8464 |
| all | all-point | centre | 0.25 | per-video | 0.8349 |
| all | 101pt | iou | 0.001 | pooled | 0.8430 |
| all | 101pt | iou | 0.001 | per-video | 0.8239 |
| all | 101pt | iou | 0.25 | pooled | 0.8618 |
| all | 101pt | iou | 0.25 | per-video | 0.8392 |
| all | 101pt | centre | 0.001 | pooled | 0.8840 |
| all | 101pt | centre | 0.001 | per-video | 0.8699 |
| all | 101pt | centre | 0.25 | pooled | 0.8903 |
| all | 101pt | centre | 0.25 | per-video | 0.8755 |
| all | 11pt | iou | 0.001 | pooled | 0.8100 |
| all | 11pt | iou | 0.001 | per-video | 0.7888 |
| all | 11pt | iou | 0.25 | pooled | 0.7868 |
| all | 11pt | iou | 0.25 | per-video | 0.7644 |
| all | 11pt | centre | 0.001 | pooled | 0.8514 |
| all | 11pt | centre | 0.001 | per-video | 0.8282 |
| all | 11pt | centre | 0.25 | pooled | 0.8001 |
| all | 11pt | centre | 0.25 | per-video | 0.8022 |
| annotated | all-point | iou | 0.001 | pooled | 0.8426 |
| annotated | all-point | iou | 0.001 | per-video | 0.8235 |
| annotated | all-point | iou | 0.25 | pooled | 0.8095 |
| annotated | all-point | iou | 0.25 | per-video | 0.7923 |
| annotated | all-point | centre | 0.001 | pooled | 0.8838 |
| annotated | all-point | centre | 0.001 | per-video | 0.8700 |
| annotated | all-point | centre | 0.25 | pooled | 0.8465 |
| annotated | all-point | centre | 0.25 | per-video | 0.8349 |
| annotated | 101pt | iou | 0.001 | pooled | 0.8431 |
| annotated | 101pt | iou | 0.001 | per-video | 0.8240 |
| annotated | 101pt | iou | 0.25 | pooled | 0.8619 |
| annotated | 101pt | iou | 0.25 | per-video | 0.8392 |
| annotated | 101pt | centre | 0.001 | pooled | 0.8842 |
| annotated | 101pt | centre | 0.001 | per-video | 0.8700 |
| annotated | 101pt | centre | 0.25 | pooled | 0.8905 |
| annotated | 101pt | centre | 0.25 | per-video | 0.8756 |
| annotated | 11pt | iou | 0.001 | pooled | 0.8102 |
| annotated | 11pt | iou | 0.001 | per-video | 0.7889 |
| annotated | 11pt | iou | 0.25 | pooled | 0.7869 |
| annotated | 11pt | iou | 0.25 | per-video | 0.7645 |
| annotated | 11pt | centre | 0.001 | pooled | 0.8516 |
| annotated | 11pt | centre | 0.001 | per-video | 0.8282 |
| annotated | 11pt | centre | 0.25 | pooled | 0.8002 |
| annotated | 11pt | centre | 0.25 | per-video | 0.8022 |
