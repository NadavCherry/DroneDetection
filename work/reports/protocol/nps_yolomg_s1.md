# Protocol sweep -- YOLOMG NPS seed 1

One fixed set of detections, re-scored under each protocol choice. Nothing about the model changes between rows; only the counting rules do.

| configuration | AP |
|---|---|
| **our evaluator** (frames=all, ap_style=all-point, matcher=iou, conf=0.001, agg=pooled) | **0.5351** |
| **their published protocol** (frames=annotated, ap_style=101pt, matcher=iou, conf=0.001, agg=pooled) | **0.5401** |

Total gap attributable to protocol: **+0.0050**

## One axis at a time, moving from our protocol toward theirs

| axis | our value | their value | AP | delta |
|---|---|---|---|---|
| frames | all | annotated | 0.5399 | +0.0049 |
| ap_style | all-point | 101pt | 0.5352 | +0.0002 |

## Full grid

| frames | ap_style | matcher | conf | agg | AP |
|---|---|---|---|---|---|
| all | all-point | iou | 0.001 | pooled | 0.5351 |
| all | all-point | iou | 0.001 | per-video | 0.6254 |
| all | all-point | iou | 0.25 | pooled | 0.5139 |
| all | all-point | iou | 0.25 | per-video | 0.5995 |
| all | all-point | centre | 0.001 | pooled | 0.5403 |
| all | all-point | centre | 0.001 | per-video | 0.6329 |
| all | all-point | centre | 0.25 | pooled | 0.5183 |
| all | all-point | centre | 0.25 | per-video | 0.6059 |
| all | 101pt | iou | 0.001 | pooled | 0.5352 |
| all | 101pt | iou | 0.001 | per-video | 0.6252 |
| all | 101pt | iou | 0.25 | pooled | 0.5381 |
| all | 101pt | iou | 0.25 | per-video | 0.6294 |
| all | 101pt | centre | 0.001 | pooled | 0.5404 |
| all | 101pt | centre | 0.001 | per-video | 0.6325 |
| all | 101pt | centre | 0.25 | pooled | 0.5413 |
| all | 101pt | centre | 0.25 | per-video | 0.6345 |
| all | 11pt | iou | 0.001 | pooled | 0.5229 |
| all | 11pt | iou | 0.001 | per-video | 0.5981 |
| all | 11pt | iou | 0.25 | pooled | 0.4828 |
| all | 11pt | iou | 0.25 | per-video | 0.5768 |
| all | 11pt | centre | 0.001 | pooled | 0.5256 |
| all | 11pt | centre | 0.001 | per-video | 0.6035 |
| all | 11pt | centre | 0.25 | pooled | 0.4839 |
| all | 11pt | centre | 0.25 | per-video | 0.5794 |
| annotated | all-point | iou | 0.001 | pooled | 0.5399 |
| annotated | all-point | iou | 0.001 | per-video | 0.6379 |
| annotated | all-point | iou | 0.25 | pooled | 0.5161 |
| annotated | all-point | iou | 0.25 | per-video | 0.6074 |
| annotated | all-point | centre | 0.001 | pooled | 0.5453 |
| annotated | all-point | centre | 0.001 | per-video | 0.6459 |
| annotated | all-point | centre | 0.25 | pooled | 0.5205 |
| annotated | all-point | centre | 0.25 | per-video | 0.6140 |
| annotated | 101pt | iou | 0.001 | pooled | 0.5401 |
| annotated | 101pt | iou | 0.001 | per-video | 0.6378 |
| annotated | 101pt | iou | 0.25 | pooled | 0.5414 |
| annotated | 101pt | iou | 0.25 | per-video | 0.6393 |
| annotated | 101pt | centre | 0.001 | pooled | 0.5454 |
| annotated | 101pt | centre | 0.001 | per-video | 0.6453 |
| annotated | 101pt | centre | 0.25 | pooled | 0.5447 |
| annotated | 101pt | centre | 0.25 | per-video | 0.6444 |
| annotated | 11pt | iou | 0.001 | pooled | 0.5264 |
| annotated | 11pt | iou | 0.001 | per-video | 0.6081 |
| annotated | 11pt | iou | 0.25 | pooled | 0.4841 |
| annotated | 11pt | iou | 0.25 | per-video | 0.5831 |
| annotated | 11pt | centre | 0.001 | pooled | 0.5288 |
| annotated | 11pt | centre | 0.001 | per-video | 0.6159 |
| annotated | 11pt | centre | 0.25 | pooled | 0.4852 |
| annotated | 11pt | centre | 0.25 | per-video | 0.5857 |
