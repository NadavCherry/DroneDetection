# Protocol sweep -- OURS temporal NPS e100 seed 1

One fixed set of detections, re-scored under each protocol choice. Nothing about the model changes between rows; only the counting rules do.

| configuration | AP |
|---|---|
| **our evaluator** (frames=all, ap_style=all-point, matcher=iou, conf=0.001, agg=pooled) | **0.5441** |
| **their published protocol** (frames=annotated, ap_style=101pt, matcher=iou, conf=0.001, agg=pooled) | **0.5542** |

Total gap attributable to protocol: **+0.0101**

## One axis at a time, moving from our protocol toward theirs

| axis | our value | their value | AP | delta |
|---|---|---|---|---|
| frames | all | annotated | 0.5525 | +0.0083 |
| ap_style | all-point | 101pt | 0.5457 | +0.0016 |

## Full grid

| frames | ap_style | matcher | conf | agg | AP |
|---|---|---|---|---|---|
| all | all-point | iou | 0.001 | pooled | 0.5441 |
| all | all-point | iou | 0.001 | per-video | 0.5613 |
| all | all-point | iou | 0.25 | pooled | 0.5321 |
| all | all-point | iou | 0.25 | per-video | 0.5464 |
| all | all-point | centre | 0.001 | pooled | 0.5507 |
| all | all-point | centre | 0.001 | per-video | 0.5691 |
| all | all-point | centre | 0.25 | pooled | 0.5383 |
| all | all-point | centre | 0.25 | per-video | 0.5536 |
| all | 101pt | iou | 0.001 | pooled | 0.5457 |
| all | 101pt | iou | 0.001 | per-video | 0.5621 |
| all | 101pt | iou | 0.25 | pooled | 0.5503 |
| all | 101pt | iou | 0.25 | per-video | 0.5661 |
| all | 101pt | centre | 0.001 | pooled | 0.5522 |
| all | 101pt | centre | 0.001 | per-video | 0.5689 |
| all | 101pt | centre | 0.25 | pooled | 0.5554 |
| all | 101pt | centre | 0.25 | per-video | 0.5717 |
| all | 11pt | iou | 0.001 | pooled | 0.5513 |
| all | 11pt | iou | 0.001 | per-video | 0.5459 |
| all | 11pt | iou | 0.25 | pooled | 0.5263 |
| all | 11pt | iou | 0.25 | per-video | 0.5361 |
| all | 11pt | centre | 0.001 | pooled | 0.5570 |
| all | 11pt | centre | 0.001 | per-video | 0.5604 |
| all | 11pt | centre | 0.25 | pooled | 0.5288 |
| all | 11pt | centre | 0.25 | per-video | 0.5481 |
| annotated | all-point | iou | 0.001 | pooled | 0.5525 |
| annotated | all-point | iou | 0.001 | per-video | 0.5798 |
| annotated | all-point | iou | 0.25 | pooled | 0.5380 |
| annotated | all-point | iou | 0.25 | per-video | 0.5610 |
| annotated | all-point | centre | 0.001 | pooled | 0.5592 |
| annotated | all-point | centre | 0.001 | per-video | 0.5878 |
| annotated | all-point | centre | 0.25 | pooled | 0.5443 |
| annotated | all-point | centre | 0.25 | per-video | 0.5682 |
| annotated | 101pt | iou | 0.001 | pooled | 0.5542 |
| annotated | 101pt | iou | 0.001 | per-video | 0.5808 |
| annotated | 101pt | iou | 0.25 | pooled | 0.5582 |
| annotated | 101pt | iou | 0.25 | per-video | 0.5839 |
| annotated | 101pt | centre | 0.001 | pooled | 0.5609 |
| annotated | 101pt | centre | 0.001 | per-video | 0.5877 |
| annotated | 101pt | centre | 0.25 | pooled | 0.5633 |
| annotated | 101pt | centre | 0.25 | per-video | 0.5894 |
| annotated | 11pt | iou | 0.001 | pooled | 0.5591 |
| annotated | 11pt | iou | 0.001 | per-video | 0.5615 |
| annotated | 11pt | iou | 0.25 | pooled | 0.5304 |
| annotated | 11pt | iou | 0.25 | per-video | 0.5490 |
| annotated | 11pt | centre | 0.001 | pooled | 0.5646 |
| annotated | 11pt | centre | 0.001 | per-video | 0.5779 |
| annotated | 11pt | centre | 0.25 | pooled | 0.5328 |
| annotated | 11pt | centre | 0.25 | per-video | 0.5624 |
