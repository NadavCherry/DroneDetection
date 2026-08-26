# Accuracy vs target size -- local_ft

Bins on sqrt(area) in pixels (mission). Evaluator: rule=iou, IoU=0.5, conf>=0.001, pooled over 1 sequences. Every arm is scored by this one evaluator; the arms differ only in which detections they read.

`n` is GT instances in the bin. `+-` is the sample standard deviation over training seeds. `CI` is a 95 % bootstrap interval over SEQUENCES.

| bin | n | ours | yolomg | ours-single |
|---|---|---|---|---|
| <8 px | 123 | 0.430 ± 0.088 | 0.319 ± 0.153 | 0.032 ± 0.056 |
| 8-10 px | 81 | 0.767 ± 0.060 | 0.536 ± 0.079 | 0.434 ± 0.053 |
| 10-16 px * | 46 | 0.707 ± 0.150 | 0.317 ± 0.172 | 0.280 ± 0.121 |

`*` = fewer than 50 GT instances; treat as underpowered.

## Crossover: ours minus yolomg

| bin | n | delta | leader |
|---|---|---|---|
| <8 px | 123 | +0.110 | ours |
| 8-10 px | 81 | +0.231 | ours |
| 10-16 px * | 46 | +0.390 | ours |

Bins excluded from the crossover test: `10-16 px` — fewer than 50 GT instances, or a delta below 0.001.

**No crossover on local_ft.** ours leads in every bin with enough GT to judge (`<8 px`, `8-10 px`), so there is no size regime that separates the methods here.

`local_ft` contains **no ground truth at all** in `16-25 px`, `>25 px`. Nothing about those sizes can be concluded from this dataset.

