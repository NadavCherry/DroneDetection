# Accuracy vs target size -- local_ft

Bins on sqrt(area) in pixels (aitod). Evaluator: rule=iou, IoU=0.5, conf>=0.001, pooled over 1 sequences. Every arm is scored by this one evaluator; the arms differ only in which detections they read.

`n` is GT instances in the bin. `+-` is the sample standard deviation over training seeds. `CI` is a 95 % bootstrap interval over SEQUENCES.

| bin | n | ours | yolomg | ours-single |
|---|---|---|---|---|
| very-tiny | 123 | 0.430 ± 0.088 | 0.319 ± 0.153 | 0.032 ± 0.056 |
| tiny | 127 | 0.789 ± 0.079 | 0.554 ± 0.090 | 0.398 ± 0.071 |

`*` = fewer than 50 GT instances; treat as underpowered.

## Crossover: ours minus yolomg

| bin | n | delta | leader |
|---|---|---|---|
| very-tiny | 123 | +0.110 | ours |
| tiny | 127 | +0.236 | ours |

**No crossover on local_ft.** ours leads in every bin with enough GT to judge (`very-tiny`, `tiny`), so there is no size regime that separates the methods here.

`local_ft` contains **no ground truth at all** in `small`, `medium`. Nothing about those sizes can be concluded from this dataset.

