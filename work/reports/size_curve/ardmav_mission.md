# Accuracy vs target size -- ardmav

Bins on sqrt(area) in pixels (mission). Evaluator: rule=iou, IoU=0.5, conf>=0.001, pooled over 15 sequences. Every arm is scored by this one evaluator; the arms differ only in which detections they read.

`n` is GT instances in the bin. `+-` is the sample standard deviation over training seeds. `CI` is a 95 % bootstrap interval over SEQUENCES.

| bin | n | ours | yolomg | ours-single |
|---|---|---|---|---|
| <8 px | 5677 | 0.503 ± 0.011 | 0.420 ± 0.017 | 0.378 ± 0.031 |
| 8-10 px | 5055 | 0.602 ± 0.007 | 0.507 ± 0.015 | 0.493 ± 0.029 |
| 10-16 px | 7529 | 0.732 ± 0.013 | 0.787 ± 0.010 | 0.728 ± 0.033 |
| 16-25 px | 4731 | 0.729 ± 0.029 | 0.888 ± 0.006 | 0.758 ± 0.039 |
| >25 px | 5168 | 0.739 ± 0.023 | 0.905 ± 0.018 | 0.771 ± 0.054 |

`*` = fewer than 50 GT instances; treat as underpowered.

## Crossover: ours minus yolomg

| bin | n | delta | leader |
|---|---|---|---|
| <8 px | 5677 | +0.083 | ours |
| 8-10 px | 5055 | +0.095 | ours |
| 10-16 px | 7529 | -0.056 | yolomg |
| 16-25 px | 4731 | -0.159 | yolomg |
| >25 px | 5168 | -0.167 | yolomg |

**Crossover between `8-10 px` and `10-16 px`.** ours leads on the smaller side (+0.095) and trails on the larger (-0.056).

