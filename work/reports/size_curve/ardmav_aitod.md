# Accuracy vs target size -- ardmav

Bins on sqrt(area) in pixels (aitod). Evaluator: rule=iou, IoU=0.5, conf>=0.001, pooled over 15 sequences. Every arm is scored by this one evaluator; the arms differ only in which detections they read.

`n` is GT instances in the bin. `+-` is the sample standard deviation over training seeds. `CI` is a 95 % bootstrap interval over SEQUENCES.

| bin | n | ours | yolomg | ours-single |
|---|---|---|---|---|
| very-tiny | 5677 | 0.503 ± 0.011 | 0.420 ± 0.017 | 0.378 ± 0.031 |
| tiny | 12584 | 0.748 ± 0.006 | 0.760 ± 0.009 | 0.698 ± 0.025 |
| small | 7931 | 0.810 ± 0.017 | 0.911 ± 0.011 | 0.836 ± 0.031 |
| medium | 1968 | 0.474 ± 0.036 | 0.893 ± 0.013 | 0.488 ± 0.084 |

`*` = fewer than 50 GT instances; treat as underpowered.

## Crossover: ours minus yolomg

| bin | n | delta | leader |
|---|---|---|---|
| very-tiny | 5677 | +0.083 | ours |
| tiny | 12584 | -0.012 | yolomg |
| small | 7931 | -0.101 | yolomg |
| medium | 1968 | -0.419 | yolomg |

**Crossover between `very-tiny` and `tiny`.** ours leads on the smaller side (+0.083) and trails on the larger (-0.012).

