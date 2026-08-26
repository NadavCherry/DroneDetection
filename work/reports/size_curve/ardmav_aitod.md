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

## Is the difference real? Paired, seed-matched, over sequences

Paired bootstrap **and** permutation over the 15 shared sequences; a bin is called significant only when both agree, matching `tools/make_summary.py`. Seeds are matched pairwise.

| bin | seed | d AP | 95% CI | p perm | verdict |
|---|---|---|---|---|---|
| very-tiny | 0 | +0.065 | [-0.121, +0.241] | 0.5315 | no difference |
| very-tiny | 1 | +0.104 | [-0.062, +0.271] | 0.2837 | no difference |
| very-tiny | 2 | +0.079 | [-0.079, +0.242] | 0.4346 | no difference |
| tiny | 0 | -0.028 | [-0.089, +0.047] | 0.4545 | no difference |
| tiny | 1 | +0.000 | [-0.059, +0.064] | 0.9970 | no difference |
| tiny | 2 | -0.008 | [-0.050, +0.045] | 0.8352 | no difference |
| small | 0 | -0.125 | [-0.228, -0.059] | 0.0010 | **significant** |
| small | 1 | -0.070 | [-0.168, -0.017] | 0.0140 | **significant** |
| small | 2 | -0.108 | [-0.234, -0.038] | 0.0020 | **significant** |
| medium | 0 | -0.467 | [-0.674, -0.256] | 0.0010 | **significant** |
| medium | 1 | -0.371 | [-0.627, -0.189] | 0.0020 | **significant** |
| medium | 2 | -0.420 | [-0.644, -0.220] | 0.0020 | **significant** |

**No bin where ours wins significantly on every seed.**

