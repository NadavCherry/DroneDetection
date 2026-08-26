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

## Is the difference real? Paired, seed-matched, over sequences

Paired bootstrap **and** permutation over the 15 shared sequences; a bin is called significant only when both agree, matching `tools/make_summary.py`. Seeds are matched pairwise.

| bin | seed | d AP | 95% CI | p perm | verdict |
|---|---|---|---|---|---|
| <8 px | 0 | +0.065 | [-0.121, +0.241] | 0.5315 | no difference |
| <8 px | 1 | +0.104 | [-0.062, +0.271] | 0.2837 | no difference |
| <8 px | 2 | +0.079 | [-0.079, +0.242] | 0.4346 | no difference |
| 8-10 px | 0 | +0.076 | [-0.055, +0.154] | 0.4176 | no difference |
| 8-10 px | 1 | +0.114 | [-0.015, +0.199] | 0.3616 | no difference |
| 8-10 px | 2 | +0.095 | [-0.035, +0.182] | 0.5674 | no difference |
| 10-16 px | 0 | -0.081 | [-0.143, -0.023] | 0.0140 | **significant** |
| 10-16 px | 1 | -0.041 | [-0.092, +0.017] | 0.1499 | no difference |
| 10-16 px | 2 | -0.045 | [-0.090, -0.017] | 0.0839 | no difference |
| 16-25 px | 0 | -0.185 | [-0.322, -0.099] | 0.0010 | **significant** |
| 16-25 px | 1 | -0.121 | [-0.258, -0.057] | 0.0010 | **significant** |
| 16-25 px | 2 | -0.171 | [-0.312, -0.088] | 0.0010 | **significant** |
| >25 px | 0 | -0.206 | [-0.422, -0.081] | 0.0010 | **significant** |
| >25 px | 1 | -0.126 | [-0.347, -0.012] | 0.0320 | **significant** |
| >25 px | 2 | -0.169 | [-0.439, -0.044] | 0.0020 | **significant** |

**No bin where ours wins significantly on every seed.**

