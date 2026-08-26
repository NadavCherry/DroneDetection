# Accuracy vs target size -- nps

Bins on sqrt(area) in pixels (aitod). Evaluator: rule=iou, IoU=0.5, conf>=0.001, pooled over 10 sequences. Every arm is scored by this one evaluator; the arms differ only in which detections they read.

`n` is GT instances in the bin. `+-` is the sample standard deviation over training seeds. `CI` is a 95 % bootstrap interval over SEQUENCES.

| bin | n | ours | yolomg | ours-single |
|---|---|---|---|---|
| tiny | 6700 | 0.394 ± 0.052 | 0.415 ± 0.011 | 0.407 ± 0.008 |
| small | 2424 | 0.266 ± 0.055 | 0.353 ± 0.061 | 0.308 ± 0.060 |

`*` = fewer than 50 GT instances; treat as underpowered.

## Crossover: ours minus yolomg

| bin | n | delta | leader |
|---|---|---|---|
| tiny | 6700 | -0.021 | yolomg |
| small | 2424 | -0.087 | yolomg |

**No crossover on nps.** yolomg leads in every bin with enough GT to judge (`tiny`, `small`), so there is no size regime that separates the methods here.

`nps` contains **no ground truth at all** in `very-tiny`, `medium`. Nothing about those sizes can be concluded from this dataset.

## Is the difference real? Paired, seed-matched, over sequences

Paired bootstrap **and** permutation over the 10 shared sequences; a bin is called significant only when both agree, matching `tools/make_summary.py`. Seeds are matched pairwise.

| bin | seed | d AP | 95% CI | p perm | verdict |
|---|---|---|---|---|---|
| tiny | 0 | -0.018 | [-0.148, +0.052] | 0.7013 | no difference |
| tiny | 1 | +0.027 | [-0.127, +0.122] | 0.6723 | no difference |
| tiny | 2 | -0.072 | [-0.174, -0.020] | 0.0440 | **significant** |
| small | 0 | -0.026 | [-0.229, +0.106] | 0.7463 | no difference |
| small | 1 | -0.030 | [-0.121, +0.064] | 0.4935 | no difference |
| small | 2 | -0.205 | [-0.302, -0.077] | 0.0020 | **significant** |

**No bin where ours wins significantly on every seed.**

