# Accuracy vs target size -- nps

Bins on sqrt(area) in pixels (mission). Evaluator: rule=iou, IoU=0.5, conf>=0.001, pooled over 10 sequences. Every arm is scored by this one evaluator; the arms differ only in which detections they read.

`n` is GT instances in the bin. `+-` is the sample standard deviation over training seeds. `CI` is a 95 % bootstrap interval over SEQUENCES.

| bin | n | ours | yolomg | ours-single |
|---|---|---|---|---|
| 10-16 px | 6700 | 0.394 ± 0.052 | 0.415 ± 0.011 | 0.407 ± 0.008 |
| 16-25 px | 2396 | 0.265 ± 0.055 | 0.353 ± 0.061 | 0.306 ± 0.060 |
| >25 px * | 28 | 0.005 ± 0.001 | 0.005 ± 0.002 | 0.007 ± 0.002 |

`*` = fewer than 50 GT instances; treat as underpowered.

## Crossover: ours minus yolomg

| bin | n | delta | leader |
|---|---|---|---|
| 10-16 px | 6700 | -0.021 | yolomg |
| 16-25 px | 2396 | -0.088 | yolomg |
| >25 px * | 28 | +0.000 | tie |

Bins excluded from the crossover test: `>25 px` — fewer than 50 GT instances, or a delta below 0.001.

**No crossover on nps.** yolomg leads in every bin with enough GT to judge (`10-16 px`, `16-25 px`), so there is no size regime that separates the methods here.

`nps` contains **no ground truth at all** in `<8 px`, `8-10 px`. Nothing about those sizes can be concluded from this dataset.

## Is the difference real? Paired, seed-matched, over sequences

Paired bootstrap **and** permutation over the 10 shared sequences; a bin is called significant only when both agree, matching `tools/make_summary.py`. Seeds are matched pairwise.

| bin | seed | d AP | 95% CI | p perm | verdict |
|---|---|---|---|---|---|
| 10-16 px | 0 | -0.018 | [-0.148, +0.052] | 0.7013 | no difference |
| 10-16 px | 1 | +0.027 | [-0.127, +0.122] | 0.6723 | no difference |
| 10-16 px | 2 | -0.072 | [-0.174, -0.020] | 0.0440 | **significant** |
| 16-25 px | 0 | -0.027 | [-0.230, +0.109] | 0.7353 | no difference |
| 16-25 px | 1 | -0.033 | [-0.125, +0.062] | 0.4605 | no difference |
| 16-25 px | 2 | -0.205 | [-0.302, -0.076] | 0.0020 | **significant** |
| >25 px | 0 | +0.001 | [-0.011, +0.004] | 0.8591 | no difference |
| >25 px | 1 | +0.002 | [+0.000, +0.012] | 0.9860 | no difference |
| >25 px | 2 | -0.003 | [-0.027, +0.000] | 0.4236 | no difference |

**No bin where ours wins significantly on every seed.**

