# dt ablation on full-frame held-out test AP

10 NPS test clips, tiled full-frame inference, unified evaluator, conf 0.001 — the same path every headline number in this repository uses.

| dt | n | test AP | sd | per-seed | vs dt=6 |
|---|---|---|---|---|---|
| 2 | 3 | **0.5061** | 0.0268 | 0.4909 / 0.4903 / 0.5371 | +0.0192 |
| 4 | 3 | **0.4822** | 0.0370 | 0.5046 / 0.5026 / 0.4396 | -0.0047 |
| **6** | 3 | **0.4869** | 0.0549 | 0.4819 / 0.5441 / 0.4347 | — |
| 8 | 3 | **0.4933** | 0.0433 | 0.5002 / 0.5328 / 0.4471 | +0.0064 |
| 12 | 3 | **0.4761** | 0.0335 | 0.5148 / 0.4556 / 0.4579 | -0.0108 |

**Test ranking:** dt2 > dt8 > dt6 > dt4 > dt12

## Paired, seed-matched, over the 10 shared sequences

Positive delta favours dt=6. Significant only when the bootstrap CI excludes zero **and** the permutation p < 0.05, matching `tools/make_summary.py`.

| vs | seed | d AP | 95% CI | p perm | verdict |
|---|---|---|---|---|---|
| dt2 | 0 | -0.0090 | [-0.0617, +0.0383] | 0.8886 | no difference |
| dt2 | 1 | +0.0538 | [+0.0047, +0.0863] | 0.0105 | **significant** |
| dt2 | 2 | -0.1024 | [-0.1358, -0.0577] | 0.0045 | **significant** |
| dt4 | 0 | -0.0226 | [-0.0730, +0.0223] | 0.8246 | no difference |
| dt4 | 1 | +0.0415 | [-0.0385, +0.0974] | 0.2929 | no difference |
| dt4 | 2 | -0.0049 | [-0.0375, +0.0338] | 0.9210 | no difference |
| dt8 | 0 | -0.0183 | [-0.0460, +0.0003] | 0.7746 | no difference |
| dt8 | 1 | +0.0114 | [-0.0480, +0.0593] | 0.7786 | no difference |
| dt8 | 2 | -0.0124 | [-0.0394, +0.0310] | 0.7121 | no difference |
| dt12 | 0 | -0.0329 | [-0.0897, +0.0254] | 0.6482 | no difference |
| dt12 | 1 | +0.0886 | [+0.0086, +0.1329] | 0.0920 | no difference |
| dt12 | 2 | -0.0232 | [-0.0577, +0.0172] | 0.5407 | no difference |

**2 of 12 paired comparisons reached significance.**

