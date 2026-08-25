# 07_05 — one held-out flight

Protocol `specklock-centre` (centre distance, tau=12 px), 548 labelled instances over 548 frames, resampled as **19 blocks of 30 frames**.

> **This interval is WITHIN ONE SEQUENCE.** It measures whether a difference holds across the segments of this single flight. It is *not* evidence that the difference generalises to another flight — two videos cannot support that claim, and resampling one of them harder does not change what was measured.

| method | AP | Δ vs baseline | 95% CI on Δ | p | verdict |
|---|---|---|---|---|---|
| **singleframe** (baseline) | 0.039 | — | — | — | — |
| temporal | 0.836 | +0.797 | [+0.676, +0.894] | 0.0000 | **better** |
| yolomg | 0.000 | -0.038 | [-0.092, -0.008] | 0.0000 | **DID NOT CONVERGE — not a fair comparison** |

> ⚠ **yolomg scored below 0.05 AP and is treated as NOT CONVERGED.** A near-zero AP is what a model that never trained looks like, not what a method that lost looks like. Do not quote a margin over it as a win: check its own training curve first, and if it also failed on its own validation set then this corpus says something about trainability, not about the method.


### False alarms on labelled distractors

| method | bird | bird#2 | bird#3 | bird#4 | bird#5 | bird#6 | bird#7 | bird#8 | near | total |
|---|---|---|---|---|---|---|---|---|---|---|
| singleframe | 18 | 21 | 19 | 21 | 21 | 11 | 4 | 3 | 759 | 877 |
| temporal | 88 | 75 | 67 | 52 | 46 | 25 | 19 | 2 | 268 | 642 |
| yolomg | 11 | 3 | 2 | 1 | 0 | 0 | 0 | 0 | 7091 | 7108 |
