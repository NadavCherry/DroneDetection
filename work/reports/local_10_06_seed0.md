# 10_06 — one held-out flight

Protocol `specklock-centre` (centre distance, tau=12 px), 250 labelled instances over 361 frames, resampled as **13 blocks of 30 frames**.

> **This interval is WITHIN ONE SEQUENCE.** It measures whether a difference holds across the segments of this single flight. It is *not* evidence that the difference generalises to another flight — two videos cannot support that claim, and resampling one of them harder does not change what was measured.

| method | AP | Δ vs baseline | 95% CI on Δ | p | verdict |
|---|---|---|---|---|---|
| **singleframe** (baseline) | 0.277 | — | — | — | — |
| temporal | 0.550 | +0.273 | [+0.119, +0.456] | 0.0000 | **better** |
| yolomg | 0.010 | -0.267 | [-0.535, -0.017] | 0.0000 | **DID NOT CONVERGE — not a fair comparison** |

> ⚠ **yolomg scored below 0.05 AP and is treated as NOT CONVERGED.** A near-zero AP is what a model that never trained looks like, not what a method that lost looks like. Do not quote a margin over it as a win: check its own training curve first, and if it also failed on its own validation set then this corpus says something about trainability, not about the method.


### False alarms on labelled distractors

_No detection landed on a labelled distractor in any arm. In this GT the birds are `ignore=True`, so a hit on one is recorded here rather than silently counted as a false positive against the background._
