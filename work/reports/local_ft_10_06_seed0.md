# 10_06 — one held-out flight

Protocol `specklock-centre` (centre distance, tau=12 px), 250 labelled instances over 250 frames, resampled as **9 blocks of 30 frames**.

> **This interval is WITHIN ONE SEQUENCE.** It measures whether a difference holds across the segments of this single flight. It is *not* evidence that the difference generalises to another flight — two videos cannot support that claim, and resampling one of them harder does not change what was measured.

| method | AP | Δ vs baseline | 95% CI on Δ | p | verdict |
|---|---|---|---|---|---|
| **singleframe** (baseline) | 0.061 | — | — | — | — |
| temporal | 0.834 | +0.773 | [+0.610, +0.910] | 0.0000 | **better** |
| yolomg | 0.422 | +0.360 | [+0.195, +0.578] | 0.0000 | **better** |

### False alarms on labelled distractors

| method | mover1 | mover2 | total |
|---|---|---|---|
| singleframe | 0 | 0 | 0 |
| temporal | 0 | 0 | 0 |
| yolomg | 1 | 1 | 2 |
