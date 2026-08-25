# 10_06 — one held-out flight

Protocol `specklock-centre` (centre distance, tau=12 px), 250 labelled instances over 250 frames, resampled as **9 blocks of 30 frames**.

> **This interval is WITHIN ONE SEQUENCE.** It measures whether a difference holds across the segments of this single flight. It is *not* evidence that the difference generalises to another flight — two videos cannot support that claim, and resampling one of them harder does not change what was measured.

| method | AP | Δ vs baseline | 95% CI on Δ | p | verdict |
|---|---|---|---|---|---|
| **singleframe** (baseline) | 0.094 | — | — | — | — |
| temporal | 0.773 | +0.679 | [+0.444, +0.875] | 0.0000 | **better** |
| yolomg | 0.648 | +0.555 | [+0.323, +0.773] | 0.0000 | **better** |

### False alarms on labelled distractors

| method | mover1 | total |
|---|---|---|
| singleframe | 0 | 0 |
| temporal | 0 | 0 |
| yolomg | 13 | 13 |
