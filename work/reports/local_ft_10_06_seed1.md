# 10_06 — one held-out flight

Protocol `specklock-centre` (centre distance, tau=12 px), 250 labelled instances over 250 frames, resampled as **9 blocks of 30 frames**.

> **This interval is WITHIN ONE SEQUENCE.** It measures whether a difference holds across the segments of this single flight. It is *not* evidence that the difference generalises to another flight — two videos cannot support that claim, and resampling one of them harder does not change what was measured.

| method | AP | Δ vs baseline | 95% CI on Δ | p | verdict |
|---|---|---|---|---|---|
| **singleframe** (baseline) | 0.291 | — | — | — | — |
| temporal | 0.914 | +0.623 | [+0.396, +0.847] | 0.0000 | **better** |
| yolomg | 0.743 | +0.452 | [+0.245, +0.664] | 0.0000 | **better** |

### False alarms on labelled distractors

| method | mover1 | mover2 | total |
|---|---|---|---|
| singleframe | 0 | 0 | 0 |
| temporal | 0 | 0 | 0 |
| yolomg | 3 | 2 | 5 |
