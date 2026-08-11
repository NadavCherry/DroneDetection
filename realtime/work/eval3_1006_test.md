# Detection method comparison (tau=12.0px, center-distance matching)

| method | AP(far) | AP(all) | best-F1 | R(far) | R(near) | P | FP/frame | med-err(far) px | fps |
|---|---|---|---|---|---|---|---|---|---|
| tracked-rt-c-full1280 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.00 | 0.7 | 84.8 |
| rt-c-full1280 | 0.889 | 0.889 | 0.848 | 0.795 | 0.000 | 0.908 | 0.08 | 0.6 | 84.8 |
| tracked-rt-d-full640 | 0.668 | 0.668 | 0.760 | 0.668 | 0.000 | 0.882 | 0.09 | 1.4 | 104.5 |
| rt-d-full640 | 0.631 | 0.631 | 0.741 | 0.599 | 0.000 | 0.971 | 0.02 | 1.4 | 104.5 |

AP(far): average precision on the tiny moving drone only (near drone as ignore).
R/P/FP at the best-F1 threshold of each method. fps is end-to-end single-thread.