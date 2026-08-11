# Detection method comparison (tau=12.0px, center-distance matching, frames 342..570)

| method | AP(far) | AP(all) | best-F1 | R(far) | R(near) | P | FP/frame | med-err(far) px | fps |
|---|---|---|---|---|---|---|---|---|---|
| tracked-rt-c-full1280 | 0.995 | 0.995 | 0.995 | 0.995 | 0.000 | 0.995 | 0.00 | 1.4 | 76.3 |
| tracked-rt-d-full640 | 0.738 | 0.738 | 0.849 | 0.738 | 0.000 | 1.000 | 0.00 | 1.7 | 106.5 |
| rt-c-full1280 | 0.717 | 0.717 | 0.730 | 0.670 | 0.000 | 0.802 | 0.17 | 1.2 | 76.3 |
| rt-d-full640 | 0.243 | 0.243 | 0.351 | 0.223 | 0.000 | 0.821 | 0.05 | 2.0 | 106.5 |

AP(far): average precision on the tiny moving drone only (near drone as ignore).
R/P/FP at the best-F1 threshold of each method. fps is end-to-end single-thread.