# Detection method comparison (tau=12.0px, center-distance matching)

| method | AP(far) | AP(all) | best-F1 | R(far) | R(near) | P | FP/frame | med-err(far) px | fps |
|---|---|---|---|---|---|---|---|---|---|
| rt-c-full1280 | 0.894 | 0.894 | 0.894 | 0.844 | 0.000 | 0.950 | 0.04 | 0.7 | 78.1 |
| rt-d-full640 | 0.702 | 0.702 | 0.774 | 0.652 | 0.000 | 0.953 | 0.03 | 1.7 | 103.6 |
| rt-b-verify256 | 0.519 | 0.519 | 0.700 | 0.676 | 0.000 | 0.725 | 0.26 | 0.4 | 24.4 |
| rt-e-decimated | 0.356 | 0.356 | 0.456 | 0.332 | 0.000 | 0.728 | 0.12 | 0.4 | 28.5 |
| rt-f-single1280 | 0.188 | 0.188 | 0.274 | 0.220 | 0.000 | 0.362 | 0.39 | 0.9 | 69.9 |
| rt-a-classic | 0.180 | 0.180 | 0.355 | 0.504 | 0.000 | 0.274 | 1.34 | 0.4 | 36.1 |

AP(far): average precision on the tiny moving drone only (near drone as ignore).
R/P/FP at the best-F1 threshold of each method. fps is end-to-end single-thread.