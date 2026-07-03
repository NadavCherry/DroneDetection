# Detection method comparison (tau=12.0px, center-distance matching, frames 342..570)

| method | AP(far) | AP(all) | best-F1 | R(far) | R(near) | P | FP/frame | med-err(far) px | fps |
|---|---|---|---|---|---|---|---|---|---|
| rt-c-full1280 | 0.679 | 0.679 | 0.658 | 0.587 | 0.000 | 0.747 | 0.20 | 1.0 | 74.3 |
| rt-b-verify256 | 0.388 | 0.388 | 0.516 | 0.743 | 0.000 | 0.395 | 1.14 | 0.9 | 19.4 |
| rt-e-decimated | 0.300 | 0.300 | 0.389 | 0.413 | 0.000 | 0.368 | 0.71 | 0.9 | 22.1 |
| rt-d-full640 | 0.220 | 0.220 | 0.372 | 0.335 | 0.000 | 0.418 | 0.47 | 3.0 | 102.0 |
| rt-a-classic | 0.217 | 0.217 | 0.326 | 0.636 | 0.000 | 0.219 | 2.27 | 0.9 | 27.6 |
| rt-f-single1280 | 0.091 | 0.091 | 0.183 | 0.121 | 0.000 | 0.373 | 0.20 | 1.3 | 83.1 |

AP(far): average precision on the tiny moving drone only (near drone as ignore).
R/P/FP at the best-F1 threshold of each method. fps is end-to-end single-thread.