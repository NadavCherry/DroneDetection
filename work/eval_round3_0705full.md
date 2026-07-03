# Detection method comparison (tau=12.0px, center-distance matching)

| method | AP(far) | AP(all) | best-F1 | R(far) | R(near) | P | FP/frame | med-err(far) px | fps |
|---|---|---|---|---|---|---|---|---|---|
| tracked-pcmax | 0.998 | 0.998 | 0.998 | 0.998 | 0.000 | 0.998 | 0.00 | 0.9 | 4.0 |
| tracked-moe3v3 | 0.962 | 0.962 | 0.974 | 0.964 | 0.000 | 0.985 | 0.01 | 1.0 | 4.0 |
| tracked-moe3-cls | 0.962 | 0.962 | 0.974 | 0.964 | 0.000 | 0.985 | 0.01 | 1.0 | 4.0 |
| fullS-s1280 | 0.922 | 0.922 | 0.884 | 0.819 | 0.000 | 0.959 | 0.03 | 0.6 | 44.6 |
| pc-max | 0.916 | 0.916 | 0.915 | 0.894 | 0.000 | 0.937 | 0.06 | 0.7 | 4.0 |
| moe3-stacked | 0.767 | 0.767 | 0.861 | 0.794 | 0.000 | 0.942 | 0.05 | 1.0 | 4.0 |
| moe3-stacked | 0.754 | 0.754 | 0.831 | 0.772 | 0.000 | 0.900 | 0.09 | 1.0 | 4.0 |

AP(far): average precision on the tiny moving drone only (near drone as ignore).
R/P/FP at the best-F1 threshold of each method. fps is end-to-end single-thread.