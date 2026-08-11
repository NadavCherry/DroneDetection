# Detection method comparison (tau=12.0px, center-distance matching)

| method | AP(far) | AP(all) | best-F1 | R(far) | R(near) | P | FP/frame | med-err(far) px | fps |
|---|---|---|---|---|---|---|---|---|---|
| tracked-pcmax | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.00 | 0.9 | 4.4 |
| tracked-moe3-cls | 0.996 | 0.996 | 0.997 | 0.997 | 0.000 | 0.997 | 0.00 | 1.0 | 4.1 |
| tracked-moe3v3 | 0.996 | 0.996 | 0.997 | 0.997 | 0.000 | 0.997 | 0.00 | 0.9 | 4.4 |
| pc-max | 0.910 | 0.910 | 0.871 | 0.834 | 0.000 | 0.912 | 0.08 | 0.7 | 4.4 |
| fullS-s1280 | 0.879 | 0.879 | 0.813 | 0.881 | 0.000 | 0.754 | 0.29 | 0.7 | 50.5 |
| moe3-stacked | 0.778 | 0.778 | 0.857 | 0.757 | 0.000 | 0.988 | 0.01 | 0.8 | 4.4 |

AP(far): average precision on the tiny moving drone only (near drone as ignore).
R/P/FP at the best-F1 threshold of each method. fps is end-to-end single-thread.