# Detection method comparison (tau=12.0px, center-distance matching, frames 342..570)

| method | AP(far) | AP(all) | best-F1 | R(far) | R(near) | P | FP/frame | med-err(far) px | fps |
|---|---|---|---|---|---|---|---|---|---|
| tracked-moe3v3 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.00 | 1.0 | 4.0 |
| tracked-moe3-cls | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.00 | 1.0 | 4.0 |
| tracked-pcmax | 0.995 | 0.995 | 0.995 | 0.995 | 0.000 | 0.995 | 0.00 | 1.1 | 4.0 |
| pc-max | 0.775 | 0.775 | 0.804 | 0.757 | 0.000 | 0.857 | 0.13 | 0.9 | 4.0 |
| fullS-s1280 | 0.744 | 0.744 | 0.696 | 0.568 | 0.000 | 0.900 | 0.06 | 1.0 | 44.6 |
| moe3-stacked | 0.700 | 0.700 | 0.828 | 0.796 | 0.000 | 0.863 | 0.13 | 0.9 | 4.0 |
| moe3-stacked | 0.656 | 0.656 | 0.760 | 0.767 | 0.000 | 0.752 | 0.25 | 1.0 | 4.0 |

AP(far): average precision on the tiny moving drone only (near drone as ignore).
R/P/FP at the best-F1 threshold of each method. fps is end-to-end single-thread.