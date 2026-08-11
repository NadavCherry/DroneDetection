# End-to-end analysis — 62 engagements

## 1. Hit rate

**54/62 = 87.1%**   95% CI [76.6%, 93.3%]  (Wilson)

Failures split by cause — only the second kind is the guidance law's:

| failure | n | meaning |
|---|---|---|
| never acquired | 3 | the detector never produced a confirmed track |
| acquired, did not close | 5 | locked on and still failed to arrive |

## 2. Where the intercepts land

| statistic | value |
|---|---|
| mean true closest approach | **0.239 m** |
| median | 0.188 m |
| p90 / p95 / max | 0.474 / 0.569 / 0.630 m |
| best | 0.025 m |
| inside 0.5 m | 93% |

- **vertical**: -1.8 cm ± 10.2 (t=-1.27) — centred
- **lateral**: -0.7 cm ± 25.8 (t=-0.19) — centred

## 3. Does anything predict failure?

Holm-corrected across all factors — testing six things at p<0.05 finds one by luck about a quarter of the time.

| factor | p (raw) | p (Holm) | significant |
|---|---|---|---|
| det (hit vs miss) | 0.0000 | 0.0001 | **yes** |
| trk (hit vs miss) | 0.0472 | 0.2361 | no |
| entry: - vs inbound | 0.0789 | 0.3158 | no |
| r0 (hit vs miss) | 0.2477 | 0.7431 | no |
| scene: rivermark vs skydome | 0.2554 | 0.7431 | no |
| policy: barrel vs flee | 0.2768 | 0.7431 | no |

## 4. Perception versus guidance

- detection rate on **hits**: 0.865 (median 0.892, n=54)
- detection rate on **misses**: 0.291 (median 0.168, n=8)
- Mann-Whitney p = 0.0000
- ranges overlap: hits [0.42, 0.99], misses [0.00, 0.78]

## 5. Frame rate

| stage | ms | share |
|---|---|---|
| detector | 130.69 | 100% |
| tracker | 0.10 | 0% |
| guidance | 0.06 | 0% |
| **total** | **130.85** | |

- **7.6 FPS** mean, **6.4 FPS** at p95
- 20 Hz control budget is 50 ms — **NOT met** (130.8 ms)

## 6. Time to intercept

median **8.95 s**, p90 14.70 s, max 47.00 s
