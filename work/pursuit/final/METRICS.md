# End-to-end pursuit metrics

54 of 62 engagements intercepted (**87.1%**) across 2 environments.

A hit is a true closest approach inside 1.0 m — two Iris airframes touching are about 0.5 m centre to centre.

## Headline

| metric | value |
|---|---|
| engagements | 62 |
| intercepts | **54 / 62** (87.1%) |
| mean true miss distance | **0.239 m** |
| median / p95 true miss | 0.188 m / 0.569 m |
| best pass | 0.025 m |
| median time to intercept | **8.95 s** |
| p95 time to intercept | 30.45 s |
| median time to acquire | 3.65 s |
| start range | 30–115 m (median 70 m) |

## Frame rate

Pipeline rate — detector + tracker + guidance. **Not** the test run's wall-clock rate, which is dominated by Isaac rendering five frames per control tick and does not exist on the aircraft.

| stage | ms / frame | share |
|---|---|---|
| detector | 130.7 | 100% |
| tracker | 0.10 | 0% |
| guidance | 0.06 | 0% |
| **total** | **130.8** | |

- **7.6 FPS** mean, **6.5 FPS** at p95 (worst-frame)
- control loop runs at 20 Hz (50 ms), so the pipeline does NOT meet real time (130.8 ms vs 50 ms budget)
- the detector is 100% of the cost: it is the only stage worth optimising for edge hardware

## By environment

| environment | intercepts | mean miss | median t | det rate | FPS |
|---|---|---|---|---|---|
| Rivermark (urban) | **25/31** (81%) | 0.273 m | 8.95 s | 72.8% | 7.8 |
| Skydome (open sky) | **29/31** (94%) | 0.210 m | 8.95 s | 85.5% | 7.6 |

## By approach direction

How the intruder arrived, which is the axis a real engagement varies along.

| arrival | intercepts | mean miss | median t | median acquire |
|---|---|---|---|---|
| `behind` | **1/2** | 0.461 m | 30.45 s | 35.40 s |
| `crossing` | **1/2** | 0.025 m | 8.90 s | 7.75 s |
| `far-left` | **2/2** | 0.513 m | 15.07 s | 6.75 s |
| `far-right` | **2/2** | 0.302 m | 14.27 s | 9.40 s |
| `head-on` | **2/2** | 0.366 m | 10.68 s | 6.10 s |
| `high-left` | **2/2** | 0.087 m | 8.80 s | 3.35 s |
| `high-right` | **2/2** | 0.171 m | 6.55 s | 3.80 s |
| `inbound` | **4/6** | 0.285 m | 11.32 s | 6.72 s |
| `left` | **2/2** | 0.240 m | 8.62 s | 4.30 s |
| `left-to-right` | **6/6** | 0.256 m | 11.93 s | 4.88 s |
| `low-left` | **2/2** | 0.279 m | 12.68 s | 3.00 s |
| `low-right` | **2/2** | 0.133 m | 27.80 s | 19.33 s |
| `outbound` | **5/6** | 0.148 m | 7.60 s | 0.25 s |
| `overhead` | **2/2** | 0.316 m | 8.10 s | 5.00 s |
| `right` | **1/2** | 0.534 m | 9.45 s | 6.45 s |
| `right-to-left` | **4/6** | 0.277 m | 13.88 s | 8.65 s |

## By evasion policy

| policy | intercepts | mean miss | median t |
|---|---|---|---|
| `barrel` | **8/8** | 0.338 m | 8.57 s |
| `break_turn` | **2/2** | 0.184 m | 5.70 s |
| `evasive` | **7/8** | 0.241 m | 8.90 s |
| `flee` | **14/18** | 0.241 m | 9.50 s |
| `jink` | **2/2** | 0.141 m | 3.42 s |
| `orbit` | **2/2** | 0.056 m | 6.10 s |
| `sweep` | **2/2** | 0.079 m | 3.72 s |
| `weave` | **17/20** | 0.248 m | 9.65 s |

## Where the intercepts land

Signed, in the chaser's own axes at the true closest approach. Positive vertical means it passed above the target.

| axis | mean | sd |
|---|---|---|
| vertical | -1.8 cm | 10.2 cm |
| lateral | -0.7 cm | 25.8 cm |

## Every engagement

| scenario | env | arrival | evasion | start | result | miss | t | acquire | det | trk | FPS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `L1-sweep` | rivermark | - | `sweep` | 35 m | **HIT** | 0.043 m | 3.40 s | 0.45 s | 0.96 | 0.96 | 7.9 |
| `L2-weave` | rivermark | - | `weave` | 35 m | **HIT** | 0.346 m | 6.05 s | 1.20 s | 0.98 | 0.98 | 7.9 |
| `L3-barrel` | rivermark | - | `barrel` | 35 m | **HIT** | 0.195 m | 7.45 s | 1.20 s | 0.72 | 0.98 | 7.4 |
| `L4-orbit` | rivermark | - | `orbit` | 35 m | **HIT** | 0.069 m | 5.70 s | 0.25 s | 0.97 | 0.97 | 7.8 |
| `L5-break_turn` | rivermark | - | `break_turn` | 35 m | **HIT** | 0.198 m | 5.00 s | 0.35 s | 0.98 | 0.98 | 8.2 |
| `L6-jink` | rivermark | - | `jink` | 35 m | **HIT** | 0.151 m | 3.00 s | 1.10 s | 0.92 | 0.97 | 8.4 |
| `L7-evasive` | rivermark | - | `evasive` | 35 m | **HIT** | 0.030 m | 4.90 s | 2.05 s | 0.67 | 0.94 | 7.9 |
| `in-behind` | rivermark | behind | `flee` | 106 m | miss | - m | - s | 49.80 s | 0.59 | 0.79 | 7.0 |
| `in-crossing` | rivermark | crossing | `evasive` | 115 m | miss | - m | - s | 9.75 s | 0.62 | 0.71 | 6.7 |
| `in-far-left` | rivermark | far-left | `barrel` | 88 m | **HIT** | 0.569 m | 9.25 s | 6.75 s | 0.84 | 0.45 | 7.8 |
| `in-far-right` | rivermark | far-right | `evasive` | 97 m | **HIT** | 0.247 m | 14.70 s | 10.60 s | 0.96 | 0.82 | 7.6 |
| `in-head-on` | rivermark | head-on | `barrel` | 106 m | **HIT** | 0.462 m | 10.55 s | 6.10 s | 0.86 | 0.51 | 7.8 |
| `in-high-left` | rivermark | high-left | `weave` | 115 m | **HIT** | 0.114 m | 9.10 s | 3.65 s | 0.96 | 0.90 | 8.5 |
| `in-high-right` | rivermark | high-right | `barrel` | 70 m | **HIT** | 0.301 m | 6.65 s | 3.65 s | 0.93 | 0.53 | 7.9 |
| `in-left` | rivermark | left | `flee` | 70 m | **HIT** | 0.360 m | 7.20 s | 5.20 s | 0.62 | 0.87 | 8.0 |
| `in-low-left` | rivermark | low-left | `flee` | 88 m | **HIT** | 0.389 m | 12.75 s | 2.55 s | 0.91 | 0.85 | 8.7 |
| `in-low-right` | rivermark | low-right | `weave` | 97 m | **HIT** | 0.040 m | 8.95 s | 5.70 s | 0.72 | 0.80 | 7.9 |
| `in-overhead` | rivermark | overhead | `evasive` | 79 m | **HIT** | 0.474 m | 10.85 s | 8.05 s | 0.73 | 0.96 | 7.8 |
| `in-right` | rivermark | right | `weave` | 79 m | **HIT** | 0.534 m | 9.45 s | 6.45 s | 0.85 | 0.89 | 7.8 |
| `inbound-45m` | rivermark | inbound | `flee` | 45 m | **HIT** | 0.628 m | 6.50 s | 3.00 s | 0.88 | 0.95 | 8.3 |
| `inbound-70m` | rivermark | inbound | `flee` | 70 m | miss | - m | - s | 26.30 s | 0.14 | 0.58 | 7.2 |
| `inbound-95m` | rivermark | inbound | `flee` | 95 m | **HIT** | 0.246 m | 12.25 s | 7.75 s | 0.88 | 0.77 | 8.0 |
| `left-to-right-45m` | rivermark | left-to-right | `weave` | 45 m | **HIT** | 0.414 m | 13.05 s | 3.15 s | 0.89 | 0.98 | 8.1 |
| `left-to-right-70m` | rivermark | left-to-right | `weave` | 70 m | **HIT** | 0.451 m | 14.20 s | 11.90 s | 0.42 | 0.61 | 7.3 |
| `left-to-right-95m` | rivermark | left-to-right | `weave` | 95 m | **HIT** | 0.053 m | 10.80 s | 5.95 s | 0.77 | 0.76 | 7.8 |
| `outbound-30m` | rivermark | outbound | `flee` | 30 m | **HIT** | 0.178 m | 4.85 s | 0.30 s | 0.98 | 0.98 | 7.5 |
| `outbound-45m` | rivermark | outbound | `flee` | 45 m | miss | - m | - s | 6.80 s | 0.20 | 0.61 | 7.1 |
| `outbound-60m` | rivermark | outbound | `flee` | 60 m | **HIT** | 0.181 m | 8.95 s | 0.20 s | 0.98 | 0.99 | 8.9 |
| `right-to-left-45m` | rivermark | right-to-left | `weave` | 45 m | miss | - m | - s | - s | 0.00 | 0.34 | 6.8 |
| `right-to-left-70m` | rivermark | right-to-left | `weave` | 70 m | **HIT** | 0.145 m | 18.10 s | 12.05 s | 0.65 | 0.71 | 7.5 |
| `right-to-left-95m` | rivermark | right-to-left | `weave` | 95 m | miss | - m | - s | - s | 0.00 | 0.86 | 7.3 |
| `L1-sweep` | skydome | - | `sweep` | 35 m | **HIT** | 0.116 m | 4.05 s | 1.15 s | 0.81 | 0.81 | 7.8 |
| `L2-weave` | skydome | - | `weave` | 35 m | **HIT** | 0.245 m | 6.85 s | 1.75 s | 0.89 | 0.89 | 7.9 |
| `L3-barrel` | skydome | - | `barrel` | 35 m | **HIT** | 0.409 m | 7.90 s | 3.10 s | 0.84 | 0.84 | 8.0 |
| `L4-orbit` | skydome | - | `orbit` | 35 m | **HIT** | 0.043 m | 6.50 s | 1.55 s | 0.88 | 0.88 | 7.9 |
| `L5-break_turn` | skydome | - | `break_turn` | 35 m | **HIT** | 0.170 m | 6.40 s | 1.60 s | 0.88 | 0.88 | 8.0 |
| `L6-jink` | skydome | - | `jink` | 35 m | **HIT** | 0.131 m | 3.85 s | 2.20 s | 0.74 | 0.74 | 7.9 |
| `L7-evasive` | skydome | - | `evasive` | 35 m | **HIT** | 0.395 m | 6.30 s | 3.15 s | 0.71 | 0.71 | 7.2 |
| `in-behind` | skydome | behind | `flee` | 106 m | **HIT** | 0.461 m | 30.45 s | 21.00 s | 0.92 | 0.61 | 7.1 |
| `in-crossing` | skydome | crossing | `evasive` | 115 m | **HIT** | 0.025 m | 8.90 s | 5.75 s | 0.92 | 0.52 | 7.4 |
| `in-far-left` | skydome | far-left | `barrel` | 88 m | **HIT** | 0.458 m | 20.90 s | - s | 0.66 | 0.81 | 7.0 |
| `in-far-right` | skydome | far-right | `evasive` | 97 m | **HIT** | 0.358 m | 13.85 s | 8.20 s | 0.94 | 0.44 | 7.1 |
| `in-head-on` | skydome | head-on | `barrel` | 106 m | **HIT** | 0.271 m | 10.80 s | 6.10 s | 0.90 | 0.68 | 7.3 |
| `in-high-left` | skydome | high-left | `weave` | 115 m | **HIT** | 0.061 m | 8.50 s | 3.05 s | 0.97 | 0.77 | 7.7 |
| `in-high-right` | skydome | high-right | `barrel` | 70 m | **HIT** | 0.041 m | 6.45 s | 3.95 s | 0.89 | 0.49 | 7.4 |
| `in-left` | skydome | left | `flee` | 70 m | **HIT** | 0.120 m | 10.05 s | 3.40 s | 0.92 | 0.83 | 7.7 |
| `in-low-left` | skydome | low-left | `flee` | 88 m | **HIT** | 0.169 m | 12.60 s | 3.45 s | 0.96 | 0.84 | 7.6 |
| `in-low-right` | skydome | low-right | `weave` | 97 m | **HIT** | 0.226 m | 46.65 s | 32.95 s | 0.97 | 0.93 | 7.4 |
| `in-overhead` | skydome | overhead | `evasive` | 79 m | **HIT** | 0.158 m | 5.35 s | 1.95 s | 0.91 | 0.68 | 7.4 |
| `in-right` | skydome | right | `weave` | 79 m | miss | - m | - s | - s | 0.00 | 0.69 | 7.0 |
| `inbound-45m` | skydome | inbound | `flee` | 45 m | **HIT** | 0.118 m | 10.95 s | 2.00 s | 0.96 | 0.83 | 8.0 |
| `inbound-70m` | skydome | inbound | `flee` | 70 m | miss | - m | - s | 23.30 s | 0.78 | 0.89 | 7.1 |
| `inbound-95m` | skydome | inbound | `flee` | 95 m | **HIT** | 0.148 m | 11.70 s | 5.70 s | 0.89 | 0.76 | 7.6 |
| `left-to-right-45m` | skydome | left-to-right | `weave` | 45 m | **HIT** | 0.498 m | 13.25 s | 2.40 s | 0.97 | 0.91 | 7.8 |
| `left-to-right-70m` | skydome | left-to-right | `weave` | 70 m | **HIT** | 0.076 m | 5.20 s | 3.80 s | 0.77 | 0.95 | 7.5 |
| `left-to-right-95m` | skydome | left-to-right | `weave` | 95 m | **HIT** | 0.046 m | 10.40 s | 6.30 s | 0.84 | 0.63 | 7.5 |
| `outbound-30m` | skydome | outbound | `flee` | 30 m | **HIT** | 0.116 m | 6.15 s | 1.85 s | 0.82 | 0.82 | 8.0 |
| `outbound-45m` | skydome | outbound | `flee` | 45 m | **HIT** | 0.065 m | 7.60 s | 0.20 s | 0.99 | 0.99 | 7.7 |
| `outbound-60m` | skydome | outbound | `flee` | 60 m | **HIT** | 0.197 m | 8.95 s | 0.20 s | 0.99 | 0.99 | 7.6 |
| `right-to-left-45m` | skydome | right-to-left | `weave` | 45 m | **HIT** | 0.630 m | 9.65 s | 5.25 s | 0.89 | 0.79 | 7.6 |
| `right-to-left-70m` | skydome | right-to-left | `weave` | 70 m | **HIT** | 0.123 m | 9.65 s | 4.10 s | 0.94 | 0.66 | 7.8 |
| `right-to-left-95m` | skydome | right-to-left | `weave` | 95 m | **HIT** | 0.211 m | 47.00 s | 34.80 s | 0.92 | 0.83 | 7.4 |
