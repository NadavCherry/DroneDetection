# City defence — Rivermark, four-camera ring

24 engagements. The interceptor rises over the middle of the town and holds; an intruder arrives from a bearing drawn from the whole circle and flies at the nearest building. It is 1.5x slower than the interceptor and it does not break off.

## Outcome

| outcome | n | share |
|---|---|---|
| **intruder intercepted** | **24** | **100.0%** |
| **building struck** | **0** | 0.0% |
| neither (timed out / never acquired) | 0 | 0.0% |

Interception rate **24/24 = 100.0%**, 95% CI [86.2%, 100.0%] (Wilson). **0 buildings hit.**

## By building

| building | asset range | engagements | intercepted | struck | median margin | mean CPA |
|---|---|---|---|---|---|---|
| plaza_e | 35 m | 13 | 13 | 0 | 4.47 s | 0.08 m |
| plaza_f | 93 m | 11 | 11 | 0 | 3.90 s | 0.08 m |

## The intercepts themselves

| metric | value |
|---|---|
| mean true closest approach | **0.080 m** |
| median / p90 | 0.073 / 0.144 m |
| inside 0.5 m | 100% |
| median time to spare before the strike | **4.29 s** |
| worst margin | 2.82 s |
| median time to intercept | 6.68 s |
| median time to acquire | **0.90 s** (p90 0.90 s) |
| mean detection rate (of frames the target was in a camera) | 0.92 |
| mean track rate | 0.92 |
| **time locked on something that was not the drone** | **0.00** of tracked frames |

## Frame rate

*Not measured here.* This run scores **closure**, and its sensor is the simulator's own bounding box -- there is no detector in the loop, so the loop time below is the tracker and the guidance law alone. The pipeline rate is measured separately, with the real sensor on live Rivermark (`work/pursuit/city_pipe`, 3 engagements): appearance detector **16.2 ms** on a 640 px crop, motion detector **208.0 ms** across four 2048x704 images (threaded), tracker **7.4 ms** — **231 ms of perception, 4.4 FPS** (3.9–5.0 across the run). Aiming the network at a crop made the appearance stage roughly eight times cheaper than the nose camera's whole-frame pass (130.7 ms, `work/pursuit/final/METRICS.md`); what is left is almost entirely the classical motion stage, run over four wide images of a cluttered city.

### stage breakdown

| stage | ms | share |
|---|---|---|
| appearance detector | 0.03 | 14% |
| motion detector | 0.00 | 0% |
| tracker | 0.11 | 59% |
| guidance | 0.05 | 27% |
| **total** | **0.18** | |

- **5505.8 FPS** mean, **4395.6 FPS** at p95
- 20 Hz control budget is 50 ms — **met** (0.2 ms)

## Every engagement

| scenario | building | outcome | acquire | t | CPA | margin | det |
|---|---|---|---|---|---|---|---|
| city-000 | plaza_e | **HIT** | 0.20 s | 6.00 s | 0.076 m | 4.47 s | 0.97 |
| city-015 | plaza_e | **HIT** | 0.90 s | 7.15 s | 0.419 m | 4.69 s | 0.89 |
| city-030 | plaza_e | **HIT** | 0.90 s | 6.65 s | 0.081 m | 4.22 s | 0.88 |
| city-045 | plaza_e | **HIT** | 0.20 s | 6.55 s | 0.001 m | 5.29 s | 0.98 |
| city-060 | plaza_f | **HIT** | 0.90 s | 6.50 s | 0.033 m | 4.07 s | 0.88 |
| city-075 | plaza_f | **HIT** | 0.90 s | 6.50 s | 0.144 m | 3.79 s | 0.87 |
| city-090 | plaza_f | **HIT** | 0.20 s | 6.30 s | 0.168 m | 4.85 s | 0.98 |
| city-105 | plaza_f | **HIT** | 0.90 s | 7.00 s | 0.125 m | 3.33 s | 0.88 |
| city-120 | plaza_f | **HIT** | 0.90 s | 6.85 s | 0.099 m | 2.82 s | 0.88 |
| city-135 | plaza_f | **HIT** | 0.20 s | 6.75 s | 0.121 m | 3.90 s | 0.99 |
| city-150 | plaza_f | **HIT** | 0.90 s | 6.65 s | 0.046 m | 2.86 s | 0.88 |
| city-165 | plaza_f | **HIT** | 0.90 s | 6.60 s | 0.022 m | 3.83 s | 0.88 |
| city-180 | plaza_f | **HIT** | 0.20 s | 6.55 s | 0.036 m | 5.05 s | 0.98 |
| city-195 | plaza_f | **HIT** | 0.90 s | 7.50 s | 0.079 m | 5.45 s | 0.89 |
| city-210 | plaza_f | **HIT** | 0.90 s | 6.85 s | 0.008 m | 4.91 s | 0.88 |
| city-225 | plaza_e | **HIT** | 0.20 s | 6.75 s | 0.048 m | 5.84 s | 0.99 |
| city-240 | plaza_e | **HIT** | 0.90 s | 7.25 s | 0.079 m | 5.39 s | 0.89 |
| city-255 | plaza_e | **HIT** | 0.90 s | 6.70 s | 0.031 m | 4.35 s | 0.88 |
| city-270 | plaza_e | **HIT** | 0.20 s | 6.90 s | 0.034 m | 5.51 s | 0.99 |
| city-285 | plaza_e | **HIT** | 0.90 s | 6.50 s | 0.081 m | 3.45 s | 0.88 |
| city-300 | plaza_e | **HIT** | 0.90 s | 7.75 s | 0.019 m | 3.98 s | 0.90 |
| city-315 | plaza_e | **HIT** | 0.20 s | 6.10 s | 0.081 m | 3.24 s | 0.98 |
| city-330 | plaza_e | **HIT** | 0.90 s | 6.45 s | 0.021 m | 3.05 s | 0.88 |
| city-345 | plaza_e | **HIT** | 0.90 s | 7.30 s | 0.069 m | 4.57 s | 0.89 |
