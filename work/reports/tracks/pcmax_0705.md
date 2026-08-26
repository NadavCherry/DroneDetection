# Track-level bird / false-positive analysis

`pc-max-all.json` scored against `gt_user.json`. CONF_FRAC=0.7, N_CONF=8, LONG_TRACK=120, DRONE_SCORE=0.35.

A bird raised for 150 consecutive frames is ONE false alarm to an operator, not 150. Every number below counts tracks.

Ground truth: 1 target object(s) ['far'], 8 labelled bird track(s) with 934 instances, and 1 other ignore object(s) ['near'] — ignore-flagged but **not** distractors (the landed drone is the same aircraft on the ground), so they are scored neither for nor against.

## Before tracking -- what the detector produced

| detections >= 0.25 | on target | on bird | on other-ignore | on nothing |
|---|---|---|---|---|
| 6262 | 525 | 440 | 1038 | 4259 |

## After tracking -- confusion matrix

| classified as | target | bird | other-ignore | nothing | ambiguous |
|---|---|---|---|---|---|
| **drone** | 1 | 0 | 1 | 10 | 0 |
| **near** | 0 | 0 | 1 | 1 | 0 |
| **other** | 0 | 3 | 0 | 3 | 0 |

## Operating point

- tracks raised as a target: **14** (2 of them on an ignore object that is not a distractor, excluded from the scoring below)
- judged: **12**
- of those, genuinely the drone: **1**
- **bird false alarms: 0**
- clutter false alarms: 11
- track precision: **0.083**
- target objects recovered: **1.000** (1/1)

## The LONG_TRACK bypass

Tracks promoted with no appearance evidence at all, purely for lasting >= 120 frames: **0** (birds: 0, clutter: 0).

