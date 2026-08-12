# Verified measurements — SpeckLock, 2026-08-12

Everything here I ran myself on this machine against the committed code and weights.
No claim in this file comes from an agent report; where an agent's finding conflicted
with a measurement, the measurement won and the correction is noted.

Commands are reproducible from the repo root with `.venv/bin/python`.

---

## 1. The shipped models reproduce

`final/run_final.py` runs today on both profiles and both reproduce their headline number.

| profile | fps (no TensorRT engine) | tracks emitted | AP(centre) on 10_06 | P | R | FP/frame |
|---|---|---|---|---|---|---|
| EDGE-RT | 52.6 (`.pt` fallback; 74 claimed with FP16 engine) | **1** | **1.000** | 1.000 | 1.000 | 0.000 |
| PC-MAX | 32 ms/frame detector | **4** | **1.000** | 1.000 | 1.000 | 0.000 |

95 % block-bootstrap CI on AP (30-frame blocks): **[1.000, 1.000]** for both.

So "AP/F1 = 1.000 on the unseen test video" is **true and reproducible**. The crude
accusations — that the `ignore` convention inflates it, or that the number is stale —
are both wrong, and I checked each directly.

## 2. …but PC-MAX writes three spurious drone alarms that the metric never sees

`alarms.txt` from the PC-MAX run I just did:

```
track  4 [drone]: frames 23-360 (338 covered), confirmed at frame 30 (latency 7 frames)
track 54 [drone]: frames 113-280 (165 covered), not confirmed
track 64 [drone]: frames 116-266 (151 covered), not confirmed
track 43 [drone]: frames 110-257 (148 covered), not confirmed
```

Distance from each track to the real drone, per frame, median:

| track | frames | score | median distance to the real drone |
|---|---|---|---|
| 4 | 23–360 (338) | 0.995 | **1 px** — this is the drone |
| 54 | 113–280 (165) | 0.313 | 212 px |
| 64 | 116–266 (151) | 0.356 | 140 px |
| 43 | 110–257 (148) | 0.356 | 133 px |

Three false tracks, each sustained for ~150 frames (5 s), each labelled `[drone]`.
AP is 1.000 anyway because their scores fall below the best-F1 threshold, so the
metric drops them. **The AP claim survives; the operational "zero false positives"
claim does not.** EDGE-RT emits one track and is genuinely clean.

This is the sharpest finding of the audit, and it is a reporting problem rather than a
detector problem: the number reported is not the number a user of the system experiences.

## 3. COCO AP is structurally 0.000 — the repo cannot be compared to any published work

| detections | AP(centre) | COCO AP | AP50 |
|---|---|---|---|
| EDGE-RT track-integrated, 10_06 | 1.000 | **0.000** | 0.000 |
| PC-MAX track-integrated, 10_06 | 1.000 | **0.000** | 0.001 |
| PC-MAX raw, 07_05 | 0.916 | 0.000 | 0.000 |
| moe3-stacked raw, 07_05 | 0.767 | 0.098 | **0.373** |

Cause, traced end to end:

* `tools/make_datasets_v3.py:50` — `LABEL = 24.0`. Every training label on the user's
  own data is a 24 × 24 square regardless of the drone's true size.
* The detector therefore always predicts ~24 px boxes (measured: det width is a
  constant 24.0 px, range 24.0–24.0).
* `dronedet/track.py:180` EMAs the detector's box into the track, so tracks inherit 24 px.
* The 10_06 ground truth is itself a synthetic constant: **GT box width is exactly 8.0 px
  on every one of 337 frames**.
* Measured det-vs-GT IoU on 10_06: mean 0.111, **max 0.111**, fraction ≥ 0.5 = **0.000**.
  The ceiling is (8·8)/(24·24) = 0.111, so no COCO threshold can ever match.

`moe3-stacked` scores AP50 0.373 precisely because it is the one method that predicts
variable-extent boxes. This is the difference between "cannot be compared" and "can".

**The 24 px inflation was a deliberate, well-motivated choice** — IoU-based label
assignment starves few-pixel GTs of positive anchors, which is documented in round 1.
But `dronedet/nwd.py` now solves that same problem the way the literature does. Keeping
both means paying the comparability cost twice.

## 4. Birds: the pipeline is much better than the docs claim, and the docs cannot say so

`work/gt_user.json` contains **8 bird tracks / 934 boxes**, all flagged `ignore=True`,
so `dronedet/evaluate.py:54-56` drops every detection that lands on one — neither TP
nor FP, and never counted. Bird rejection was the pipeline's hardest job and was
going entirely unscored.

Size distributions, measured:

| object | instances | median sqrt(area) | range |
|---|---|---|---|
| drone (`far`) | 548 | **8.0 px** | 3.7 – 15.3 |
| birds (8 tracks) | 934 | **6.0 px** | 2.9 – 11.7 |

51 % of drone instances fall in the AI-TOD **very-tiny** bin (2–8 px). Birds and drone
overlap almost completely, so nothing can separate them by size.

Bird hits at matched drone recall (07_05, τ = 12 px):

| method | recall | threshold | bird hits / 934 | other FP/frame |
|---|---|---|---|---|
| PC-MAX raw | 0.80 | 0.900 | **0** | 0.046 |
| PC-MAX raw | 0.90 | 0.771 | **0** | 0.091 |
| PC-MAX raw | 0.95 | 0.440 | **151** | 6.394 |
| moe3-v3 raw | 0.80 | 0.383 | 9 | 6.440 |
| **PC-MAX track-integrated** | **0.998** | 0.258 | **0** | **0.002** |

The headline: **99.8 % recall on the drone with 0 hits on 934 bird instances and
0.002 FP/frame.** The raw detector cannot do this — at 95 % recall it takes 151 bird
hits. The temporal track evidence is what separates a 6 px bird from an 8 px drone,
and that is a real, defensible scientific claim the repo currently does not make.

Caveat that must ship with it: n = 8 bird tracks, one video, one flock, one afternoon.
It is an existence proof, not a generalisation.

## 5. Test-set hygiene: weights are clean, hyperparameters are not

* **No dataset builder ever reads `10_06.mp4`.** Checked every one
  (`make_datasets_v3.py`, `make_dataset_ft6/ft7.py`, `make_fusion_combined.py`,
  `realtime/tools/make_datasets_rt.py`). The weights are genuinely held out. The
  crude leakage charge is false.
* **But the pipeline was tuned against it.** `tools/make_datasets_v3.py:14` says the
  augmentation policy targets "the 10_06 foliage-crossing misses", and
  `dronedet/trackclass.py:5` opens "Measured on 07_05 (moe3) **+ 10_06**" above six
  hand-set constants (`CONF_FRAC`, `N_CONF`, `LONG_TRACK`, `DRONE_SCORE`, `BIG_W`,
  `MATCH_DIST`).

The honest phrasing is **development test set**, not "unseen". That distinction is
exactly what a reviewer checks first.

## 6. Sample size behind every headline detection number

| | count |
|---|---|
| videos | **2** (07_05 train/val, 10_06 test) |
| distinct drones | **1 per video** |
| scored drone instances | 548 (07_05) + 337 (10_06) = **885** |
| bird instances | 934, all in one flock in one video |
| frames excluded from 10_06 as uncertain GT | 24 (v2); the earlier v1 excluded 111 |
| independent flights | **2** |

A 30-frame block bootstrap gives ~12 independent looks on 10_06. The CI is [1.000, 1.000]
— which means "perfectly consistent within this one flight", and says nothing at all
about a second flight. Reporting a single AP from one video in the same table as
published benchmark results is the specific thing that reads as amateurish.

## 6b. ARD-MAV, measured from the real annotations (downloaded 2026-08-12)

14.6 GB from the [GLAD repo's Google Drive](https://github.com/WestlakeIntelligentRobotics/Global-Local-MAV-Detection),
60 videos, **107,497 annotation files** — matching the published frame count exactly.
106,456 boxes parsed by `tools/dataset_stats.py`.

| | value |
|---|---|
| sqrt(area) | min **1.7**, p25 8.4, **median 11.8**, p75 20.7, max 102.8 px |
| AI-TOD bins | very-tiny **21.3 %**, tiny **43.8 %**, small 26.0 %, medium 8.9 % |
| official 15-video test split | median target **12.1 px** (range 6.3–28.8) |
| whole dataset, per-video medians | median **11.2 px** |

### The headline ARD-MAV number is measured on the third-easiest video in the dataset

`phantom16` is the single clip behind the README's **ARD-MAV AP 0.994**. Ranked by median target
size across all 60 videos (rank 0 = smallest targets = hardest):

| | median target | rank |
|---|---|---|
| **phantom16 (the headlined clip)** | **39.1 px** | **57 of 59 — the 97th percentile** |
| dataset median | 11.2 px | — |
| official test split median | 12.1 px | — |
| hardest five | phantom63 6.3 · phantom75 6.7 · phantom38 6.9 · phantom64 6.9 · phantom43 7.3 px | 0–4 |

The targets in that clip are **3.5× larger** than the benchmark's median, which puts them in the
AI-TOD *medium* bin. A 39 px drone is not a few-pixel target, so the number does not test this
project's thesis at all — and it is the number the README leads with.

### Label inflation: what it costs, per dataset

`_iou_after_inflation` gives the ceiling for a perfectly-centred prediction (a real one is worse),
because a concentric grown box contains the true box, so IoU = true area / inflated area.

**ARD-MAV** (`make_dataset_external.py --min-side`, default 12):

| min_side | boxes grown | median best IoU | can reach IoU 0.5 | can reach IoU 0.25 |
|---|---|---|---|---|
| 8 | 29.3 % | 1.000 | **96.9 %** | 99.9 % |
| **12 (current default)** | 59.0 % | 0.833 | **74.4 %** | 95.0 % |
| 16 | 72.9 % | 0.547 | 52.5 % | 78.7 % |
| 24 | 88.0 % | 0.243 | 32.0 % | 49.1 % |

**07_05, our own data** (`make_datasets_v3.py:50`, `LABEL = 24.0`):

| min_side | boxes grown | median best IoU | can reach IoU 0.5 | can reach IoU 0.25 |
|---|---|---|---|---|
| 8 | 77.9 % | 0.802 | **90.0 %** | 99.8 % |
| 12 | 98.9 % | 0.440 | 38.1 % | 89.1 % |
| 16 | 100.0 % | 0.249 | 4.4 % | 49.5 % |
| **24 (current)** | 100.0 % | **0.110** | **0.0 %** | 2.9 % |

Two conclusions, and the first corrects something I asserted before measuring it:

* **ARD-MAV at min_side 12 is tolerable, not fatal** — 95 % of boxes can still reach MGMD's IoU 0.25
  and 74 % can reach 0.5. I had assumed it was disqualifying; it is not. Dropping to 8 would take
  IoU-0.5 reachability to 96.9 %.
* **Our own data at `LABEL = 24` is fatal.** Median ceiling 0.110 — which is exactly the 0.111 max
  IoU I measured on real detections in §3, from a completely independent direction. Zero per cent of
  boxes can reach IoU 0.5, so COCO AP is not low, it is *arithmetically impossible*.

## 7. Engineering hygiene

| check | result |
|---|---|
| `pyproject.toml` / installable package | **absent** |
| dependencies pinned | yes — `requirements.txt` pins exact versions (better than expected) |
| random seed in training | **none set** anywhere in `tools/train_yolo.py` |
| type-hint coverage (`dronedet` + `tools` + `realtime`) | 189/353 functions = **54 %** |
| `print()` vs `logging` | 296 `print()`, **0** `import logging` |
| tests covering the detection half | **0** before today (pytest.ini pinned `testpaths = pursuit/tests`) |
| weights committed to git | ~230 MB across `work/models`, `final/`, `work/runs` |
| external datasets present | **none** — `data/external/` does not exist; ARD-MAV and NPS were deleted |

## 8. What I changed today

* `dronedet/metrics.py` — new. Centre-distance **and** IoU/COCO AP from the same
  detections, AI-TOD size bins, confuser (bird) accounting as a first-class outcome,
  moving-block bootstrap CIs, and `pick_threshold`/`summarise` split so an operating
  point can be chosen on val and applied to test.
* `dronedet/tests/test_metrics.py` — 27 tests, including a regression test for the
  GT-is-(cx,cy,w,h) convention that I got wrong on the first attempt (it cost w/2 px of
  phantom error and would have silently poisoned every number), and a test that pins
  the COCO-AP-is-zero-from-box-extent behaviour so the cause stays documented.
* `dronedet/cli.py` — `python -m dronedet bench`, which prints the table above.
* `pytest.ini` — `testpaths` now includes `dronedet/tests`. Full suite: **567 pass**.

Nothing was removed, no existing number was overwritten, nothing was pushed.
