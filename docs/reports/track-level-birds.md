# Bird rejection, measured at the track

**Every "track-level" bird number this project has published until now was a per-frame
number in disguise.** `tools/tracks_to_dets.py` flattens tracks back into per-frame boxes
and `dronedet.metrics` scores those, which answers "how many frames contained a spurious
box". That is not the operational question. A bird raised for 150 consecutive frames is
**one** false alarm to an operator, not 150.

`tools/track_level_birds.py` scores tracks as tracks. Nothing below is derived from a
per-frame AP.

## What is being counted

07_05 carries eight hand-labelled bird tracks — **934 instances, median 6.0 px**, the same
size band as the 8.0 px drone — flagged `ignore` in the ground truth.

One correction had to be made before any of this meant anything. **Not every `ignore`
object is a bird.** 07_05's ignore set is those eight `bird*` tracks *plus* `near`, the
**landed drone** (571 instances) — the same aircraft sitting on the ground. The first run
of this analysis counted `near` as a distractor and duly reported "1478 detections on
birds" out of 1482 bird instances, and a bird track classified `near`. What it had
actually found was the detector correctly finding the landed drone and the classifier
correctly labelling it. Birds are now identified by name prefix, the convention
`dronedet.metrics.Summary.confuser_hits(prefixes=("bird",))` already used, and other
ignore objects are scored neither for nor against.

## Results

Both videos, PC-MAX, `CONF_FRAC = 0.70`, `N_CONF = 8`, `LONG_TRACK = 120`,
`DRONE_SCORE = 0.35`.

### Before tracking — what the detector produced

| video | detections ≥ 0.25 | on target | **on bird** | on landed drone | on nothing |
|---|---|---|---|---|---|
| 07_05 | 6262 | 525 | **440** | 1038 | 4259 |
| 10_06 | 2348 | 327 | — *(no birds labelled)* | 81 | 1940 |

The detector fires on birds constantly: **440 detections land on labelled birds** on 07_05.
Any claim that this pipeline "does not see birds" is false at the detection stage. What it
does is refuse to *raise* them.

### After tracking — the confusion matrix

**07_05**

| classified as | target | bird | landed drone | nothing |
|---|---|---|---|---|
| **drone** | 1 | **0** | 1 | 10 |
| **near** | 0 | **0** | 1 | 1 |
| **other** (rejected) | 0 | **3** | 0 | 3 |

**10_06** (no labelled birds)

| classified as | target | landed drone | nothing |
|---|---|---|---|
| **drone** | 1 | 0 | 4 |
| **other** (rejected) | 0 | 0 | 1 |

### Operating point

| | 07_05 | 10_06 |
|---|---|---|
| tracks raised as a target | 14 (2 on the landed drone, not scored) | 5 |
| genuinely the drone | 1 | 1 |
| **bird false alarms** | **0** | — |
| clutter false alarms | **11** | **4** |
| track precision | **0.083** | **0.200** |
| target objects recovered | 1.000 (1/1) | 1.000 (1/1) |

## What this establishes, and what it costs us

**Bird rejection is real, and it is now measured where the decision is made.** Three bird
tracks form on 07_05 and the track classifier rejects all three. Zero birds are raised as
targets, over 934 bird instances that produced 440 detections. That is the strongest form
of this claim the project has ever had, because it is no longer a per-frame count.

**Clutter rejection is not real, and the per-frame metric was hiding that.** Eleven tracks
on 07_05 and four on 10_06 are raised as `drone` while sitting on nothing. They are not
low-confidence noise — several run 150–330 frames (5–11 seconds at 30 fps) with `conf_frac`
between 0.77 and 1.000, comfortably past both thresholds:

| id | frames | tracked | n_conf | conf_frac | what it is on |
|---|---|---|---|---|---|
| 243 | 326 | 72 | 71 | 0.986 | nothing |
| 612 | 239 | 50 | 50 | 1.000 | nothing |
| 636 | 238 | 170 | 132 | 0.776 | nothing |
| 645 | 235 | 192 | 148 | 0.771 | nothing |
| 196 | 207 | 28 | 28 | 1.000 | nothing |

Per-frame AP reports 1.000 on 10_06 anyway, because AP is score-weighted and these tracks
fall below the best-F1 threshold — exactly the failure
[`INFRA.md` audit item 6](../research/INFRA.md) predicted: *"Report tracked results as
coverage plus false-track count; reserve AP for the per-frame detector."* This is that
report, and it says track precision is **0.083 on the hard video**.

**A claim in the round-3 report is false.** It states that *"the labeled GT birds never
even form tracks (the verifier already suppresses them below tracker threshold)"*. They do
form tracks — three of them. The conclusion survives, the stated mechanism does not: birds
are stopped by the **track classifier**, not by the verifier, and the difference matters
because it is the classifier's thresholds that would have to hold on new data.

**The `LONG_TRACK` bypass never fired.** `dronedet.trackclass` promotes any track of ≥ 120
tracked frames to `drone` with **no appearance evidence at all**, and
`dronedet/track.py` says in its own comment that the kinematic filter does not exclude
birds. It is the one rule that could raise a bird without the verifier ever agreeing.
Measured: **0 tracks promoted by it on either video.** The clause is currently harmless —
but it is harmless by luck of the data, not by design, and it is worth keeping this
measurement whenever the classifier changes.

## A defect this work fixed

`tools/eval_tracks.py` decided whether a track was false by looping over **all** GT objects
with no `ignore` check — while the coverage loop twenty lines above it *does* check.
One file, two conventions. A track that rode a labelled bird for its whole life therefore
"matched a GT object" and never counted as a false track: 14 reported where an
ignore-aware count gives 19, and three of the five it forgave were riding birds.

Bird rejection is the hardest thing this pipeline does and the thing the project is sold
on, so a metric that silently forgave bird tracks was hiding the exact failure mode that
matters — in our favour. Tracks are now sorted three ways (true / distractor / clutter),
with distractor tracks reported separately from clutter, because "followed a real bird"
and "fired at nothing" are different failures with different fixes.

## Reproducing

```bash
PYTHONPATH=. python tools/track_level_birds.py \
    --gt work/gt_user.json \
    --tracks work/tracks3/0705/pc-max-all.json \
    --dets work/det3/0705/pc-max.json \
    --out work/reports/tracks/pcmax_0705.md
```
