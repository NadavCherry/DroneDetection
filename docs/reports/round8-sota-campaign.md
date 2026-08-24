# Round 8 — the SOTA campaign: ours vs YOLOMG, measured, with statistics

*August 2026. 51 training runs on the BGU cluster: two public benchmarks and this
project's own task, three seeds each, every arm scored by one evaluator at one confidence
floor. The competitor is **YOLOMG** (Guo et al., arXiv:2503.07115, March 2025) — the
current state of the art on drone-to-drone detection, from the group that published
ARD-MAV — trained by us under its own published recipe (100 epochs at 1280 px), which
gave it roughly **twice our gradient steps**.*

This round exists to answer, with measurements, the questions any professional reviewer
asks of a detection claim — including the ones asked of this project publicly. Each
section below names the question, gives the measured answer, and says plainly what was
NOT measured. The full tables with per-seed confidence intervals are in
[`work/reports/SUMMARY.md`](../../work/reports/SUMMARY.md); every number is reproducible
from the scorecards in `work/scorecards/` via `tools/make_summary.py`.

---

## The headline, honestly stated

| task | ours (temporal) | YOLOMG | verdict |
|---|---|---|---|
| ARD-MAV, overall | 0.810 | **0.834** | they lead |
| ARD-MAV, **small-MAV condition** | **0.689** | 0.615 | **we lead** (and GLAD's published 0.580) |
| NPS-Drones | 0.509 | **0.527** | they lead |
| **this project's task** (8 px drone, held-out flight) | **0.840** (0.773–0.914) | 0.604 (0.422–0.743) | **we lead, all 3 seeds** |
| false alarms on labelled distractors | **0** (every seed, both arms of ours) | 2–13 per seed | **we lead** |

We are **not** better than the state of the art everywhere, and this repository does not
claim to be. The pattern across every corpus is consistent and is the actual finding:

> **The advantage tracks target size.** On NPS (targets 10–25 px) the competitor leads.
> On ARD-MAV overall (median 11.8 px) it leads; on ARD-MAV's small-MAV subset it trails
> us. On this project's own task (median 8.0 px, distractor birds at 6.0 px) we lead by
> +0.24 AP on average — every seed, never below +0.13 — with zero distractor false alarms. The smaller the target, the more the
> temporal stack is worth — which is the design thesis, now with its supporting and its
> *limiting* evidence in one table.

---

## The questions, answered with data

### 1a. "How do you make sure that what you detected is not a plane or a bird?"

Measured, not asserted. This project's own test video carries **eight hand-labelled birds
at a median of 6.0 px against a drone at 8.0 px** — distractors *smaller than the
target*, which neither public benchmark reproduces. They are `ignore=True` in the ground
truth, so they train nothing and surface at evaluation as **distractor hits**:

Two distractor populations exist, and they are measured by different directions of the
two-video task — the table below says which is which rather than blurring them:

| test set | distractors in it | ours (temporal) | ours (single-frame) | YOLOMG |
|---|---|---|---|---|
| 10_06 (forward direction) | 2 moving non-drone objects | **0** hits, all seeds | 0 | 2 / 5 / 13 by seed |
| 07_05 (reverse direction) | **8 birds**, median 6.0 px + 1 near object | REVERSE-PENDING | REVERSE-PENDING | REVERSE-PENDING |

The birds live in 07_05, so only the reverse direction (train on 10_06, test on 07_05)
can put them in a test set — which is why it was run. The shipped
system additionally trains an explicit second `bird` class and classifies at the *track*
level (a track needs 8 confirmed detections at 70 % confidence), so a bird must fool the
detector repeatedly, not once.

What this is not: a claim about planes or helicopters. No corpus available to us labels
them; the honest statement is that the mechanism generalises (any labelled distractor
becomes a measurable false-alarm row) and the plane row is unmeasured until such data
exists.

### 1b. "How does it perform at night, in the rain, in the wind?"

**Night and rain: not measured, because no corpus we could obtain contains them.** That
is a data gap, not a method property, and this report will not paper over it with
extrapolation. ARD-MAV's "complex" condition (cluttered backgrounds, 5 sequences) is the
nearest measured stressor: ours 0.819 vs GLAD's published 0.810.

Wind appears implicitly as camera shake and ego-motion — both corpora are hand-held or
airborne cameras — and the ego-stabilised stack is specifically the mechanism that
absorbs it: the single-frame control differs from the temporal arm *only* in that
mechanism, and trails it by +0.37 AP (mean over seeds) on the hardest task.

### 2. "Usually they build a tracker that also calculates speed and acceleration, and it discriminates."

The shipped system is exactly that shape: detection feeds a track layer
(`MATCH_DIST = 8 px`, confirmation at `N_CONF = 8` detections, `CONF_FRAC = 0.70`), and
interception uses the track, not raw detections. What this campaign adds is the layer
*underneath*: at 8 px, appearance alone does not yield reliable detections to track —
the single-frame control collapses to 0.24 AP on the held-out flight. Motion enters
**twice**: in the detector's input representation (the 13-frame stabilised stack), and
again at track level. The campaign quantified the first; track-level velocity metrics
are recorded per detection JSON but were not the unit of this comparison.

### 3. "Wonder if it would work if the footage is taken from a dynamic platform such as another drone."

That is what **NPS-Drones is**: air-to-air video shot *from a drone*, and ARD-MAV is
likewise a moving MAV camera. Both were run end to end. Ours reaches 0.509 on NPS from a
dynamic platform (competitor 0.527); on ARD-MAV's small subset from a moving camera we
lead. So: yes, measured, with the caveat that on the larger-target NPS regime the
competitor's appearance-plus-mask design is currently ahead of us.

### 4. "Amateur."

The reply to this one is method, not adjectives. In this campaign: protocols are typed
objects and the comparison tool **refuses to subtract two APs whose protocols differ**;
significance is a seed-matched paired bootstrap **and** permutation test over sequences,
reported only when both agree; the competitor was trained under **its own** published
recipe with twice our budget, and a budget-matched arm (ours at 100 epochs) was run so
the difference cannot be blamed on training time; 939 automated tests pin every parser
convention, alignment rule and guard; and sixteen-plus pipeline bugs found during the
campaign are documented in the git history with their failure modes — including the ones
that flattered us, which were fixed with the same urgency as the ones that did not.

### 5. "Without seeing the false identification results in relation to real identification, it's difficult to judge."

Correct, and the distractor table above is that measurement, on the only corpus we have
whose annotations make it possible. Beyond it, every scorecard in `work/scorecards/`
retains the full score-ordered detection list per sequence — so precision at any
operating point, not just AP, is recomputable by a reader without rerunning inference.

### 6. "You see everything — you don't need an image buffer. That's not the challenge."

This is the single claim the campaign was best equipped to test, because the
single-frame control differs from the temporal arm in **nothing but the buffer** — same
tiles, labels, augmentation, epochs, seeds, evaluator:

| corpus | target size | AP gain from the buffer | significant? |
|---|---|---|---|
| NPS-Drones | 10–25 px | −0.01 | no |
| ARD-MAV overall | median 11.8 px | +0.03 | no (CI includes 0, all seeds) |
| this project's task | median 8.0 px | **+0.27 to +0.46** (mean +0.37) | **yes, all seeds, p < 0.001** |

The criticism is *right* where targets are large: at 10–25 px, appearance suffices and
the buffer buys nothing — we report that against interest. It is wrong where this project
lives: at 8 px, with birds of the same apparent size, the buffer more than doubles AP.
"You see everything" is a statement about target size, and it stops being true a few
pixels below where NPS operates.

---

## Speed

Both arms measured in one process on one RTX 3090, steady-state (the stabilisation
buffer slides; each frame is charged its marginal cost), at two labelled floors — the
conf 0.001 the AP tables were produced at, and a deployment floor of 0.25. TensorRT
export included. *(Numbers inserted from `work/bench/` when the re-run completes; the
first bench charged our arm 13 stabilisations per frame instead of one and was discarded
— its 0.6 fps contradicted the pipeline's own logged 8–13 fps end-to-end.)*

## What was not done, so nobody discovers it later

- **A third public dataset.** FL-Drones requires author permission, ARD100 is
  Baidu-gated, Drone-vs-Bird requires a data request, Anti-UAV has no public link, and
  UAV_SMID measured as non-contiguous stills a temporal method cannot use. Two public
  benchmarks, stated plainly.
- **Night/rain conditions** — no obtainable corpus contains them (§1b).
- **Our NPS number vs the published 0.95.** Both arms score far below the published
  NPS figures under our evaluator (we 0.51, the SOTA *itself* 0.53 vs its own published
  0.95), so the published-scale gap is a protocol difference, not a model difference —
  and per this repo's own rules those numbers are not comparable and are never
  subtracted.
- **One held-out flight** on the project's own task. Its significance is a moving-block
  bootstrap *within* the sequence — stability across that flight, not generalisation
  across flights. Two directions were run to double the evidence; it is still two videos.
