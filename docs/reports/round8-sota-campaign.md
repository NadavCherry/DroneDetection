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
from the scorecards in [`work/scorecards/`](../../work/scorecards/) via
`PYTHONPATH=. python tools/make_summary.py --scorecards work/scorecards`. Those 42
scorecards ship **gzipped** (`*.json.gz`, 29.1 MB for 623.7 MB of raw evidence);
`benchmarks.scorecard.Scorecard.load` reads either form.

---

## The headline, honestly stated

| task | ours (temporal) | YOLOMG | verdict |
|---|---|---|---|
| ARD-MAV, overall | 0.809 | **0.834** | they lead |
| ARD-MAV, **small-MAV condition** | **0.689** | 0.619 | point estimate favours us on **every seed** (+0.06–+0.08), but **not significant at n=5** — the permutation floor is 0.03 and p ≈ 0.3. GLAD's published: 0.580 |
| NPS-Drones | 0.487 | **0.527** | they lead |
| **this project's task** (8 px drone, held-out flight; both arms fine-tuned from NPS weights) | **0.840** (0.773–0.914) | 0.604 (0.422–0.743) | **we lead, all 3 seeds** |
| false alarms on 10_06's labelled distractors | **0**, every seed | 2–13 per seed | **we lead** |
| bird hits per detected drone (07_05, deployed floor) | **~0.21** | not measurable‡ | see §1a |

We are **not** better than the state of the art everywhere, and this repository does not
claim to be. The pattern across every corpus is consistent and is the actual finding:

> **The advantage tracks target size.** On NPS (targets 10–25 px) the competitor leads.
> On ARD-MAV overall (median 11.8 px) it leads; on ARD-MAV's small-MAV subset the point
> estimate favours us on every seed (five sequences cannot make that significant). On this project's own task (median 8.0 px, distractor birds at 6.0 px) we lead by
> +0.24 AP on average — every seed, never below +0.12. Distractors are a split result and
> §1a gives both halves: zero hits on 10_06's two moving objects, but birds DO draw fire
> on 07_05 — what separates the arms there is selectivity, roughly half the bird hits per
> drone found, not immunity. The smaller the target, the more the
> temporal stack is worth — which is the design thesis, now with its supporting and its
> *limiting* evidence in one table.

---

## The questions, answered with data

### 1a. "How do you make sure that what you detected is not a plane or a bird?"

Measured, not asserted. This project's own test video carries **eight hand-labelled birds
at a median of 6.0 px against a drone at 8.0 px** — distractors *smaller than the
target*, which neither public benchmark reproduces. They are `ignore=True` in the ground
truth, so they train nothing and surface at evaluation as **distractor hits**:

Two distractor populations exist, and they behave differently enough that reporting one
as if it were both would mislead in whichever direction was picked:

**10_06's two moving objects** (forward direction, both arms fine-tuned): our arms placed
**zero** detections on them at any confidence, every seed; YOLOMG placed 2, 5 and 13.

**07_05's eight birds** (reverse direction — the only test set containing birds — at the
deployed floor, conf ≥ 0.25, 571 frames, 3 seeds):

| arm | drone true positives | bird hits | bird hits per detected drone |
|---|---|---|---|
| **ours, temporal** | **406 / 381 / 421** | 73 / 96 / 86 | **0.18 / 0.25 / 0.20** |
| ours, single-frame | 184 / 28 / 178 | 87 / 8 / 67 | 0.47 / 0.29 / 0.38 |
| YOLOMG (from scratch)‡ | 0 / 2 / 0 | 0 | not measurable |

At the deployed floor the temporal arm finds **3.1× the drones** of its own control
(1,208 vs 390 true positives pooled over seeds; 2.2× against the control's best seed) at
roughly **half the bird hits per drone found**. Birds are not invisible to it — they
move, and a motion-sensitive detector sees them — but its per-detection selectivity is
better, and the two failure modes mirror the inputs: the single-frame arm's stray mass
sits on the *static* near object (759–991 low-confidence hits at the evaluation floor,
all gone by 0.25), the temporal arm's on the *moving* birds. At the 0.001 evaluation
floor every arm hits birds by the hundreds; the unthresholded tables are in
`work/reports/local_rev_07_05_seed*.md`. Single detections are also not what the shipped
system acts on: a track needs 8 confirmed detections at 70 % confidence to exist.

‡ *YOLOMG trained from scratch on this direction's 250 frames and did not converge
(AP 0.000–0.007; its own validation mAP ≈ 0) — the same small-data collapse the forward
direction showed before fine-tuning. Its zero bird hits are an artefact of detecting
almost nothing, and count neither for it nor against it.*

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
mechanism, and trails it by +0.37 AP on the hardest task (the from-scratch A/B of §6; the gap widens to +0.69 when both arms are fine-tuned).

### 2. "Usually they build a tracker that also calculates speed and acceleration, and it discriminates."

The shipped system is exactly that shape: detection feeds a track layer
(`MATCH_DIST = 8 px`, confirmation at `N_CONF = 8` detections, `CONF_FRAC = 0.70`), and
interception uses the track, not raw detections. What this campaign adds is the layer
*underneath*: at 8 px, appearance alone does not yield reliable detections to track —
the from-scratch single-frame control averages 0.24 AP on the held-out flight (0.15 fine-tuned). Motion enters
**twice**: in the detector's input representation (the 13-frame stabilised stack), and
again at track level. The campaign quantified the first; track-level velocity metrics
are recorded per detection JSON but were not the unit of this comparison.

### 3. "Wonder if it would work if the footage is taken from a dynamic platform such as another drone."

That is what **NPS-Drones is**: air-to-air video shot *from a drone*, and ARD-MAV is
likewise a moving MAV camera. Both were run end to end. Ours reaches 0.487 on NPS from a
dynamic platform (competitor 0.527); on ARD-MAV's small subset from a moving camera we
lead. So: yes, measured, with the caveat that on the larger-target NPS regime the
competitor's appearance-plus-mask design is currently ahead of us.

### 4. "Amateur."

The reply to this one is method, not adjectives. In this campaign: protocols are typed
objects and the comparison tool **refuses to subtract two APs whose protocols differ**;
significance is a seed-matched paired bootstrap **and** permutation test over sequences,
reported only when both agree; the competitor was trained under **its own** published
recipe with twice our budget, and a budget-matched arm (ours at 100 epochs) was run so
the difference cannot be blamed on training time; 941 automated tests (939 passing, 2 environment-gated skips) pin every parser
convention, alignment rule and guard; and sixteen-plus pipeline bugs found during the
campaign are documented in the git history with their failure modes — including the ones
that flattered us, which were fixed with the same urgency as the ones that did not.

### 5. "Without seeing the false identification results in relation to real identification, it's difficult to judge."

Correct, and the distractor table above is that measurement, on the only corpus we have
whose annotations make it possible. Beyond it, every scorecard in `work/scorecards/`
retains the full score-ordered detection list per sequence, down to the 0.001
confidence floor — so precision at any operating point, not just AP, is recomputable
by a reader without rerunning inference. That tail is two thirds of each file and the
reason they are shipped gzipped rather than not shipped at all.

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

Both arms in one process on one RTX 3090, fp16, steady-state — the stabilisation buffer
slides, so each frame is charged its **marginal** cost, which is what a video pipeline
pays. (A first bench charged us 13 stabilisations per frame and read 0.6 fps; it was
discarded because it contradicted the pipeline's own logged 8–13 fps, and the error is
documented in `tools/sota/bench_both.py`.) Two labelled floors, because they answer
different questions:

| configuration | ours (8×640 px tiles) | YOLOMG (1280 px dual-stream) |
|---|---|---|
| conf 0.25 — as deployed | **146 ms → 6.8 fps** | 312 ms → 3.2 fps |
| conf 0.001 — as the AP tables were scored | 478 ms → 2.1 fps | 315 ms → 3.2 fps |
| of which: front end | 120 ms (stabilise + stack + tile) | **226 ms** (mask32: 2× KLT + RANSAC, CPU) |

Deployed, ours is **2.1× faster** despite running eight tiles per frame; at the
evaluation floor our NMS over eight tiles' candidates dominates and the competitor is
faster. The competitor's floor barely matters to it because its cost is its CPU mask.
TensorRT FP16 (batch-8 engine, same card): **139 ms → 7.2 fps** deployed, 468 ms at the
evaluation floor. The engine cuts network+NMS from 26 ms to ~19 ms; the 120 ms CPU front
end now dominates, so the next speed lever is stabilisation, not the network. *(The
first export segfaulted after ultralytics' auto-updater swapped a dependency mid-process;
that updater is now disabled in every job and the re-run exported cleanly.)*

## What was not done, so nobody discovers it later

- **A third public dataset.** FL-Drones requires author permission, ARD100 is
  Baidu-gated, Drone-vs-Bird requires a data request, Anti-UAV has no public link, and
  UAV_SMID measured as non-contiguous stills a temporal method cannot use. Two public
  benchmarks, stated plainly.
- **Night/rain conditions** — no obtainable corpus contains them (§1b).
- **Our NPS number vs the published 0.95.** Both arms score far below the published
  NPS figures under our evaluator (we 0.49, the SOTA *itself* 0.53 vs its own published
  0.95), so the published-scale gap is a protocol difference, not a model difference —
  and per this repo's own rules those numbers are not comparable and are never
  subtracted.
- **One held-out flight** on the project's own task. Its significance is a moving-block
  bootstrap *within* the sequence — stability across that flight, not generalisation
  across flights. Two directions were run to double the evidence; it is still two videos.
