# The temporal spacing dt: a real sweep, and the rule for reading it

**Status: closed, with a negative result.** The full-frame test comparison landed. Under
the rule fixed in section 3 *before* the numbers existed, **the sweep does not establish
dt=6 on the deployed protocol** — see section 4. The validation curve in section 1 stands
as what it is: evidence about training-time fitting, not about deployed accuracy.

## Why this exists

`dt = 6` — taps at t−12 / t−6 / t — is the founding constant of this project. "Three
moments as colour channels" *is* the method, and the spacing is the one free parameter in
it. Until now its entire published justification was a single prose row in
[round 3](round3-deliverables.md) reporting that dt=9 was tried and rejected. That row
cannot be re-derived: no detection JSON, checkpoint, seed or eval file for it survives
anywhere in `work/`, `work/det3/`, `work/eval_round3_*.md` or `realtime/work/`. It was a
transcription of a run nobody can reproduce, standing in for the justification of the
project's central design choice.

This replaces it with 15 runs whose artifacts are kept.

## 1. The sweep: dt in {2, 4, 6, 8, 12}, three seeds each

Identical in every respect except the tap spacing — same `yolov8s-p2`, same 100-epoch cap,
same `patience: 25`, same batch 8, same 640 px, same seeds, same NPS training clips. Only
the dataset build differs, and only in `--dt`. Verified from each run's own `args.yaml`
rather than from directory names.

| dt | aperture | seeds | val mAP50 | per-seed | vs dt=6 |
|---|---|---|---|---|---|
| 2 | 5 frames | 3 | 0.9320 ± 0.0041 | 0.9351 / 0.9336 / 0.9273 | −0.0092 |
| 4 | 9 frames | 3 | 0.9342 ± 0.0007 | 0.9334 / 0.9348 / 0.9344 | −0.0070 |
| **6** | **13 frames** | 3 | **0.9412 ± 0.0025** | 0.9411 / 0.9387 / 0.9437 | — |
| 8 | 17 frames | 3 | 0.9351 ± 0.0018 | 0.9356 / 0.9366 / 0.9330 | −0.0061 |
| 12 | 25 frames | 3 | 0.9312 ± 0.0057 | 0.9339 / 0.9246 / 0.9352 | −0.0100 |

**An inverted U with its peak at dt=6, falling monotonically on both sides.** dt=6's
*worst* seed (0.9387) beats every one of the twelve runs at other spacings — the seed
ranges do not overlap anywhere.

The shape is interpretable rather than merely favourable. Too short an aperture (dt=2, five
frames) and a slow target has not moved far enough between taps to leave a coloured trail;
too long (dt=12, twenty-five frames) and the target's own motion smears across the channels
while accumulated camera motion and foliage sway add clutter the stabiliser cannot remove.

## 2. Why this is not yet the answer

**These are tile-level validation numbers, and this project has already measured that its
validation metric does not predict its test result.** In
[the NPS discrepancy investigation](yolomg-nps-discrepancy.md), the same weights score
val mAP50 **0.941** and full-frame test AP **0.487** — a tile metric on held-out tiles
against a full-frame metric on held-out *videos*. A spacing chosen on validation tiles is
chosen on the weaker of the two available metrics.

So the sweep above is evidence about training-time fitting, not about deployed accuracy,
and the ablation cannot close on it.

## 3. The decision rule, fixed before the test numbers exist

`cluster/dt_eval.sbatch` scores dt in {2, 4, 8, 12} x 3 seeds through the identical
pipeline the headline numbers use — tiled full-frame inference over the 10 held-out NPS
test clips, then the unified evaluator — against the dt=6 scorecards already on disk.

### 3a. The power problem, stated in advance

| metric | seed SD | effect under test |
|---|---|---|
| val mAP50 | 0.0025 | 0.006 – 0.010 |
| **full-frame test AP** | **0.0549** | 0.006 – 0.010 |

dt=6's three test-AP seeds are **0.4819 / 0.5441 / 0.4347** — mean 0.4869, spread 0.109.
**The seed noise on the test metric is 22x that on the validation metric, and 5–9x the
entire effect being measured.** A three-seed unpaired comparison on test AP therefore has
essentially no power to resolve a 0.01 difference, and would not have it even if the effect
were real and exactly as the validation sweep describes.

The test is therefore the **paired** one: paired bootstrap **and** permutation over the 10
shared test sequences, seed-matched, both required to agree — the same machinery
`tools/make_summary.py` and `tools/size_curve.py` use, for the same reason (it removes the
between-sequence variance both arms share). Even so, ARD-MAV's size curve showed a
consistent-on-every-seed effect failing to reach significance over *15* sequences. Ten is
fewer.

**The most likely outcome is "inconclusive", and that is a result, not a failure.**

### 3b. The rule

* **dt=6 ranks first AND beats the runners-up significantly** — the spacing is empirically
  established on the deployed protocol. Say so plainly.
* **dt=6 ranks first but not significantly** — the validation and test curves *agree in
  direction*, and the choice is supported but **not established**. Report the ranking, the
  p-values and the power limitation together. This must not be written as though it were
  the first outcome.
* **Another dt ranks first** — the curves **disagree**. Analyse the disagreement; it is more
  interesting than the ablation was. Do not fall back on the validation story because it is
  the more convenient one. A metric already shown not to predict test behaviour does not get
  to arbitrate when it conflicts with the metric that does.
* **No dt separates from any other** — say that the sweep does not resolve the choice on the
  deployed protocol, and that dt=6 rests on the validation curve plus the mechanism argument
  in section 1, which is weaker than this project has been claiming.

### 3c. What would invalidate the comparison

Check before reading any of it:

* any of the 12 eval tasks not `COMPLETED`;
* a detection file whose `meta.dt` is not the dt of the arm that produced it — the job
  guards this explicitly, because `tools/infer_tiled.py` **defaults `--dt` to 6**, and a
  silently-defaulted run would evaluate every checkpoint through a dt=6 stack and produce a
  plausible, meaningless curve;
* a `weights_sha256` in a new scorecard matching one from a different arm.


## 4. The verdict: the sweep does not resolve dt on the deployed protocol

All 12 evaluations completed. Every arm passed both guards — each task verified the
checkpoint's own `args.yaml` names the matching `_dt<N>/` dataset, and every one of the 10
detection files per arm recorded the correct `meta.dt`. No arm was silently scored through
a dt=6 stack.

### 4a. The two curves disagree

| dt | val mAP50 | val rank | **test AP** | **test rank** |
|---|---|---|---|---|
| 2 | 0.9320 | 4 | **0.5061** | **1** |
| 4 | 0.9342 | 3 | 0.4822 | 4 |
| **6** | **0.9412** | **1** | 0.4869 | **3** |
| 8 | 0.9351 | 2 | 0.4933 | 2 |
| 12 | 0.9312 | 5 | 0.4761 | 5 |

```
validation:  dt6 > dt8 > dt4 > dt2 > dt12
test:        dt2 > dt8 > dt6 > dt4 > dt12
```

**dt=6 ranks first on validation and third on test. dt=2 ranks fourth on validation and
first on test.** Only the last place agrees. The orderly inverted U of section 1 does not
survive the change of metric.

### 4b. But nothing separates, and the two "significant" results contradict each other

Paired bootstrap **and** permutation over the 10 shared test sequences, seed-matched.
**2 of 12 comparisons reached significance — and both belong to the same pair, pointing in
opposite directions:**

| vs dt=6 | seed | d AP | 95% CI | p perm | verdict |
|---|---|---|---|---|---|
| dt2 | 0 | −0.0090 | [−0.0617, +0.0383] | 0.889 | no difference |
| **dt2** | **1** | **+0.0538** | [+0.0047, +0.0863] | **0.011** | **dt=6 better** |
| **dt2** | **2** | **−0.1024** | [−0.1358, −0.0577] | **0.005** | **dt=2 better** |
| dt4 | 0 / 1 / 2 | −0.023 / +0.042 / −0.005 | — | 0.82 / 0.29 / 0.92 | no difference |
| dt8 | 0 / 1 / 2 | −0.018 / +0.011 / −0.012 | — | 0.77 / 0.78 / 0.71 | no difference |
| dt12 | 0 / 1 / 2 | −0.033 / +0.089 / −0.023 | — | 0.65 / 0.09 / 0.54 | no difference |

Seed 1 says dt=6 beats dt=2 significantly (p = 0.011). Seed 2 says dt=2 beats dt=6
significantly (p = 0.005). **Same comparison, same protocol, same evaluator, opposite
conclusions.** That is not a finding about dt; it is the signature of seed variance
dominating the effect, and it is worth more than the ablation it came from:

> Had this sweep been run with **one seed**, as ablations usually are, it would have
> produced "dt=2 is significantly better than dt=6 (p = 0.005)" or "dt=6 is significantly
> better than dt=2 (p = 0.011)" depending only on which seed was drawn. Both are
> publishable-looking. Both are wrong.

The power problem recorded in 3a is exactly what materialised: within-setting seed spreads
run 0.047–0.109, while the differences *between* settings span 0.005–0.030. The noise is
2–6× the effect. Twelve comparisons at α = 0.05 would yield ~0.6 false positives by chance
alone; two turned up, contradicting each other.

### 4c. What this licenses, under the rule

Section 3b's fourth outcome applies: **no dt separates from any other on the deployed
protocol.** Therefore:

* **dt=6 is NOT empirically established as optimal.** It must not be described that way,
  and this repository will not.
* **Nor is it refuted.** dt=2's higher test mean is not significant, rests on the seed that
  contradicts its own sibling, and comes from the setting whose validation score was fourth
  of five.
* What dt=6 rests on is the **validation curve** (a clean, non-overlapping inverted U over
  15 runs) plus the **mechanism argument** in section 1. That is real evidence and it is
  weaker than this project has previously implied. The honest description is *a design
  choice supported by validation and not contradicted by held-out test*, not *the measured
  optimum*.
* Resolving it would need **more test sequences**, not more seeds and not more spacings.
  NPS test has 10 clips; ARD-MAV's size curve already showed a consistent-on-every-seed
  effect failing significance over 15.

This is a better outcome than the alternative, which was to publish the validation curve as
though it settled the question. The founding constant of the project now has 27 runs behind
it — 15 training, 12 evaluation — and an explicit statement of what they do and do not
show, in place of a prose row whose run nobody kept.

## Reproducing

```bash
sbatch cluster/dt_build.sbatch          # datasets, CPU only
sbatch cluster/dt_train.sbatch          # 15 runs, 5 dt x 3 seeds
sbatch cluster/dt_eval.sbatch           # full-frame test AP, 12 tasks
sbatch cluster/dt_compare.sbatch        # paired tests -> work/reports/dt_test_ap.md
# or, without SLURM:  PYTHONPATH=. python tools/dt_compare.py --out work/reports/dt_test_ap.md
```

Artifacts: `work/runs_dt/dt{2,4,8,12}-s{0,1,2}/`, `work/det/nps_dt/`,
`work/scorecards/dt*_nps-s*.json`. The dt=6 arm is `work/runs_e100/temporal_nps-s*` and
`work/scorecards/temporal_nps-e100-s*.json`.
