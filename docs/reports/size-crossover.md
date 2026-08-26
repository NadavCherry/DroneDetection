# Accuracy against target size: where the crossover is, and what survives a significance test

## Why this exists

Three of this project's results looked contradictory as single numbers:

| | ours | YOLOMG |
|---|---|---|
| ARD-MAV, official 15-video split | 0.809 | **0.834** |
| NPS-Drones, video-disjoint test | 0.487 | **0.527** |
| our own 8 px task, fine-tuned | **0.840** | 0.604 |

They are only contradictory if "accuracy" is one number. They are consistent — and far more
useful — if accuracy depends on target size and the two methods have different curves that
cross somewhere. `tools/size_curve.py` measures that instead of asserting it.

Everything is held fixed except target size: one evaluator, one matching rule, one IoU
threshold, one confidence floor, the same sequences, the same pooling. The arms differ only
in which detections file they read.

## The curve

**ARD-MAV**, official 15-video test split, 3 seeds per arm, ~28,000 GT instances.
Bins on √area in pixels. `±` is the sample standard deviation over training seeds.

| bin | n | ours | YOLOMG | ours single-frame | ours − YOLOMG |
|---|---|---|---|---|---|
| **<8 px** | 5677 | **0.503 ± 0.011** | 0.420 ± 0.017 | 0.378 ± 0.031 | **+0.083** |
| **8–10 px** | 5055 | **0.602 ± 0.007** | 0.507 ± 0.015 | 0.493 ± 0.029 | **+0.095** |
| 10–16 px | 7529 | 0.732 ± 0.013 | **0.787 ± 0.010** | 0.728 ± 0.033 | −0.056 |
| 16–25 px | 4731 | 0.729 ± 0.029 | **0.888 ± 0.006** | 0.758 ± 0.039 | −0.159 |
| >25 px | 5168 | 0.739 ± 0.023 | **0.905 ± 0.018** | 0.771 ± 0.054 | −0.167 |

**The means cross between 8–10 px and 10–16 px.** Every bin carries thousands of instances,
and the sign is consistent across all three seeds on both sides.

## What survives a significance test — the part that matters

The spreads above are each arm's own variation over seeds. They are **not** the test. The
sequences in ARD-MAV differ enormously in difficulty, and that variance is shared by both
arms, so the right test is a **paired** one over sequences — which is what the rest of this
project reports, and what `tools/make_summary.py` already does for overall AP.

Paired bootstrap **and** permutation over the 15 shared sequences, seed-matched. A bin is
called significant only when both agree:

| bin | seed 0 | seed 1 | seed 2 | verdict |
|---|---|---|---|---|
| <8 px | +0.065 (p=0.53) | +0.104 (p=0.28) | +0.079 (p=0.43) | **not significant** |
| 8–10 px | +0.076 (p=0.42) | +0.114 (p=0.36) | +0.095 (p=0.57) | **not significant** |
| 10–16 px | −0.081 (p=0.014) ✔ | −0.041 (p=0.15) | −0.045 (p=0.084) | 1 of 3 |
| 16–25 px | −0.185 (p=0.001) ✔ | −0.121 (p=0.001) ✔ | −0.171 (p=0.001) ✔ | **significant, all 3** |
| >25 px | −0.206 (p=0.001) ✔ | −0.126 (p=0.032) ✔ | −0.169 (p=0.002) ✔ | **significant, all 3** |

> **Only the competitor's side of the crossover is statistically significant.**
>
> Our advantage below 10 px points the same way on every seed and every bin, and it is not
> small (+0.065 to +0.114). But with 15 sequences and the variance between them, a paired
> test cannot distinguish it from zero — p<sub>perm</sub> between 0.28 and 0.57. Their
> advantage above 16 px clears the same bar on all three seeds with p ≈ 0.001.
>
> The honest reading: **the crossover is visible in the means and consistent in direction,
> but it is not yet established.** Three seeds all favouring us in both small bins is
> suggestive — a sign test on 3 seeds cannot reach p < 0.05 no matter how it lands — and
> more test sequences, not more seeds, is what would settle it.

## NPS cannot speak to this at all

| bin | n | ours | YOLOMG |
|---|---|---|---|
| 10–16 px | 6700 | 0.394 ± 0.052 | **0.415 ± 0.011** |
| 16–25 px | 2396 | 0.265 ± 0.055 | **0.353 ± 0.061** |

**NPS-Drones contains no ground truth below 10 px.** The `<8 px` and `8–10 px` bins are
empty — not zero, absent. No claim about tiny targets can be supported or refuted with this
dataset, in either direction. YOLOMG leads in both populated bins, significant on 1 seed of 3.

An earlier version of the tool announced a "crossover between 16–25 px and >25 px" here. That
was a bug in the reporting, not a finding: the `>25 px` bin holds 28 instances, both arms
score 0.005, and the delta is +0.000. Underpowered bins and sub-0.001 deltas are now excluded
from the crossover test, and the exclusion is printed rather than silent.

## The 8 px task, where we do win

| bin | n | ours | YOLOMG | ours single-frame |
|---|---|---|---|---|
| <8 px | 123 | **0.430** | 0.319 | 0.032 |
| 8–10 px | 81 | **0.767** | 0.536 | 0.434 |
| 10–16 px * | 46 | **0.707** | 0.317 | 0.280 |

`*` underpowered. We lead in every bin — but this is **one flight**, so the sequence-level
paired test does not apply and a sequence bootstrap is degenerate. The interval that belongs
here is a moving-block bootstrap *within* the sequence, which
[Round 8](round8-sota-campaign.md) reports for the overall number.

The single-frame column is the point worth keeping: **0.032 against 0.430 at <8 px**, a 13×
gap on the same network and recipe. Whatever is or is not true about beating YOLOMG, the
temporal stack is doing the work at this size.

## Method notes

* A false positive has no size, so it is charged to **every** bin. Otherwise a method could
  look strong on very-tiny targets by flooding the frame with large spurious boxes no
  small-target bin ever pays for. This is why per-bin AP is lower than overall AP, and why a
  bin with few GT instances scores near zero regardless of merit.
* A true positive on an out-of-bin target is dropped rather than counted as a false positive.
  Counting it as one — the naive way to subset — would depress each bin by an amount that
  depends on how the *other* bins are populated.
* Both bin sets are produced: AI-TOD's four standard bins for comparability with the
  small-object literature, and the finer "mission" bins above, because AI-TOD lumps 8–16 px
  into one cell and that interval is exactly where the transition lives.

## Reproducing

```bash
PYTHONPATH=. python tools/size_curve.py --dataset ardmav --bins mission \
    --arm ours work/ext_datasets/gt/ardmav work/det/ardmav/temporal_ardmav-e100-s0 \
    --arm yolomg work/ext_datasets/gt/ardmav work/det/ardmav/yolomg_ardmav_seed0 \
    --out work/reports/size_curve
```

Saved output: `work/reports/size_curve/{ardmav,nps,local_ft}_{mission,aitod}.{md,json}`.
