# Result — ARD-MAV, official 15-video split, at the published protocol

**Run 2026-08-13.** The first number this project has produced that can legitimately sit
beside a published one.

Reproduce:

```bash
python tools/train.py --config singleframe_ardmav --seed 0 --batch 16 --device 0
python tools/infer_tiled.py --weights work/runs/singleframe_ardmav-s0/weights/best.pt \
    --mode rgb --gt-dir work/ext_datasets/gt/ardmav --out-dir work/det_bench/singleframe_ardmav
python tools/evaluate.py --dataset ardmav --model singleframe_ardmav \
    --gt work/ext_datasets/gt/ardmav --dets work/det_bench/singleframe_ardmav \
    --protocol ardmav-official --official-split --split official-test-15 \
    --conditions work/ext_datasets/ardmav_conditions.json --out work/scorecards/singleframe_ardmav.json
python tools/compare.py --scorecards work/scorecards/singleframe_ardmav.json \
    --vs-published --dataset ardmav
```

---

## 1. The protocol, and the correction that had to come first

**GLAD, arXiv 2312.11008 / IEEE T-ITS 2024**, experiments section, verbatim:

> "Following the protocol in [21], the performance evaluation is based on Precision,
> Recall, F-Score, and AP. We set the intersection over union (IOU) threshold between
> predictions and ground truths to **0.5**."

on "15 videos from the ARD-MAV dataset", 28,322 frames. The 15 IDs are stated in GLAD's
repository README and again in `GLAD.py:31-33`.

Until 2026-08-12 this repo recorded the ARD-MAV bar as **MGMD, AP 0.55 at IoU 0.25, on the
official 15-video split** — one `Protocol` object spliced from two papers, binding MGMD's
threshold to GLAD's split under a citation naming both. MGMD's 0.55 is at IoU 0.25 on a
split it never enumerates. The real bar is **higher and at a threshold twice as strict**,
and because the splice lived in `benchmarks/protocol.py`, it was compiled into the engine
that decides what is comparable: any ARD-MAV number produced before that fix was scored
against nothing published.

## 2. The result

`singleframe_ardmav`, **seed 0 only**, YOLOv8s-p2 @ 640 px, true extents (`min_side 0`) +
NWD, 30 epochs, batch 16. Scored by `tools/infer_tiled.py` over full frames on a 640 px
grid with 128 px overlap, duplicates merged by centre distance.

| | AP@0.5 | 95 % CI | comparable? |
|---|---|---|---|
| **ours** | **0.754** | [0.638, 0.850] | — |
| GLAD | 0.800 | — | ✅ indistinguishable at this sample size |
| TPH-YOLOv5l | 0.730 | — | ✅ indistinguishable at this sample size |
| MGMD | 0.550 @ IoU 0.25 | — | ❌ different threshold **and** split |

n = 15 sequences / 28,160 instances. "Indistinguishable" means our interval covers their
point estimate. **No p-value appears and none can**: a published AP is a single scalar with
no distribution, so there is nothing to test against. That is a limit of the comparison,
not a result — see §5.

## 3. By GLAD's own conditions, which is the comparison that means something

GLAD splits the 15 test videos three ways and reports each (`GLAD.py:31-33`):

| condition | sequences | n_gt | **ours** | **GLAD** | |
|---|---|---|---|---|---|
| ordinary | 5 | 9,230 | **0.948** | 0.91 | **we win** |
| complex | 5 | 9,578 | 0.762 | 0.81 | we lose |
| **small MAVs** | 5 | 9,352 | **0.530** | **0.58** | **we lose** |

Precision/recall, ours: ordinary 0.908/0.928, complex 0.869/0.708, small 0.756/0.476.

**We beat the incumbent where targets are large and lose where they are small.** Recall is
what collapses — 0.928 on ordinary against 0.476 on small — while precision holds at 0.756.
The detector is not confusing clutter for drones; it is failing to fire on them.

That is the shape a single-frame appearance detector should have, and it is exactly where
a stabilised temporal stack is supposed to earn its keep. The temporal arm therefore has a
specific number to beat — **0.530, on named videos, at a published protocol** — rather than
an argument.

## 4. What this run is not

* **One seed.** `local_extent` showed sd 0.1140 on COCO AP across three seeds of the same
  config, driven by one arm early-stopping. A single seed here is a point, not a mean.
* **The interval is wide** — [0.638, 0.850], because 15 sequences is a small n for a
  block bootstrap. It covers GLAD's 0.800 and TPH-YOLOv5l's 0.730 simultaneously, which
  says the sample cannot separate them, not that they are equal.
* **28,160 instances against GLAD's stated 28,322** — 0.6 % fewer. Our GT is parsed from
  the released per-frame XML; the difference is small but unexplained, and until it is,
  the two totals are not quite the same population.
* **Not this project's method.** This is the single-frame control. The contribution under
  test is the temporal stack, and it has not run here yet.

## 5. Why there is no p-value, and what would produce one

Against a published scalar there is nothing to test: one number has no distribution. The
table reports our interval and whether it covers their point estimate, and prints no p.

A real significance test needs a rival **run on our sequences**, which makes the comparison
paired and admits a bootstrap and a permutation test over sequences. That is now possible:
GLAD ships its weights (mirrored to `work/mirrors/glad`, 113 MB — global detector, local
detector, appearance classifier). Two caveats travel with any such run:

* **No licence file** in GLAD's repository, so all rights are reserved by default.
  Producing comparison numbers is ordinary practice; redistributing or shipping a
  derivative is not.
* The release is **not the published method** — its README states the Kalman filter and
  adaptive search region are unreleased. A re-run measures GLAD-as-released and must be
  labelled that way; scoring under 0.80 would be the missing components, not a refutation.
