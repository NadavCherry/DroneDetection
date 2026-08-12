# Result — `local_extent`: the box-extent defect, demonstrated and fixed

**Run 2026-08-12.** MISSION §4 item 1. Six runs: `baseline_local` and `trueextent_local`,
three seeds each, 60 epochs, YOLOv8s-p2 @ 640 px, batch 8, on an RTX 4080 Laptop (sm_89).
Reproduce with `python tools/train.py --config local_extent --seeds 3`.

The question: on this repo's **own** data, `tools/make_datasets_v3.py` wrote every training
label as a fixed 24 px square (`LABEL = 24.0`). Does that cost anything measurable, and does
training on true extents with NWD assignment recover it?

Both. The defect is larger than "COCO AP is low" and the fix works.

---

## 1. The headline: the same predictions, scored two ways

Protocol: IoU / COCO, **n = 206** val boxes (07_05, held-out frames 342–547, whole-frame
temporal split). Mean ± sd over 3 seeds. The val **images are byte-identical** between the
two builds (verified 206/206), so the only thing that changes between columns is which
ground truth the identical predictions are scored against.

| config | scored against | mAP50 | COCO mAP50-95 |
|---|---|---|---|
| `baseline_local` | its own 24 px labels | **0.9934** ± 0.0005 | **0.8596** ± 0.0069 |
| `baseline_local` | **true extents** | **0.0000** ± 0.0000 | **0.0000** ± 0.0000 |
| `trueextent_local` | **true extents** | **0.9204** ± 0.0987 | **0.3912** ± 0.1140 |
| `trueextent_local` | the 24 px labels | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

**Read the first two rows together.** The same weights score 0.9934 and 0.0000 depending only
on which ground truth is used. A reader shown the first row alone would conclude the detector
is near-perfect; it localises nothing. That is the entire audit finding, reproduced as an
experiment rather than asserted.

The fourth row is the symmetry check, and it is why the zero is trustworthy: true-extent
predictions score 0.000 against inflated labels for the same geometric reason, in the opposite
direction. Neither zero is a bug in the scorer.

## 2. Why the zero is forced, measured rather than argued

Predicted box widths on the same 206 val images (seed 1, conf 0.25):

| model | n_det | predicted width, median | range | distinct widths | best achievable IoU |
|---|---|---|---|---|---|
| `baseline_local` | 225 | **23.90 px** | 21.69–24.60 | **15** | **0.150** |
| `trueextent_local` | 238 | **8.80 px** | 5.17–15.10 | 58 | 1.000 |

Ground-truth width on those frames: median **9.25 px**, range 5.18–14.05.

`baseline_local` emits a ~24 px box whatever the target is — it inherited the label inflation
exactly, across only 15 distinct widths. Against a 9.25 px truth a concentric 23.90 px
prediction caps at IoU **0.150**, so **no** detection can reach the 0.5 threshold and COCO AP
is arithmetically 0.000 however good the detector actually is. It is not a low score; it is an
unreachable one.

`trueextent_local` predicts a median 8.80 px against a 9.25 px truth, over 58 distinct widths.
Size regression is recovered, and the IoU ceiling with it.

This is the same 0.11 ceiling `verified-measurements-2026-08.md` §3 measured from committed
detections, arrived at independently from the model's own predictions.

## 3. What this unblocks, and what it does not

**Unblocked.** COCO AP on this repo's own data is no longer structurally zero: **0.3912 ±
0.1140**. Every IoU-based comparison to published work was previously impossible *before any
model was trained*; it is now possible. This was the precondition in MISSION §3.3.

**Not established.** Nothing here is a SOTA comparison, and this number must not be placed
beside a published one:

* **n = 1 video.** 07_05 only, one drone, one flight. The val split is a temporal holdout
  *within* that flight, not an independent recording.
* **No birds in val.** All 934 bird instances are in frames 2–304; the val split starts at 342.
  The bird class contributes to training and to nothing in this measurement.
* **Seed variance is large** — sd 0.1140 on COCO AP, driven by seed 0, which early-stopped at
  epoch 38 (COCO 0.2599) against seeds 1 and 2 (0.4546, 0.4422). A single-seed number here
  would have been misleading in either direction. Three seeds was not a formality.
* **Centre-distance AP is not reported here.** Ultralytics scores IoU only. The protocol these
  configs declare is `specklock-centre`, and that column needs `dronedet/metrics.py` over
  exported detections — pending.

## 4. Two defects found while running it

Both were caught by guards already in the repo, before the GPU was booked.

1. **`--split-at 548` produces no validation set.** It is documented as "all labeled frames",
   and 07_05's labels end at 547, so every frame went to train and val was empty. Rule 8 wants
   the operating point chosen on val, and `trueextent_local`'s own note says "score the val
   split first". Rebuilt at `--split-at 342` → 962 train / 206 val, identical in both arms so
   the comparison stays paired.

2. **Both configs declared one class; their data has two.** `make_datasets_v3.py` pastes a bird
   bank as an explicit class 1 (2,614 drone vs 3,144 bird boxes in train). `tools/train.py`'s
   data check aborted before the first batch. A 1-class head on 2-class labels does not crash —
   it trains, converges, and reports a number.
