# Round 7 — Learning from the NPS SOTA: one RGB+motion model for *all* datasets

**Goal (user's framing):** not to *beat* the NPS leaders (Dogfight, TransVisDrone) on their home
turf, but to **study their architecture / training / preprocessing and fold what helps into our
one generalist model** — so a *single* model is strong on *every* dataset (ARD-MAV, NPS, and the
user's black drone), measured the way that matters here: **fast acquisition + never losing the
track** (center-distance recall / coverage, not tighter boxes).

## What the SOTA actually taught us (and what to ignore)

Reading TransVisDrone's own ablation was decisive: a **plain single-frame YOLOv5-l already scores
0.93** on NPS; their VideoSwin spatio-temporal transformer adds only **+0.02** (→0.95), and a
follow-up reports it **collapses to ~0.15 on the harder ARD100** tiny set. So the headline
architecture is an NPS-specialist tool — **the wrong thing to copy for a generalist.** The ideas
that *generalize* are:

1. **Learned motion-fusion (YOLOMG-style)** — fuse an ego-motion-compensated motion-difference
   map with RGB *inside* the network. This is the actual SOTA-tying idea and the biggest lever on
   the *hardest* tiny targets (ARD100 0.33 → 0.78) — exactly our black-drone weak spot.
2. **NWD (Normalized Wasserstein Distance)** — a tiny-object-aware assignment/loss that stops the
   task-aligned assigner from starving 3–14 px GTs of positive anchors.

We built both, and *skipped* the video transformer and per-dataset specialization (both fight the
generalization goal).

## What we built

- **NWD** (`dronedet/nwd.py`) — blends NWD into **both** the task-aligned assigner (the big lever)
  and the box regression, as a non-invasive patch (`--nwd` on `tools/train_yolo.py`).
- **RGB+motion 4-channel fusion** — the generalist version of "motion in the input", keeping colour
  (our old temporal expert stacked 3 greys and threw colour away):
  - `tools/make_fusion_combined.py` — builds `[B, G, R, motion]` `.npy` tiles; channel 4 is the
    ego-compensated 3-frame difference (`min` over t-dt, t-2dt after grid-LK+RANSAC registration).
  - `dronedet/mc_data.py` — small shim so ultralytics trains on >3-channel `.npy` (its intended
    multi-channel path, only missing the format/verify plumbing).
  - `configs/yolov8{s,m}-p2-ch4.yaml` — P2 detector with a 4-channel first conv.
  - `dronedet/methods/fusion.py` — inference: rebuilds the 4-channel frame per timestep and runs a
    manual forward (the high-level predictor can't drive 4 channels).
  - Trained end-to-end **with NWD**, on all datasets combined (ARD-MAV + NPS + user 07_05).

## Results — one model, every dataset (held-out clips, τ=12 px)

Each column is a **single generalist model** trained on all datasets. "fusion(s/m)" = the 4-channel
RGB+motion model at nano / medium scale.

**Identification — per-frame detection AP / recall:**

| clip | baseline (m) | NWD (m) | fusion (s) | **fusion_m** |
|---|---|---|---|---|
| phantom16 (ARD-MAV) | 0.992 / 0.951 | 0.991 / 0.971 | 0.988 / 0.967 | **0.994 / 0.995** |
| Clip_19 (NPS) | 0.592 / 0.749 | 0.685 / 0.851 | 0.616 / 0.841 | **0.801 / 0.848** |
| 10_06 (black) | 0.266 / 0.243 | 0.226 / 0.199 | **0.791 / 0.742** | 0.685 / 0.644 |

**Tracking — coverage / false tracks:**

| clip | baseline | NWD | fusion (s) | **fusion_m** |
|---|---|---|---|---|
| phantom16 | 0.977 / 5 | 0.973 / 9 | 0.988 / 11 | 0.971 / 10 |
| Clip_19 (NPS) | 0.835 / 1 | 0.994 / 3 | 0.957 / 4 | **0.990 / 3** |
| 10_06 (black) | 0.659 / 0 | 0.662 / 0 | 0.875 / 0 | 0.875 / 0 |

### Reading the table (honestly)
- **fusion_m is the best *single-model* generalist detector.** With *one* model and no regime logic
  it leads identification on ARD-MAV (0.994/0.995) and **NPS (det 0.801, cov 0.990)** — the NPS
  number that was our soft spot, now the best of any model — while lifting **black-drone detection
  0.27 → 0.69** (0.79 at nano scale). NWD and motion-fusion are complementary and both real: NWD
  alone lifts NPS coverage 0.835 → 0.994; motion-fusion alone lifts black-drone detection 0.27 →
  0.79; fusion carries both (it is trained with NWD).
- **Where it does *not* win — near-static black-drone *tracked* coverage.** The already-shipped
  round-6 regime pipeline tracks the black drone to **1.000**, and `run_max --profile fusion` gives
  **0.875** on the same clip. Classical ego-motion differencing (mc-hybrid) is near-perfect-recall
  on a *near-static, colour-poor* target, and the tracker coasts the rest — a single learned
  detector doesn't beat that on this specific case, even though it detects the drone far better
  per-frame. So fusion is a big win for **generalization, NPS, and one-model simplicity**, but it
  **does not replace** the classical-motion path for the user's own near-static black drone.
- **s-fusion is the edge model** — same architecture at nano scale, **faster** (10.5–17 fps vs
  baseline's 7–10), and it edges fusion_m on the black drone (0.79 vs 0.69 detection).
- **False tracks on moving cameras — fusion's real cost.** The motion channel fires on parallax
  over textured ground. On phantom16 `run_max`'s kinematic directedness + track classification cut
  false tracks **10 → 5** (coverage 0.965) — better than raw, but *not* the **0** the round-6 regime
  pipeline achieved on ARD-MAV by gating motion out on moving cameras. On the near-static black
  drone fusion is clean (**0 false, one track**). So fusion trades some moving-camera precision for
  one-model simplicity + far better black-drone detection + best-in-class NPS. Drone-vs-bird at a
  few pixels is the unchanged honest frontier.

## Shipping

The fusion detector is regime-agnostic (motion is fused inside the net) and runs as one command:

```bash
# PC generalist flagship (m-scale: best single-model detector, best on ARD-MAV + NPS)
python tools/run_max.py --profile fusion \
    --weights work/runs/combined-fusion-m-p2-2/weights/best.pt --video x.mp4 --out out --gt gt.json
# Edge (s-scale: faster, best black-drone detection)
python tools/run_max.py --profile fusion \
    --weights work/runs/combined-fusion-s-p2-2/weights/best.pt --video x.mp4 --out out
```

**Bottom line.** The **RGB+motion fusion architecture + NWD** is the new **best single generalist
model** — one model, no regime split, strongest per-frame detector across datasets and best on the
NPS case that was our weak spot. For the user's **own near-static black drone**, the round-6
regime pipeline (classical motion + tracker, cov 1.000) still tracks marginally better, so it stays
the recommendation there. The natural next step — fusing the learned RGB+motion detector *with* the
classical-motion branch on near-static cameras — would combine both wins into one pipeline.

## Reproduce
```bash
python tools/make_fusion_combined.py                       # 4-channel RGB+motion tiles
python tools/train_yolo.py --data work/ext_datasets/combined_fusion/data.yaml \
    --model configs/yolov8m-p2-ch4.yaml --weights yolov8m.pt --imgsz 640 --batch 8 \
    --hsv 0 0 0 --scale 0.4 --mosaic 0.3 --name combined-fusion-m-p2 --nwd --mc
python tools/eval_improvements.py --detectors baseline nwd fusion fusion_m   # the tables above
```
