# HiveLab — Tiny-Drone Detection & Tracking

Detecting and tracking a drone that occupies **3–14 pixels** in 720p RGB video — including long stretches where it is so low-contrast against field clutter that a *human* can only find it by flipping between frames.

<p align="center">
  <img src="docs/media/hivelab_track_10_06.gif" width="820" alt="HiveLab pipeline tracking a tiny drone on an unseen video"/>
  <br/>
  <em>The full pipeline on a video it was never trained on (<code>10_06.mp4</code>): one continuous track (id1) from the low-contrast field crossing to the sky climb — flight trail, live zoom inset, no false alarms.</em>
</p>

## Headline results

**Held-out evaluation** (hand-labeled ground truth; the val segment is a 4–10 px drone drifting over bushes at ~0.5 px/frame — the hardest regime):

| method | AP | best F1 | Recall | Precision | FP/frame |
|---|---|---|---|---|---|
| **HiveLab: pipeline + track integration** (`tracked-moe3`) | **0.960** | **0.938** | **1.000** | 0.884 | 0.13 |
| HiveLab: per-frame (`moe3-stacked`) | 0.656 | 0.760 | 0.767 | 0.752 | 0.25 |
| same pipeline, single-frame verifier (ablation) | 0.222 | 0.316 | 0.558 | 0.220 | 1.98 |
| plain fine-tuned YOLO (best of 5 recipes) | 0.023 | 0.060 | 0.049 | — | 0.57 |
| pretrained YOLO — every variant (640/1280/SAHI) | ≈ 0.000 | ≤ 0.001 | ≤ 0.005 | — | — |

**Tracking**: the drone's entire 548-frame flight — descent, bush-skim, fast dash, slow drift — is covered as **one track: 97.1% coverage, zero ID switches, 1.0 px median error**.

**Against a strong conventional baseline** on the unseen video — YOLO26n trained on a real multi-scene drone dataset (imgsz 1760, 300 epochs):

<p align="center">
  <img src="docs/media/baseline_vs_hivelab.gif" width="960" alt="Baseline YOLO26n vs HiveLab pipeline side by side"/>
</p>

| | flight coverage | where it works |
|---|---|---|
| Baseline YOLO26n (single frame) | **12.5%** | only the final second, drone against open sky (excellent there: conf 0.6–0.84, zero FPs) |
| HiveLab pipeline | **continuous track** | the whole flight, including 300 frames of ground clutter where the baseline outputs nothing even at conf 0.02 |

That gap **is** the thesis: single-frame appearance handles sky silhouettes; everything below the treeline requires motion.

Videos: [pipeline on unseen video](docs/media/10_06_tracks_confirmed.mp4) · [baseline on the same video](docs/media/10_06_baseline_dets.mp4) · [training video with hand labels painted](docs/media/07_05_round2_tracks.mp4)

---

## The method

### The core idea in one picture

A drone crossing a field at 4–10 px is invisible in a single frame — to a detector *and* to a human. But stack three **stabilized** grayscale frames (t−12, t−6, t) as the color channels of one image, and everything static cancels to gray while anything that moved appears as colored blips — one blip per moment in time:

<p align="center">
  <img src="docs/media/temporal_input.jpg" width="900" alt="single frame vs stacked temporal input"/>
  <br/>
  <em>Same spot, same moment, unseen video. Left: find the drone (you can't — and neither can any single-frame detector, at any confidence). Right: the verifier's input — <b>yellow</b> = where the drone was 12 frames ago, <b>magenta</b> = 6 frames ago, <b>cyan</b> (circled) = now. The trail even shows its direction of flight.</em>
</p>

The verifier is an ordinary YOLOv8s with a stride-4 (P2) head — no architecture changes — trained on this 3-channel temporal representation with two classes (drone / bird). On identical held-out instances, identical training recipe: **single-frame input mAP50 = 0.06, temporal input mAP50 = 0.83**. The representation, not the network, is the breakthrough. (It formalizes exactly how our human labeler found the drone: flip frames, watch what moves.)

### The full pipeline, frame by frame

Every frame `t` (1280×720 BGR) goes through six stages:

<p align="center">
  <img src="docs/media/architecture.svg" width="1000" alt="HiveLab pipeline architecture"/>
</p>

Reading the figure: the **top lane** finds and identifies movers — stabilization gives every later stage a camera-motion-free world; two complementary background models propose everything that moves (the lagged one catches drifters as slow as half a pixel per frame, MOG2 maximizes recall); each candidate is judged by the temporal verifier on its motion trail. The **bottom lane** completes the picture — a separate full-frame expert catches drones that do not move (hovering, landed), fusion produces one scored detection list per frame, and the tracker integrates over time: it carries the target through fades, re-locks with a strict local search, and rejects clutter by kinematics. The output is a decision, not a box soup: **drone alarms**, **unconfirmed flying objects** (birds), and silence.

Why each stage exists — every one replaced a simpler version that **measurably failed**:

| stage | without it |
|---|---|
| stabilization | sub-pixel camera drift makes every high-contrast edge flicker; all motion logic downstream breaks |
| **lagged** background (≥ 90 frames old) | a drone drifting at 0.5 px/frame is *absorbed into its own background model* and vanishes — recall on the hard segment was 0.05 before, 0.57 after |
| flicker map + MAD noise | wind-blown foliage produced ~50 false candidates/frame; foliage oscillates in place so its own history raises its threshold, while a transiting drone passes clean |
| MOG2 union | the lagged channel alone misses ~30% of the hardest frames; MOG2 sees 86% at hopeless precision — the union feeds the verifier maximal recall and lets it supply the precision |
| **temporal verifier** | the single-frame ablation: AP 0.22 vs 0.66 with everything else identical |
| full-frame expert | motion is blind to a landed/hovering drone; the expert holds it at 100% recall (a single 526-frame track at 0.57 px error) |
| 2-class (bird) training | birds are the dominant false-alarm source; classifying them *in the verifier* suppresses them before tracking |
| coasting + strict re-acquisition | the drone fades for up to ~45 frames crossing low-contrast patches; coasting bridges, re-acquisition re-locks — a *loose* version latched onto foliage and made clutter tracks immortal, hence the unique-peak test and use cap |
| kinematic track filter | gust-triggered tracks survive confirmation; directedness kills them (foliage nets ~0 displacement), while the appearance exemption protects the stationary landed drone |

The result of stage 5 is also reusable as a detector: converting tracks *back* into per-frame detections (`tracked-moe3`) fills detection gaps with track persistence — that is the F1 0.94 / recall 1.0 row in the headline table.

**Training the temporal verifier** has its own recipe, each element earned in [the reports](REPORT2.md): tiny labels inflated to fixed 24 px boxes (IoU-based label assignment starves true-size sub-10 px boxes to exactly zero recall); copy-paste augmentation where patches are pasted **per-channel along a simulated velocity** (rigid for drones — including hover; faster with per-channel size jitter for birds — wing flap); atmospheric-haze jitter (distant targets fade toward the background); and **never mixing large and tiny instances in one training set** — a 180 px object's alignment scores dilute the tiny targets' gradient to nothing (the landed drone is erased from the verifier's backgrounds and handled by the separate expert).

---

## What's in the project

Two full experiment rounds, 19 evaluated method variants, 7 trained models, a labeling tool, and an evaluation harness built for few-pixel targets (center-distance matching — IoU is meaningless at 4 px).

### Reports

- **[REPORT2.md](REPORT2.md)** — the definitive results: evaluation against hand labels, the temporal-vs-single-frame ablation, the slow-mover fix, SR ablation, bird handling.
- **[REPORT.md](REPORT.md)** — round 1: building the pipeline, the resolution/SAHI/motion comparisons, and the tiny-object training failure analysis (why plain fine-tuning gets 0.000 recall on <10 px targets and what fixes it).

Notable negative results (measured, not assumed): image super-resolution on crops (FSRCNN ×4 vs bicubic: no meaningful gain), plain fine-tuning at any resolution/slicing without the tiny-object recipe, and semi-automatic ground truth (our auto-GT confidently tracked a *bird* for 234 frames — the manual labels corrected it).

### Repository layout

```
dronedet/               the pipeline library
  stabilize.py            global camera-motion estimation
  motion.py               background-model motion detector (lag, MAD noise, flicker map)
  methods/                all detection methods behind one interface
    hybrid2.py              the best pipeline (dual proposals + temporal verifier + expert)
    hybrid.py, yolo.py, motion_only.py, ...
  track.py                Kalman tracker + re-acquisition + track filters
  evaluate.py             center-distance evaluation (AP / F1 / per-object recall)
  render.py, viz.py       annotated video rendering
  cli.py                  python -m dronedet {detect,eval,track,render}
tools/
  run_best.py             one-command inference on any video  <-- start here
  run_baseline.py         same visualization for any plain detector
  label.py                browser labeling UI (wheel-zoom, per-frame autosave)
  build_gt_user.py        manual labels -> canonical GT + agreement analysis
  make_dataset_ft6/ft7.py training-set builders (single-frame / temporal)
  train_yolo.py           training recipe (P2 head, tiny-object settings)
  final_round2.py         reproduce the full round-2 comparison
work/
  gt_user.json            ground truth (from manual labels) - authoritative
  models/                 all trained weights (ft1 expert, ft7 temporal verifier, ...)
  eval_user_*.md          method-comparison tables
  det*/ tracks*/ infer/   per-method detections, tracks, inference outputs
baseline/                 external baseline weights (YOLO26n)
```

### Run inference on any video

```bash
pip install -r requirements.txt   # torch needs the cu128 index on Blackwell GPUs
python tools/run_best.py --video your_video.mp4
```

Outputs in `work/infer/<video>/`: `tracks_confirmed.mp4` (drone alarms only), `tracks_all.mp4` (every flying object), `dets.mp4` (raw detections), JSONs, and a per-track summary that attributes each track to *drone* or *unconfirmed flying object*. Track labels: solid `id1` = detected this frame; `id1?` (yellow) = coasted/re-acquired estimate.

Baseline comparison for any plain detector:

```bash
python tools/run_baseline.py --weights baseline/yolo26n-new-data_full__2026_Jan_19.pt --video your_video.mp4
```

### Evaluate / retrain / relabel

```bash
# score any detection JSONs against the GT (val split = frames 342+)
python -m dronedet eval --gt work/gt_user.json --dets work/det2/*.json --frames 342:571

# label a new video (browser UI), then canonicalize
python tools/label.py                      # -> work/labels.json
python tools/build_gt_user.py              # -> work/gt_user.json + agreement stats

# retrain the temporal verifier and reproduce the full comparison
python tools/make_dataset_ft7.py
python tools/train_yolo.py --data work/dataset_ft7/data.yaml --imgsz 640 --batch 16 --name ft7-p2-640-stacked
python tools/final_round2.py
```

### Honest limitations & next steps

- Trained/validated on footage from one location (train/val split is by time, and the pipeline held up on a different-day unseen video — but new scenes, drone types, and strong camera motion need the planned heavy fine-tune).
- At 4 px, birds and drones genuinely converge: two bird tracks still pick up occasional drone confirmations. Next step: a track-level classifier on kinematics + appearance across the whole track (flap signature).
- Production fine-tune checklist (evidence-based, see reports): temporal input channels first, NWD/RFLA assignment for tiny boxes, copy-paste with scale+haze+trajectory jitter, scale-separated experts, and center-distance evaluation — never bare mAP@0.5.
