# Methods — every algorithm, its pipeline, its models, its results

This is the deep reference: what each detection method **is**, the complete pipeline it runs, which trained **models** live inside it, and how it **scored**. If you just want to run something, see [run-inference.md](run-inference.md); for the narrative build story, see the [reports](../reports/).

Two things are true of every method here:

- **Two coordinate frames.** Everything works in *original* frame pixels (all stored detections/GT/tracks) and *stabilized* reference-frame coordinates (internal to motion detection & tracking). Per-frame global shifts are estimated once and cached in each detection JSON's `meta.shifts` so no stage recomputes them.
- **Center-distance evaluation, never IoU.** A 1 px shift on a 6 px box swings IoU wildly, so matching uses center distance `max(τ, 0.5·√area)` with τ = 12 px, reported as size-binned AP / best-F1 / per-object recall / false-alarms-per-frame. See [`dronedet/evaluate.py`](../../dronedet/evaluate.py).

---

## 1. The shared building blocks

Most methods are assembled from these. Think of them as the parts bin.

| block | file | what it does |
|---|---|---|
| **Stabilizer** | [`dronedet/stabilize.py`](../../dronedet/stabilize.py) | estimates the per-frame global camera transform (phase-correlation vs frame 0; affine LK+RANSAC variant available) so downstream stages see a camera-motion-free world. The edge variant ([`realtime/rt_stabilize.py`](../../realtime/rt_stabilize.py)) correlates a full-res 768×448 central crop — 3.6 ms and *more* accurate than the PC's 16.5 ms full-frame pass, because it ignores the moving-foliage borders. |
| **Motion detector** | [`dronedet/motion.py`](../../dronedet/motion.py) | temporal-median background + per-pixel MAD noise (a pixel is foreground at `k_sigma`×its own historical noise) + a **flicker map** (an EMA of past foreground rate raises the threshold where motion recurs — kills wind-blown foliage, which oscillates in place, while a transiting drone passes clean). A **lagged** background (≥ 90 frames old) is what lets it see a drone drifting at 0.5 px/frame instead of absorbing it into its own model. |
| **Temporal verifier** | trained YOLOv8-P2, 2-class | an ordinary detector with a stride-4 (P2) head reading the 3-moment stabilized stack (t−12, t−6, t as R/G/B). Classifies each candidate **drone / bird**. This is the component that carries the signal — see [the core idea](../../README.md#the-core-idea-in-one-picture). |
| **Full-frame expert** | trained YOLOv8, 1-class big | catches the drone that is *not moving* (landed / hovering), which motion is blind to. Held at 100% recall on the landed drone (a single 526-frame track at 0.57 px error). |
| **Tracker** | [`dronedet/track.py`](../../dronedet/track.py) | Kalman constant-velocity tracker in stabilized coords; Hungarian assignment on center distance; coasting through fades; **strict local re-acquisition** (unique-peak test + use cap — a loose version latches onto foliage and makes clutter tracks immortal); post-filters for **directedness** (flying objects sustain a direction; foliage jitter is zero-mean) and duplicate-track merging. |
| **Track classifier** | [`dronedet/trackclass.py`](../../dronedet/trackclass.py) | aggregates the verifier's per-detection confidence over a whole track and announces *drone* once it accumulates 8 verifier-confirmed detections. The aggregated `conf_frac` separates drone from clutter essentially perfectly (drone tracks ≈ 0.94–1.00, clutter far below). |

**Why each block earned its place** — every one replaced a simpler version that *measurably* failed:

| stage | without it |
|---|---|
| stabilization | sub-pixel camera drift makes every high-contrast edge flicker; all motion logic downstream breaks |
| **lagged** background | a 0.5 px/frame drifter is absorbed into its own background and vanishes — recall on the hard segment was 0.05 before, 0.57 after |
| flicker map + MAD noise | wind-blown foliage produced ~50 false candidates/frame; its own history raises its threshold while a transiting drone passes clean |
| MOG2 union | the lagged channel alone misses ~30% of the hardest frames; MOG2 sees 86% at hopeless precision — the union feeds the verifier maximal recall and lets it supply precision |
| **temporal verifier** | single-frame ablation: AP 0.22 vs 0.66 with everything else identical |
| full-frame expert | motion can't see a landed/hovering drone; the expert holds it at 100% recall |
| 2-class (bird) training | birds are the dominant false-alarm source; classifying them *in the verifier* suppresses them before tracking |
| coasting + strict re-acquisition | the drone fades for up to ~45 frames over low-contrast patches; coasting bridges, re-acquisition re-locks — the strict unique-peak test stops it latching onto foliage |
| kinematic track filter | gust-triggered tracks survive confirmation; directedness kills them, while an appearance exemption protects the stationary landed drone |

---

## 2. The methods, from baseline to flagship

Every name below is a real entry in the [`build_method`](../../dronedet/methods/__init__.py) registry (or a `realtime` pipeline). Run any of them with:

```bash
python -m dronedet detect --video data/videos/07_05.mp4 --method <name> --out work/det/<name>.json
```

### 2a. Pretrained YOLO baselines — `yolo-640`, `yolo-1280`, `yolo-sahi`

- **Pipeline:** a single off-the-shelf COCO YOLO over the raw frame, at 640, at 1280, or SAHI-tiled; drones read out of the airplane/bird/kite classes at very low confidence.
- **Models:** pretrained `yolov8s.pt` / `yolo11s.pt` (auto-downloaded).
- **Result:** AP ≈ **0.000** on the tiny target at every resolution and with tiling. By the time a 6 px object passes through the backbone it is gone. This is the control that motivates everything else.

### 2b. Classical motion only — `motion-median`, `motion-mog2`, `motion-slow`

- **Pipeline:** stabilize → background model (temporal median / MOG2 / lagged-slow) → flicker-map + MAD threshold → connected components. No neural network.
- **Models:** none.
- **Result:** high recall, hopeless precision (rt-a, the edge packaging of this, reaches R 0.50 / P 0.27). Excellent as a **proposal generator** or a wake-up pre-filter; unusable as a product on its own.

### 2c. Fine-tuned YOLO — `yolo-ft`, `yolo-ft-hybrid`, `yolo-ft-sahi`

- **Pipeline:** the same shapes as the baselines/hybrid but with a *fine-tuned* single-class detector (`drone_classes=None`).
- **Models:** `work/models/yolo-ft*-best.pt`.
- **Result:** best-of-five recipes still only **AP 0.023**. The [round-1 report](../reports/round1-pipeline.md) explains why: fine-tuning on frames that contain *both* the 180 px landed drone and the <10 px flying drone yields **zero** recall on the tiny target — IoU-based label assignment scales the tiny GT's loss by its (near-zero) predicted IoU while the big object dominates the gradient. The fixes (tiny specialist, erased big object, inflated fixed-size labels) are what the verifier uses.

### 2d. Hybrid, round 1 — `hybrid`, `yolo-ft-hybrid`

- **Pipeline:** motion proposals → zoomed/native crops → YOLO verification, **unioned** with a full-frame YOLO pass (to catch the static/landed drone motion can't see). Supports mixture-of-experts: a second model handles the full-frame pass.
- **Models:** verifier + `work/models/yolo-ft-best.pt` (ft1 expert) + `yolo-ft5-best.pt` (tiny specialist).
- **Result:** the round-1 champion (`moe-hybrid`, single-frame verifier): full-video AP(far) 0.320 / AP(all) 0.670. Good, but the single-frame verifier is the ceiling — which round 2 broke.

### 2e. Temporal mixture-of-experts, round 2 — `moe2-hybrid`, `moe3-stacked`  ★ flagship (round 2)

- **Pipeline:** [`dronedet/methods/hybrid2.py`](../../dronedet/methods/hybrid2.py). Dual motion proposals (lagged + MOG2) → **temporal verifier** on each candidate's 3-moment motion trail → unioned with a full-frame expert → fused → tracker → tracks re-projected to per-frame detections (`tracked-moe3`).
- **Models:** `work/models/yolo-ft7-best.pt` (2-class **temporal** verifier) + `yolo-ft-best.pt` (full-frame expert). `moe2` uses the single-frame verifier `yolo-ft6-best.pt` as the ablation.
- **Result (07_05 val, hand labels):**

  | variant | AP | best F1 | Recall | Precision | FP/frame |
  |---|---|---|---|---|---|
  | **`tracked-moe3`** (verifier + track integration) | **0.960** | **0.938** | 1.000 | 0.884 | 0.13 |
  | `moe3-stacked` (per-frame, temporal verifier) | 0.656 | 0.760 | 0.767 | 0.752 | 0.25 |
  | same pipeline, **single-frame** verifier (ablation) | 0.222 | 0.316 | 0.558 | 0.220 | 1.98 |
  | plain fine-tuned YOLO (best of 5) | 0.023 | 0.060 | 0.049 | — | 0.57 |
  | pretrained YOLO (any of 640/1280/SAHI) | ≈ 0.000 | ≤ 0.001 | ≤ 0.005 | — | — |

  The temporal-vs-single-frame row (0.66 vs 0.22, everything else identical) is the whole thesis, at the pipeline level.

### 2f. PC-MAX, round 3 — the shipped desktop model  ★★ best

- **Pipeline:** the round-2 flagship, evolved. A **third** detection stream — a full-frame temporal YOLOv8s-P2 @1280 — is fused with the verifier and expert streams by center-agreement noisy-OR, then the tracker runs and the **track classifier** makes the drone/not-drone decision.
- **Models:** `final/pc_max/verifier640.pt` (temporal verifier) + `final/pc_max/expert1280.pt` (round-1 ft1 expert) + `final/pc_max/fullS.pt` (full-frame temporal s-model).
- **Result:** tracked **AP/F1/R/P = 1.000, zero false positives** on the unseen test (per-frame AP 0.85–0.91); drone confirmed 7 frames after track birth; also reports the landed drone as a separate `near` track. ~4 fps on an RTX 5070 laptop.
- **Run it:** `python final/run_final.py --profile pc-max --video …`. Details: [final/README.md](../../final/README.md).

### 2g. EDGE-RT, round 3 — the shipped real-time model  ★★ fastest

- **Pipeline:** one YOLOv8-nano-P2 reads the stabilized 3-moment stack **full-frame** — no proposals, no crops, no expert — under TensorRT FP16, with the same tracker and track classifier behind it.
- **Models:** `final/edge_rt/edge_n1280.pt` (+ a TRT FP16 `.engine` built for the host machine — rebuild on the target device).
- **Result:** tracked **AP/F1/R/P = 1.000, zero false positives** (per-frame AP 0.88); **~74–85 fps** end-to-end (9.5 ms/frame). Projected 10–15 fps FP16 on a Jetson Orin Nano @1280.
- **Run it:** `python final/run_final.py --profile edge-rt --video …`.

### 2h. The six edge pipelines — `rt-a` … `rt-f`

The [`realtime/`](../../realtime/README.md) package compares six edge architectures on the pure test set. EDGE-RT is `rt-c` productized.

| pipeline | what it is | AP | F1 | R | P | fps | ms |
|---|---|---|---|---|---|---|---|
| **rt-c-full1280** | temporal nano, full frame @1280 TRT | **0.894** | **0.894** | 0.844 | 0.950 | 78 | 10.4 |
| rt-d-full640 | same @640 (2× downscale) | 0.702 | 0.774 | 0.652 | 0.953 | **104** | 7.4 |
| rt-b-verify256 | motion proposals → temporal nano on 256 px crops | 0.519 | 0.700 | 0.676 | 0.725 | 24 | 39.2 |
| rt-e-decimated | rt-b, verify every 2nd frame | 0.356 | 0.456 | 0.332 | 0.728 | 29 | 33.2 |
| rt-f-single1280 | **single-frame** nano @1280 (temporal ablation) | 0.188 | 0.274 | 0.220 | 0.362 | 70 | 12.6 |
| rt-a-classic | classical only, no NN | 0.180 | 0.355 | 0.504 | 0.274 | 36 | 26.0 |

Key finding: the **proposal→verify architecture that wins on PC (rt-b) loses on edge** — with a nano verifier the crops no longer dominate cost, so one batched full-frame 1280 pass is both faster *and* more accurate. Architecture choices don't transfer across compute classes; measure, don't assume.

---

## 3. <a id="weights"></a>Weights — which file is what, and where

All trained weights are kept in git (the pretrained base weights auto-download and are gitignored).

| file | what it is | used by |
|---|---|---|
| `final/pc_max/verifier640.pt` | 2-class temporal verifier @640 (`final-ft7-s640`) | PC-MAX |
| `final/pc_max/expert1280.pt` | full-frame landed/big-drone expert (round-1 ft1) | PC-MAX |
| `final/pc_max/fullS.pt` | full-frame temporal YOLOv8s-P2 @1280 (`final-ftS-s1280`) | PC-MAX |
| `final/edge_rt/edge_n1280.pt` (+`.engine`,`.onnx`) | single temporal YOLOv8n-P2 @1280 (`final-ftC-n1280-e25`) | EDGE-RT |
| `work/models/yolo-ft-best.pt` | ft1 — near/big full-frame expert (round 1) | hybrid, moe, PC-MAX |
| `work/models/yolo-ft5-best.pt` | tiny specialist (round 1) — the paste-only recipe | hybrid |
| `work/models/yolo-ft6-best.pt` | 2-class **single-frame** verifier (round-2 ablation) | moe2 |
| `work/models/yolo-ft7-best.pt` | 2-class **temporal** verifier (round-2 flagship) | moe3-stacked |
| `work/models/yolo-ft7v3-best.pt`, `yolo-ftSv3-best.pt` | v3 verifier / s-model (round 3) | round-3 builds |
| `work/models/yolo-ft2/ft3-best.pt` | intermediate round-1 fine-tunes | ablations |
| `work/models/FSRCNN_x4.pb` | super-resolution ablation (negative result — no gain over bicubic) | SR ablation |
| `realtime/work/models/verifier_n256.pt`, `v3_verifier_n256.pt` | nano 256 px crop verifiers | rt-b/e |
| `realtime/work/models/full_temporal_n1280/n640.pt`, `v3_*` | full-frame temporal nano nets | rt-c/d |
| `realtime/work/models/full_single_n1280.pt` | single-frame nano (temporal ablation) | rt-f |
| `baseline/yolo26n-new-data_full__2026_Jan_19.pt` | external baseline — YOLO26n trained on a real multi-scene drone dataset | baseline comparison |
| `yolov8n.pt`, `yolov8s.pt`, `yolo11n.pt`, `yolo11s.pt`, `yolo26n.pt` (repo root) | pretrained base weights (auto-downloaded, **gitignored**) | training / baselines |

`.engine` (TensorRT) and `.onnx` files are **architecture-specific and regenerable** — they are gitignored and rebuilt by [`realtime/tools/export_models.py`](../../realtime/tools/export_models.py) / rebuilt on the target device.

---

## 4. Negative results worth knowing (measured, not assumed)

- **Super-resolution on crops** (FSRCNN ×4 vs bicubic): no meaningful gain.
- **Plain fine-tuning** at any resolution/slicing *without* the tiny-object recipe: AP 0.023.
- **Semi-automatic ground truth**: our first auto-GT confidently tracked a *bird* for 234 frames; the manual labels ([`work/gt_user.json`](../../work/gt_user.json)) corrected it — which is why the hand labels are authoritative.
- **INT8** on the desktop GPU: net-negative (slower *and* less accurate than FP16) — recalibrate on the target device before trusting it.
- **DT=9 temporal stacks** and a **full-frame** edge stabilizer: both tried and rejected (see [round-3 report §5](../reports/round3-deliverables.md)).
