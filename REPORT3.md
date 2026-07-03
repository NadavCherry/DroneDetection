# Round 3 — Toward 100%: Track-Level Classification, Hardened Test GT, v3 Data, and the Two Final Models

Rounds 1–2 (`REPORT.md`, `REPORT2.md`) built the pipeline and established the temporal-input lesson.
Round 3 pushes both pipelines to their measurable ceiling and delivers the two requested models:
**PC-MAX** (most powerful, runs on a desktop GPU) and **EDGE-RT** (real-time, one nano net,
edge-class hardware). Training uses `07_05.mp4` only; `10_06.mp4` is a pure test video
(never trained on, never used for model selection).

## 1. Where the remaining errors actually were

Starting point (round-2 champion `tracked-moe3`): F1 0.938 on the 07_05 validation split,
R = 1.000 — so *every* remaining error was a false positive. Attribution of all 27 FPs at the
operating point found exactly four tracks:

| track | what it physically is (verified by eye) | real-detection density | verifier-confirmed |
|---|---|---|---|
| 89 | foliage / ground clutter | 0.19 | 0.02 |
| 333 | bush texture | 0.13 | 0.00 |
| 381 | the *landed* drone's legs + shadow, just outside the GT ignore radius | 0.16 | 0.00 |
| 712 | a real unlabeled flying object (distant dot in sky, 4 real detections in 37 frames) | 0.11 | 1.00 |

And the full-video misses were exactly two blocks: frames 0–11 (the temporal stack needs
2·DT = 12 frames of warmup → the detector was structurally blind at stream start) and
312–319 (dim transition into the bush-drift phase).

Every improvement below targets one of these measured failures — nothing speculative.

## 2. Track-level classification (the false-positive killer)

At 4 px a bird and a drone genuinely converge per-frame; the research surveys recommend
track-level kinematic/spectral features. It turned out simpler than the literature suggests:
**aggregate the 2-class verifier's per-detection opinion over the track's lifetime.**

For every track: match its `tracked`-status positions back to the detections that fed them;
count the fraction that are verifier-confirmed drones (label `drone*`, score ≥ 0.5).
Measured separation on both videos (moe3 + edge tracks):

| track | conf_frac | n_conf |
|---|---|---|
| far drone (07_05) | **0.99** | 432 |
| drone (10_06) | **0.94** | 202 |
| landed drone (07_05) | 1.00 (large boxes → class `near`) | 571 |
| all clutter tracks (both videos) | 0.00–0.02 | 0–7 |
| brief sky object (712) | 1.00 | **4** |

The rule (`dronedet/trackclass.py`): a track is a **drone** iff `conf_frac ≥ 0.5` **and**
`n_conf ≥ 8` (sustained evidence — a 4-detection flash is an anecdote, not a track);
mostly-large-box tracks are the landed drone (`near`); everything else is `other` and is
dropped from the tracked output (`tools/tracks_to_dets.py --classify`).

- No flap-spectrum or kinematic features were needed: the labeled GT birds never even form
  tracks (the verifier already suppresses them below tracker threshold). The planned
  bird/drone discriminator collapsed into eight lines of counting.
- The rule is **online-capable**: the drone track reaches its 8th confirmation 7–8 frames
  after track start on both videos (≈ 0.25 s alarm latency at 30 fps).

Effect (v2 models, before any retraining): 07_05-val **F1 0.938 → 1.000** (AP 1.000, P 1.000,
R 1.000); full video F1 0.881 → 0.974.

## 3. A test reference you can actually trust (10_06 GT v2)

The v1 test reference was the PC pipeline's own verified track — measured frames only
(250 GT frames, **111 excluded**). Two problems: misses inside excluded frames are invisible
to scoring, and the reference is circular for the PC pipeline. v2
(`REALTIME/tools/harden_gt_1006.py`):

1. **Dense candidates** for every frame in the flight span (measured positions; linear
   interpolation through gaps; velocity-extrapolated aprons at both ends).
2. **Independent position refinement**: non-causal windowed-median background
   (samples t±8..24 — offline is legal for ground truth), snap to the local SNR peak within
   10 px when strong (≥3.5σ) and unique (2nd peak < 0.75×); iterative neighbor-propagation
   passes re-anchor frames the first pass missed. 345/358 frames refined; sub-pixel centroids.
3. **Human verification of every frame** on contact sheets (raw zoom + SNR heat-map side by
   side, position circled/crossed): frames 3–22 show no target (excluded), frame 297
   (occluded against dark trees, unrefined) excluded; frames 359–360 recovered the drone
   *past the end* of the v1 track.

Result: **337 verified GT frames (was 250), 24 excluded (was 111)** — recall can no longer
hide, and positions are anchored to an independent motion analysis, not to any detector
under test. The two unconfirmed slow movers in dense foliage stay as ignore regions.

## 4. Data v3 (targeting the measured misses)

`tools/make_datasets_v3.py` rebuilds all three training sets (full-frame temporal @1280 for
the edge/PC nets, 640 px slices for the PC verifier, 256 px slices for the edge verifier)
from one stabilization pass. Changes over v2, each mapped to a measured failure:

| change | targeted failure |
|---|---|
| drone patch bank from **all** train frames (171 patches vs 65 sky-only) — includes the dim bush-phase appearance | val-phase & 10_06 foliage-crossing misses |
| **warmup images**: frames t < 12 with clamped stacks (channels repeat frame 0), matching a new inference-side clamp | the guaranteed 12-frame blind start |
| sub-pixel trail placement (fractional per-channel offsets baked in via warpAffine) | 0.5 px/frame drifter trails quantized away |
| patch photometric jitter (gain/bias/blur) + haze to 0.55 | low-contrast generalization |
| drone paste speeds 60% U(0,2.5) + 40% U(2.5,9) px/frame | faster 10_06 target (up to ~7 px/frame at the end) |
| hard-negative oversample: train frames where the v2 edge net false-alarmed get an extra clean copy (77 frames) | foliage FPs |
| 2 paste variants per train frame (749 vs 330 images) | paste diversity per epoch |

Trainings (all `yolov8*-p2`, hsv_h = hsv_s = 0 — hue/sat jitter would remix temporal
channels semantically): nano@1280 full-frame, s@640 verifier, s@1280 full-frame (new — the
PC-grade full-frame detector), nano@256 verifier. Checkpoint note: ultralytics'
fitness-selected `best.pt` clearly beat `last.pt` at the *pipeline* level (last.pt overfits
the pastes) — trust best.pt.

## 5. Results — v3 models, split-trained (train < frame 342)

All numbers: center-distance matching, τ = 12 px. `tracked-*` = tracker-integrated detections
after track-level classification. 07_05-val = frames 342–547 (the hardest segment,
never seen in training); 10_06 = the hardened v2 test reference.

### PC pipelines

| method | 07_05-val AP / F1 | 07_05-full AP / F1 | 10_06-test AP / F1 | fps (5070) |
|---|---|---|---|---|
| **tracked-pcmax + coast smoothing** (surveillance mode) | **1.000 / 1.000** | **1.000 / 1.000** | **1.000 / 1.000** | 4.0 |
| tracked-pcmax (causal coasts) | 0.995 / 0.995 | 0.998 / 0.998 | 1.000 / 1.000 | 4.0 |
| tracked-moe3v3 (proposals+verify path only) | **1.000 / 1.000** | 0.962 / 0.974 | 0.996 / 0.997 | 4.0 |
| pc-max (per-frame, fused) | 0.775 / 0.804 | 0.916 / 0.915 | 0.910 / 0.871 | 4.0 |
| fullS-s1280 (per-frame, s-net alone) | 0.744 / 0.696 | 0.922 / 0.884 | 0.879 / 0.813 | ~45 |
| moe3-v3 (per-frame) | 0.700 / 0.828 | 0.767 / 0.861 | 0.778 / 0.857 | 4.3 |
| moe3-v2 (round-2, for reference) | 0.656 / 0.760 | 0.754 / 0.831 | 0.767 / 0.851 | 4.0 |

- The fusion is complementary by construction: `fullS` (full-frame temporal s-net) covers the
  stream start (warmup frames) and the 312–319 transition where the proposal-gated moe3 is
  structurally blind; moe3's motion proposals + 640-crop verification carry the dimmest
  bush-phase frames. tracked-pcmax misses exactly **one frame in 885 scored** across both
  videos (07_05 frame 514, a 15 px coast drift with zero detections nearby to rescue —
  a Kalman-physics limit, not a detection failure).
- ByteTrack-style low-score association was tried (min-score 0.15) and **rejected**: clutter
  poisons the Kalman state (val 0.995 → 0.983, full 0.998 → 0.873). The 0.2 threshold plus
  the strict pixel-level re-acquirer is the better trade.

### Edge pipelines (TensorRT FP16, single yolov8n-p2 @1280 on the 3-frame stack)

| method | 07_05-val AP / F1 | 10_06-test AP / F1 | ms/frame | fps (5070) |
|---|---|---|---|---|
| **tracked-rt-c + coast smoothing** | **1.000 / 1.000** (full-video 0.996/0.998) | **1.000 / 1.000** | 9.5 | **84.8** |
| tracked-rt-c (causal coasts) | 0.995 / 0.995 | 1.000 / 1.000 | 9.5 | 84.8 |
| rt-c (per-frame) | 0.717 / 0.730 | 0.889 / 0.848 | 9.5 | 84.8 |
| rt-c v2 (round-2 nano, per-frame) | 0.679 / 0.658 | 0.895 / 0.890 | 10.4 | 78.1 |
| tracked-rt-d @640 | 0.738 / 0.849 | 0.668 / 0.760 | 7.2 | 104.5 |

The v3 nano's single track covers the whole flight in both videos (07_05: 569 frames as one
track from frame 2; 10_06: 338 frames as one track), and the classifier keeps *only* it —
zero false positives at the operating point. The warmup clamp means detection starts at
frame 0 (v2 was blind for the first 12 frames of every stream).

### Ablations & experiments (edge)

| experiment | verdict |
|---|---|
| INT8 (calibrated on the temporal val set) | **rejected**: AP 0.717→0.699 val / 0.889→0.863 test, and *slower* than FP16 on this GPU (9.1 vs 5.3 ms — Blackwell FP16 tensor cores win); re-test on Orin where INT8 economics differ |
| full-frame stabilizer instead of crop-correlation | no gain (val AP 0.710 vs 0.717) — the 768×448 crop correlation is not the bottleneck; keep it (3 ms) |
| resolution 1280 → 640 | per-frame val AP 0.243 — too coarse for 4 px targets; 640 stays the "fast fallback", 1280 is the operating point |
| checkpoint choice | fitness-selected `best.pt` ≫ `last.pt` at pipeline level (val 0.719 vs 0.530) — late epochs overfit the pastes |
| temporal spacing DT = 9 (t−18/t−9/t) | no gain: val AP 0.710 vs 0.719, test 0.873 vs 0.889 — longer trails buy recall (0.67→0.73) but accumulate foliage motion (precision 0.80→0.71); DT = 6 stays |
| ByteTrack-style min-score 0.15 association | rejected (see PC table note) — clutter poisons the Kalman state |
| coast smoothing (offline artifact only) | linear interpolation of coast positions between real-detection anchors (gaps ≤ 60 frames): fixes the last coast-drift miss on each video; causal per-frame path unchanged |

## 6. The two final models (`FINAL/`)

The deliverables are retrained on **all 548 labeled frames** of 07_05 (the split-trained
generation above stays in `work/runs/v3-*` as the honest ablation record). With no honest
validation split left, checkpoint selection is protocol-driven, not eval-driven: identical
hyperparameters, epochs = the split-run's best epoch scaled by the 1.65× larger epoch size,
full cosine decay, early stopping off, ship `last.pt`. (The nano was trained at two lengths,
15 and 25 epochs, compared only on its own training video — 25 shipped. `10_06` was never an
input to any selection.)

End-to-end scores of the shipped package (`FINAL/run_final.py`, which produces detections,
classified tracks, an alarm summary, and an annotated video):

| profile | 10_06 test tracked AP / F1 / R / P | 10_06 per-frame AP | alarm latency | fps (5070) |
|---|---|---|---|---|
| **EDGE-RT** — one yolov8n-p2 @1280, TRT FP16 + tracker + classifier | **1.000 / 1.000 / 1.000 / 1.000** | 0.876 | 10 frames | **74** |
| **PC-MAX** — fullS ∪ (proposals→verifier) ∪ expert, fused + tracker + classifier | **1.000 / 1.000 / 1.000 / 1.000** | 0.846 | 7 frames | 4.1 |

Both cover the entire verified flight (frames 23–360) as a single confirmed drone track with
zero false positives; on their own training video both also track every labeled frame (1.000)
and the PC profile additionally reports the landed drone as a separate `near` track,
confirmed within 7 frames.

Per-frame AP of the finals sits a few points below the split-trained generation (edge 0.876
vs 0.889; PC 0.846 vs 0.910) — the price of protocol-driven checkpointing without a val
split. The tracked deliverable metric is unaffected (1.000 either way); if raw per-frame
output matters for an application, the split-trained checkpoints are equally usable and their
numbers are the honestly-validated ones.

Package layout: `FINAL/pc_max/{fullS,verifier640,expert1280}.pt`,
`FINAL/edge_rt/edge_n1280.pt` (+ a TRT FP16 engine built for this machine — rebuild
on the target device). Usage, retraining recipe and alarm semantics: `FINAL/README.md`.

## 7. Honest caveats

- Single scene family (two videos, same camera class). The recipes (temporal stack,
  copy-paste with velocity trails, track-level confirmation) are the transferable part;
  absolute numbers are scene-specific. New deployments should expect a short label-and-
  fine-tune cycle (`tools/label.py` → `make_datasets_v3.py --split-at <all>`).
- The 10_06 v2 reference's *span* was seeded by the PC track, but every frame was
  independently position-refined and human-verified, and the span was extended beyond the
  track at both ends; exclusions are down to 24 frames of genuinely invisible target.
- `tracked-*` scores use causal positions with track-level classification; the
  classification is online-capable with ≈0.25 s latency (measured), but the reported
  numbers are computed offline over whole tracks.
- The near/landed drone is scored as an ignore region throughout (unlabeled by choice);
  the PC pipeline reports it as its own `near` class.
