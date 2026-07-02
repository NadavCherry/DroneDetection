# REALTIME — the HiveLab pipeline on edge hardware

Goal: the same task as the PC pipeline — find and track a 3–14 px drone in 720p video — at **real-time rates on Jetson-Orin-Nano-class hardware**, with minimal accuracy loss. Trained on `07_05.mp4` only (time-split); `10_06.mp4` is a pure test set never touched by training.

## Result

**RT-C** — stabilize → 3-frame temporal stack → **one YOLOv8-nano-P2 @1280 (TensorRT FP16)** → Kalman tracker — is the edge champion:

| | AP (07_05 val, hand labels) | best F1 | precision | fps (RTX 5070) | far-drone track |
|---|---|---|---|---|---|
| PC champion (`tracked-moe3`, s-model + proposals) | 0.960 | 0.938 | 0.884 | 4.0 | 97.1% cov, 1 ID |
| **edge champion (`tracked-rt-c`, one nano net)** | **0.932** | 0.862 | **1.000** (0 FP/frame) | **74.3** | **97.3% cov, 1 ID, 0.99 px** |

**18× faster than the PC pipeline at −0.03 AP**, with zero false positives at its operating point. On the unseen test video (`10_06.mp4`): per-frame **AP 0.894 / F1 0.894 / P 0.95 @ 78 fps**, and the tracker covers **99.6% of the verified flight as a single track (0.67 px median error)**.

## The six pipelines compared

Accuracy on the pure test set (10_06, against the visually-verified reference trajectory; "movers" = ignore regions):

| pipeline | what it is | AP | F1 | R | P | fps (5070) | ms/frame |
|---|---|---|---|---|---|---|---|
| **rt-c-full1280** | temporal nano, full frame @1280 TRT | **0.894** | **0.894** | 0.844 | 0.950 | 78 | 10.4 |
| rt-d-full640 | same @640 (2× downscale) | 0.702 | 0.774 | 0.652 | 0.953 | **104** | 7.4 |
| rt-b-verify256 | motion proposals → temporal nano on 256px crops | 0.519 | 0.700 | 0.676 | 0.725 | 24 | 39.2 |
| rt-e-decimated | rt-b, verification every 2nd frame | 0.356 | 0.456 | 0.332 | 0.728 | 29 | 33.2 |
| rt-f-single1280 | **single-frame** nano @1280 (temporal ablation) | 0.188 | 0.274 | 0.220 | 0.362 | 70 | 12.6 |
| rt-a-classic | classical only, no NN | 0.180 | 0.355 | 0.504 | 0.274 | 36 | 26.0 |

(07_05-val table: `work/eval_0705_val.md`. Same ranking, rt-c 0.679 AP.)

Three ablations built into the lineup:

- **temporal vs single-frame** (rt-c vs rt-f, identical everything else): AP 0.894 vs 0.188. The temporal input is worth ~5× — on edge exactly as on PC.
- **resolution** (rt-c vs rt-d): 1280 → 640 costs 0.19 AP, buys 33% more fps — the right knob when Orin-class GPU time is short.
- **network vs no network** (rt-a): classical motion+tracking alone reaches R 0.50 but P 0.27 — usable as a pre-filter, not as a product.

Surprise finding: the **proposal→verify architecture (rt-b), which wins on PC, loses on edge**. With a nano verifier the crops no longer dominate cost, so the full-frame single-pass design is simultaneously faster (one batched 1280 inference beats 8-16 small crops + motion bookkeeping in Python) *and* more accurate (no proposal-recall ceiling). Architecture choices do not transfer across compute classes — measure, don't assume.

## The bottleneck loop (how we got here)

Each iteration: profile → fix the biggest cost or accuracy hole → re-measure.

1. **Profile the PC pipeline** → verifier crops = 76% of 239 ms/frame; expert 9%; full-res stabilization 7%.
2. **Shrink the verifier**: s-model @640×20 crops → nano @256×8-16 (TRT FP16) — the nano temporal verifier trains to **mAP50 0.83 ≈ the s-model's 0.83** (the representation carries the signal, not the capacity). Verifier cost 201 → 2.4 ms.
3. **Full-frame temporal nano** (rt-c/d): kill the proposal stage entirely; first training reached only AP 0.13 on test → **added copy-paste with per-channel velocity trails to the full-frame dataset** (mAP50 0.48 → 0.70) → test AP 0.64.
4. **Proposal recall collapse on val** (rt-a/b ≈ 0): diagnosed in three steps — ranking (foliage clusters outrank a lone drone → crowding-aware ranking), knobs, and finally the real killer: **the downscaled-phase-correlation stabilizer silently drifted several px**, inflating the background noise floor. Fix: correlate a full-resolution 768×448 central crop against frame 0 — 3.6 ms, *more* accurate than the PC's 16.5 ms full-frame correlation (it ignores the moving-foliage borders). Proposal recall 0.16 → **0.89**; every pipeline improved (this is why rt-c ended at 0.894).
5. **TRT static-batch chunking** fix for >8 candidates.

## Edge projection (stated assumptions, not measurements)

Measured components (this machine):

| component | TRT FP16 (5070) | ONNX CPU (32-core laptop) |
|---|---|---|
| temporal nano @1280 | 4.9 ms | 153 ms |
| temporal nano @640 | 1.7 ms | 44 ms |
| verifier nano @256 ×8 | 2.4 ms | 41 ms |
| stabilize (crop corr) | — | 3.0 ms |
| lagged-median motion (amortized) | — | 0.2 ms |

Projection for **Jetson Orin Nano** (assuming its Ampere GPU runs nano-class CNNs 10–15× slower than a 5070 Laptop at FP16, and its A78 cores ~3× slower than this laptop's CPU — to be validated on device):

- **RT-C @1280**: ~50–75 ms NN + ~10 ms CPU → **10–15 fps** (FP16); INT8 ≈ up to 2× more.
- **RT-D @640**: ~17–26 ms NN + ~10 ms CPU → **25–35 fps** — real-time with the 0.70-AP operating point, or alternate 1280/640 frames.
- Weaker/no-GPU hardware: RT-D via ONNX-CPU ran at ~23 fps on this laptop's CPU; on small ARM boards expect a few fps — pair rt-a as a wake-up filter with duty-cycled NN verification (rt-e pattern).

All engines are FP16; INT8 calibration and DeepStream integration are the natural next steps on the device itself.

## Files

```
REALTIME/
  rt_stabilize.py    crop-correlation stabilizer (the v1→v2 story is in its docstring)
  rt_motion.py       O(1) lagged-EMA + 3-frame-diff detectors (kept for rt-a; the
                     lagged-median from dronedet proved better and is the default)
  rt_models.py       TRT/ONNX/pt detector wrappers (static-batch chunking)
  pipelines.py       RT-A..RT-F definitions + crowding-aware candidate ranking
  runner.py          per-stage-timed video runner
  tools/
    make_datasets_rt.py   the three training sets from 07_05 (train<342 only)
    export_models.py      TRT FP16 + ONNX exports
    run_all.py            run + evaluate + bench everything
    bench_cpu.py          the component table above
    build_gt_1006.py      test reference from the verified PC track
  work/                  models, engines, eval tables, bench, outputs
```

Run everything: `python REALTIME/tools/run_all.py` (or a subset: `... run_all.py rt-c-full1280`). Retrain: `make_datasets_rt.py` → `tools/train_yolo.py --data REALTIME/work/dataset_full_temporal/data.yaml --model yolov8n-p2.yaml --imgsz 1280 ...` → `export_models.py`.

## Honest caveats

- The 10_06 reference is the PC pipeline's visually-verified trajectory (measured frames only, others excluded; extra movers = ignore) — a hit-rate-along-trajectory test. The hand-labeled comparison is 07_05-val.
- Orin numbers are projections until run on the device; TRT engines must be rebuilt on the target (arch-specific).
- The near/landed drone is out of scope for rt-c/d (trained with it erased); on edge it is covered by the low-rate expert pattern (rt-b/e include it at 1/30 frames, amortized 0.7 ms).
- Single scene family; the training recipe (temporal stack + velocity-trail copy-paste) is the transferable part, per the main reports.
