# Real-time (edge) pipeline — SpeckLock on Jetson-class hardware

Goal: the same task as the PC pipeline — find and track a 3–14 px drone in 720p video — at **real-time rates on Jetson-Orin-Nano-class hardware**, with minimal accuracy loss. Trained on `data/videos/07_05.mp4` only (time-split); `data/videos/10_06.mp4` is a pure test set never touched by training.

## Result (round 3 — see docs/reports/round3-deliverables.md; round-2 numbers further below)

**RT-C** — stabilize → 3-frame temporal stack → **one YOLOv8-nano-P2 @1280 (TensorRT FP16)** → Kalman tracker → **track-level classification** — is the edge champion. Round 3 added v3 training data (dim bush-phase patch bank, sub-pixel trails, warmup-clamped stacks, hard negatives), the track classifier (`dronedet/trackclass.py`), and a hardened dense test reference (`gt_1006_v2`, every frame human-verified):

| tracked-rt-c (v3), causal | AP | best F1 | R | P | fps (RTX 5070 Laptop) |
|---|---|---|---|---|---|
| 07_05 val (hardest segment, hand labels) | 0.995 | 0.995 | 0.995 | 0.995 | 76.3 |
| **10_06 test (unseen, dense reference)** | **1.000** | **1.000** | 1.000 | 1.000 | **84.8** |

Both rows are [`work/eval3_0705_val.md`](work/eval3_0705_val.md) and [`work/eval3_1006_test.md`](work/eval3_1006_test.md), fps included — the rate really does differ by video (76.3 on 07_05, 84.8 on 10_06), so a single figure carried down a per-video column would be wrong in one of them. The offline **coast-smoothing** pass — coast positions linearly interpolated between real-detection anchors, an artifact for an operator to read back, never a live output — lifts the val row to 1.000 / 1.000 and scores the 07_05 *full* video at AP 0.996 / F1 0.998. That full-video run is recorded only in `docs/reports/round3-deliverables.md` §5; there is no table for it here.

One nano network at 9.5 ms of pipeline stage time (detector NN 5.3 + stabilize/warp 4.1 + tracker 0.1, `work/bench3.md`) against 11.8 ms of wall clock, the whole flight covered as a single track in both videos, zero false positives at the operating point (per-frame, before track integration: AP 0.717 val / 0.889 test). INT8 was net-negative on this GPU (slower *and* less accurate than FP16 — recalibrate on-device before using); DT=9 stacks and a full-frame stabilizer were tried and rejected (`docs/reports/round3-deliverables.md` §5).

## Round-2 result (previous generation, kept for the record)

| | AP (07_05 val, hand labels) | best F1 | precision | fps (RTX 5070 Laptop) | far-drone track |
|---|---|---|---|---|---|
| PC champion round 2 (`tracked-moe3`, s-model + proposals) | 0.960 | 0.938 | 0.884 | 4.0 | 97.1% cov, 1 ID |
| edge champion round 2 (`tracked-rt-c`, one nano net) | 0.932 | 0.862 | 1.000 (0 FP/frame) | 74.3 | 97.3% cov, 1 ID, 0.99 px |

**18× faster than the PC pipeline at −0.03 AP**, with zero false positives at its operating point. On the unseen test video (`data/videos/10_06.mp4`): per-frame **AP 0.894 / F1 0.894 / P 0.95 @ 78 fps**, and the tracker covers **99.6% of the verified flight as a single track (0.67 px median error)**.

One provenance note, since the −0.03 rests on it: the PC row is [`work/eval_user_val.md`](../work/eval_user_val.md) at the repo root, but the round-2 *edge* row — and with it the 97.3 % / 0.99 px and 99.6 % / 0.67 px track figures — is what this document carries with no surviving artifact. The round-2 tables in [`work/eval_0705_val.md`](work/eval_0705_val.md) and [`work/eval_1006_test.md`](work/eval_1006_test.md) carry the per-frame rows only (which is where the 0.894 / 0.894 / 0.95 @ 78 fps above is from); tracked rows were first written to disk a generation later, in [`work/eval3_0705_val.md`](work/eval3_0705_val.md). Read the round-2 tracker numbers as what was concluded at the time, not as something you can re-check.

## The six pipelines compared

Accuracy on the pure test set (10_06, against the visually-verified reference trajectory; "movers" = ignore regions):

| pipeline | what it is | AP | F1 | R | P | fps (RTX 5070 Laptop) | ms/frame |
|---|---|---|---|---|---|---|---|
| **rt-c-full1280** | temporal nano, full frame @1280 TRT | **0.894** | **0.894** | 0.844 | 0.950 | 78 | 10.4 |
| rt-d-full640 | same @640 (2× downscale) | 0.702 | 0.774 | 0.652 | 0.953 | **104** | 7.4 |
| rt-b-verify256 | motion proposals → temporal nano on 256px crops | 0.519 | 0.700 | 0.676 | 0.725 | 24 | 39.2 |
| rt-e-decimated | rt-b, verification every 2nd frame | 0.356 | 0.456 | 0.332 | 0.728 | 29 | 33.2 |
| rt-f-single1280 | **single-frame** nano @1280 (temporal ablation) | 0.188 | 0.274 | 0.220 | 0.362 | 70 | 12.6 |
| rt-a-classic | classical only, no NN | 0.180 | 0.355 | 0.504 | 0.274 | 36 | 26.0 |

(07_05-val table: [`work/eval_0705_val.md`](work/eval_0705_val.md). rt-c leads there too, at 0.679 AP, but only the first place survives the change of video: on the harder val segment rt-d falls from 2nd to 4th — 0.702 → **0.220** — and rt-b rises from 3rd to 2nd. A 2× downscale costs far more against a 3–5 px drone in foliage than against 10_06's sky, which is the same lesson as the resolution ablation below, arrived at by accident. The rt-c-versus-rt-b conclusion is the one that holds in both.)

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

| component | TRT FP16 (RTX 5070 Laptop) | ONNX CPU (32-core laptop) |
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
realtime/
  rt_stabilize.py    crop-correlation stabilizer (the v1→v2 story is in its docstring)
  rt_motion.py       O(1) lagged-EMA + 3-frame-diff detectors (kept for rt-a; the
                     lagged-median from dronedet proved better and is the default)
  rt_models.py       TRT/ONNX/pt detector wrappers (static-batch chunking)
  pipelines.py       RT-A..RT-F definitions + crowding-aware candidate ranking
  runner.py          per-stage-timed video runner
  tools/
    make_datasets_rt.py   round-2 training sets (superseded by tools/make_datasets_v3.py)
    export_models.py      TRT FP16 + ONNX exports (round 2)
    run_all.py            round-2: run + evaluate + bench everything
    run_round3.py         round-3: v3 engines, runs, tracked+classified, eval, bench
    bench_cpu.py          the component table above
    build_gt_1006.py      test reference v1 (from the verified PC track)
    harden_gt_1006.py     test reference v2: dense, independently refined,
                          every frame human-verified (gt_1006_v2.json)
  work/                  models (`.pt`), eval tables, bench, outputs
```

Run everything: `.venv/bin/python realtime/tools/run_all.py` (or a subset: `... run_all.py rt-c-full1280`). Retrain: `make_datasets_rt.py` → `tools/train_yolo.py --data realtime/work/dataset_full_temporal/data.yaml --model yolov8n-p2.yaml --imgsz 1280 ...` → `export_models.py`.

`work/models/` holds the six checkpoints and nothing else. `*.engine` is gitignored and a fresh clone contains none, because an engine is compiled for one GPU architecture and one TensorRT build and is worthless anywhere else — and every fps in this document is a TensorRT FP16 rate, so build one before quoting any of them. Both export helpers start from the *training* run directories (`work/runs/rt-ft8-n256-v2`, `rt-ftC-n1280-v2` and `rt-ftF-n1280` for round 2; `work/runs/v3-ftC-n1280` for round 3), which are gitignored and not distributed; what is tracked is the checkpoint they copied out, so export from that:

```bash
.venv/bin/pip install tensorrt   # requirements.txt deliberately does not pin it
.venv/bin/python -c "from ultralytics import YOLO; \
    YOLO('realtime/work/models/v3_full_temporal_n1280.pt').export(format='engine', half=True, imgsz=1280, batch=1, device=0)"
```

`rt_models.Detector` takes `.pt`, `.engine` and `.onnx` through the same argument, so a pipeline pointed at the `.pt` runs correctly — just not at the rate in these tables. The same call with `format='onnx'` produces the `.onnx` files, which is what the CPU column above is measured on — `bench_cpu.py` loads `full_temporal_n1280.onnx`, `full_temporal_n640.onnx` and `verifier_n256.onnx` from that directory, so it needs those three exported before it will run. `*.onnx` is gitignored here too, with `final/edge_rt/` the one exception, because the shipped edge model is the one that has to be portable.

## Honest caveats

- The 10_06 reference is the PC pipeline's visually-verified trajectory (measured frames only, others excluded; extra movers = ignore) — a hit-rate-along-trajectory test. The hand-labeled comparison is 07_05-val.
- Orin numbers are projections until run on the device; TRT engines must be rebuilt on the target (arch-specific).
- The near/landed drone is out of scope for rt-c/d (trained with it erased); on edge it is covered by the low-rate expert pattern (rt-b/e include it at 1/30 frames, amortized 0.7 ms).
- Single scene family; the training recipe (temporal stack + velocity-trail copy-paste) is the transferable part, per the main reports.
