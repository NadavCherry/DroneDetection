# The edge model: what it actually is, and what it actually runs at

**Status: measured.** Reproduced on an RTX 4090, engine rebuilt for that card, accuracy and
speed taken from the same pass.

## 1. What the "ultra-fast variant" is

It exists. It is **EDGE-RT** (research name RT-C): **one YOLOv8n-P2** reading the same
three-moment ego-stabilised stack, **full-frame at 1280 px** — no tiling, no proposal
stage, no expert, no second network. Shipped as `final/edge_rt/edge_n1280.pt` (6.6 MB) with
`.onnx` and `.fp16.onnx` beside it.

**The remembered "100+ FPS model" is not a separate model.** `rt-d-full640` is the *same
checkpoint exported at half resolution*. Verified by hashing every tensor blob inside both
`.pt` archives:

```
full_temporal_n1280.pt   441 tensors   weights-sha256 = 75c9b80997c49cfeff0d
full_temporal_n640.pt    441 tensors   weights-sha256 = 75c9b80997c49cfeff0d
```

Identical. So the remembered speed was bought with **resolution**, not with a smaller
network, and its accuracy cost belongs on the same axis as its rate. That is what the table
below does.

## 2. Accuracy against speed, one GPU, one pass

RTX 4090. `10_06.mp4` scored against `gt_1006_v2.json`, centre-distance matching at
τ = 12 px. Speed and accuracy come from **the same execution**, so the two axes cannot
drift apart.

| backend | imgsz | **AP** | recall | precision | **fps (steady p50)** | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|---|---|
| `.pt` | 1280 | **0.8793** | 0.875 | 0.799 | 35.2 | 28.4 | 30.5 | 32.1 |
| `.pt` | 640 | 0.6464 | 0.605 | 0.891 | 40.2 | 24.9 | 26.8 | 27.6 |
| **engine** | **1280** | **0.8757** | 0.858 | 0.812 | **58.9** | 17.0 | 18.9 | 19.9 |
| engine | 640 | 0.6393 | 0.602 | 0.898 | **72.1** | 13.9 | 15.3 | 15.8 |

Three readings, in order of how much they should change what anyone claims:

**The 100+ FPS figure did not reproduce.** The fastest configuration measured here is
**72.1 fps**, and it is the one whose AP is 0.639. Nothing on this GPU reached three
figures. The published 104 fps was taken on an RTX 5070 Laptop with a different checkpoint
(`v3_full_temporal_n640`, whose `.pt` is not committed — see §4), so it is not refuted, it
is *unreproduced*, and the repository should stop quoting it as if it were a property of
the shipped model.

**Halving the resolution is a bad trade.** 1280 → 640 with an engine buys **1.22×** speed
(58.9 → 72.1 fps) and costs **0.236 AP** (0.876 → 0.639). Recall collapses from 0.858 to
0.602 — the model stops finding the target, and the precision rise to 0.898 is the usual
consequence of firing less often, not an improvement. If the edge profile is ever deployed,
**1280 is the operating point** and 640 is a fallback for hardware that cannot hold it.

**The TensorRT engine is worth 1.67×** at 1280 (35.2 → 58.9 fps) and the repo does not ship
one, because engines are architecture-specific (`.gitignore` excludes `*.engine`). A fresh
clone silently runs the `.pt` at 35 fps. Both rows are given so a reader can see which
number their own machine will produce on day one.

## 3. At the fast end the bottleneck is the CPU, not the GPU

Per-stage means, ms/frame:

| arm | total | detector NN | stabilise + warp |
|---|---|---|---|
| pt @1280 | 29.2 | 15.3 | 13.9 |
| pt @640 | 24.0 | 10.2 | 13.8 |
| engine @1280 | 28.3 | 20.2 | 8.1 |
| **engine @640** | **13.1** | **5.2 (39 %)** | **8.0 (61 %)** |

In the fastest configuration, **61 % of the frame is classical CPU stabilisation and only
39 % is the network.** Making the network faster from here — a smaller backbone, INT8, a
better engine — cannot buy much, because the network is no longer what the frame is spent
on. This is the same shape as the pursuit ring, where 208 ms of a 231.5 ms loop is the CPU
motion stage.

It also explains why a laptop could plausibly beat a 4090 at this task: the limiting
resource is single-thread CPU throughput for phase-correlation stabilisation, not tensor
cores. **An FPS number for this model is as much a claim about its CPU as about its GPU**,
and neither the published figures nor these state the CPU.

## 4. Reproducibility gaps found while doing this

* **The round-3 rt-d checkpoint is not committed.** `realtime/tools/run_round3.py` writes
  `v3_full_temporal_n1280.pt` and `v3_full_temporal_n640.pt`; only the first is in the repo.
  The 104 fps row and its AP 0.631 therefore cannot be regenerated from a clean clone
  without first re-deriving that file — which is harmless only because §1 shows it is a
  byte-copy of the same weights.
* **No edge variant exists in the experiment registry.** All eight registered experiments
  are `yolov8s-p2` at 640. The shipped edge model cannot be reproduced through
  `tools/train.py --config`; it came from `realtime/`'s own scripts.
* **A one-time TensorRT cost can swamp a short clip.** The first run of this benchmark
  reported **2.03 fps end-to-end** for the engine at 1280 while its p50 was 16.6 ms — 60 fps.
  A single initialisation frame cost ~172 s and, spread over 361 frames, destroyed the
  average. `tools/bench_edge.py` now reports warm-up cost and the slowest frame explicitly,
  and leads with the steady-state percentile rather than the end-to-end mean. Without the
  warm-up discard this table would have said the faster backend was seventeen times slower
  than the slower one.

## Reproducing

```bash
sbatch cluster/edge_bench.sbatch
```

Builds engines for the local card, then runs all four arms.
Artifacts: `work/reports/edge/edge_bench.{md,json}` and per-arm `*-dets.json`.
