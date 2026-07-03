# FINAL — the two deliverable models

Both models are trained on **all 548 labeled frames of `07_05.mp4`** (the split-trained
ablation generation behind the numbers in `REPORT3.md` §5 is preserved under `work/runs/v3-*`).
`10_06.mp4` was never trained on and never used for model selection — it is the test video.

## 1. PC-MAX — most powerful, desktop GPU

Architecture (see `REPORT3.md`): two complementary detection streams fused, then
temporal integration —

```
frame → stabilize → ┌ full-frame temporal yolov8s-p2 @1280 (fullS.pt)
                    ├ motion proposals (lagged-median ∪ MOG2)
                    │   → temporal verifier yolov8s-p2 @640 crops (verifier640.pt)
                    ├ near/big expert yolov8s-p2 @1280 (expert1280.pt, full RGB frame)
                    └→ score-aware fusion → Kalman tracker (stabilized coords,
                       local re-acquisition) → track-level classification → output
```

- shipped-package score on the unseen test video: **tracked AP/F1/R/P = 1.000, zero
  false positives** (per-frame AP 0.846); drone confirmed 7 frames after track birth;
  also reports the landed drone as a separate `near` track. ~4 fps on an RTX 5070 laptop.
  (Split-trained generation for honest-ablation numbers: `REPORT3.md` §5 — same 1.000
  tracked, per-frame 0.910.)
- final weights: `pc_max/fullS.pt` (`final-ftS-s1280`), `pc_max/verifier640.pt`
  (`final-ft7-s640`), `pc_max/expert1280.pt` (round-1 ft1, unchanged)

## 2. EDGE-RT — real-time, edge hardware

One nano network on the 3-frame stabilized stack — no proposal stage, no expert:

```
frame → gray → crop-correlation stabilizer (3 ms CPU) → stack(t−12, t−6, t)
      → yolov8n-p2 @1280 TensorRT FP16 → Kalman tracker → track classification
```

- shipped-package score on the unseen test video: **tracked AP/F1/R/P = 1.000, zero
  false positives** (per-frame AP 0.876); drone confirmed 10 frames after track birth.
  **~74 fps end-to-end** on the RTX 5070 (9.5 ms/frame); projected 10–15 fps FP16 on
  Jetson Orin Nano @1280 (rebuild the engine on-device; INT8 was net-negative on
  desktop — recalibrate on Orin before trusting it). Split-trained generation:
  tracked 1.000 val / 0.996 full / 1.000 test, per-frame 0.717 val / 0.889 test.
- final weights: `edge_rt/edge_n1280.pt` (`final-ftC-n1280-e25`) + `.engine`
  (TRT FP16, built for this machine)

## Run either model on any video

```bash
.venv/bin/python FINAL/run_final.py --video path/to/video.mp4 --profile pc-max  --out out_pc
.venv/bin/python FINAL/run_final.py --video path/to/video.mp4 --profile edge-rt --out out_edge
```

Outputs: `dets.json` (per-frame, original coords, fps + stage timings in `meta`),
`tracks.json` / `tracks_drone.json` (all / classified-drone tracks),
`tracked_dets.json` (track-integrated detections, coast-smoothed),
`alarms.txt` (per drone track: span, coverage, confirmation frame + latency),
`annotated.mp4` (classified tracks painted).

Alarm semantics: a track is announced as a drone once it accumulates 8
verifier-confirmed detections (`dronedet/trackclass.py`); measured confirmation
latency on both videos is 7–8 frames (~0.25 s at 30 fps) from track birth.

## Retraining for a new deployment

```bash
.venv/bin/python tools/label.py                     # hand-label a short clip
.venv/bin/python tools/build_gt_user.py             # labels -> gt_user.json
.venv/bin/python tools/make_datasets_v3.py --split-at <n_frames> --suffix _all
# edge net: yolov8n-p2 @1280 on dsv3_full_temporal_all; PC nets: see tools/final_round3.py
```
