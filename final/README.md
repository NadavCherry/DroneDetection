# Deliverable models — PC-MAX & EDGE-RT

*The two shipped profiles, in `final/`. For the full project overview see the [main README](../README.md); for the design story see [round-3 report](../docs/reports/round3-deliverables.md).*

Both models are trained on **all 548 labeled frames of `data/videos/07_05.mp4`** (the split-trained
ablation generation behind the numbers in `docs/reports/round3-deliverables.md` §5 is preserved under `work/runs/v3-*`).
`data/videos/10_06.mp4` was never trained on and never used for model selection — it is the test video.

> **Full-length test video.** The source files hide their opening seconds behind an MP4
> edit list (`tools/recover_full_video.py` recovers them losslessly — `data/videos/10_06.mp4` is really
> 591 frames / 19.7 s, not 361 / 12 s). Both profiles were re-run end-to-end on the
> recovered full video: tracked **AP/F1/R/P = 1.000, zero false alarms** hold on the
> labeled range, and PC-MAX additionally finds + confirms the drone's earlier *exit pass*
> in the pre-roll (drone track, frames 14–86; it re-enters at the same spot 7 s later as
> the known flight). Annotated full-length videos: [PC-MAX](../docs/media/10_06_pcmax_tracks.mp4) ·
> [EDGE-RT](../docs/media/10_06_edgert_tracks.mp4) ·
> [side-by-side vs baseline](../docs/media/10_06_baseline_vs_pcmax_vs_edgert.mp4).

## 1. PC-MAX — most powerful, desktop GPU

Architecture (see `docs/reports/round3-deliverables.md`): three complementary detection streams fused, then
temporal integration —

<p align="center">
  <img src="../docs/media/architecture_pcmax.svg" width="1000" alt="PC-MAX architecture"/>
</p>

- shipped-package score on the unseen test video: **tracked AP/F1/R/P = 1.000, zero
  false positives** (per-frame AP 0.846); drone confirmed 7 frames after track birth;
  also reports the landed drone as a separate `near` track. ~4 fps on an RTX 5070 laptop.
  (Split-trained generation for honest-ablation numbers: `docs/reports/round3-deliverables.md` §5 — same 1.000
  tracked, per-frame 0.910.)
- final weights: `pc_max/fullS.pt` (`final-ftS-s1280`), `pc_max/verifier640.pt`
  (`final-ft7-s640`), `pc_max/expert1280.pt` (round-1 ft1, unchanged)

## 2. EDGE-RT — real-time, edge hardware

One nano network on the 3-frame stabilized stack — no proposal stage, no expert:

<p align="center">
  <img src="../docs/media/architecture_edgert.svg" width="1000" alt="EDGE-RT architecture"/>
</p>

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
.venv/bin/python final/run_final.py --video path/to/video.mp4 --profile pc-max  --out out_pc
.venv/bin/python final/run_final.py --video path/to/video.mp4 --profile edge-rt --out out_edge
```

Outputs: `dets.json` (per-frame, original coords, fps + stage timings in `meta`),
`tracks.json` / `tracks_drone.json` (all / classified-drone tracks),
`tracked_dets.json` (track-integrated detections, coast-smoothed),
`alarms.txt` (per drone track: span, coverage, confirmation frame + latency),
`annotated.mp4` (classified tracks painted).

Alarm semantics: a track is announced as a drone once it accumulates 8
verifier-confirmed detections (`dronedet/trackclass.py`); measured confirmation
latency from track birth is 7 frames for PC-MAX and 10 frames for EDGE-RT
(~0.25–0.35 s at 30 fps).

## Retraining for a new deployment

```bash
.venv/bin/python tools/label.py                     # hand-label a short clip
.venv/bin/python tools/build_gt_user.py             # labels -> gt_user.json
.venv/bin/python tools/make_datasets_v3.py --split-at <n_frames> --suffix _all
# edge net: yolov8n-p2 @1280 on dsv3_full_temporal_all; PC nets: see tools/final_round3.py
```
