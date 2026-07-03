# Run inference on a new video

Three ways to run, from "just give me the answer" to "run any method in the toolbox". All commands assume you run from the repo root with the venv active (or prefix `.venv/bin/python`).

```bash
pip install -r requirements.txt   # torch needs the cu128 index on Blackwell GPUs (see requirements.txt)
```

---

## 1. The shipped models (recommended) — `final/run_final.py`

Runs one of the two deliverables end to end: detect → fuse → track → classify → paint.

```bash
# most accurate (desktop GPU, ~4 fps)
python final/run_final.py --video path/to/video.mp4 --profile pc-max  --out out_pc

# real-time (TensorRT FP16, ~74–85 fps on an RTX 5070)
python final/run_final.py --video path/to/video.mp4 --profile edge-rt --out out_edge
```

Outputs in `--out`:

| file | contents |
|---|---|
| `annotated.mp4` | the video with classified tracks painted (**start here**) |
| `alarms.txt` | per drone track: start frame, confirmation frame + latency, coverage |
| `tracks_drone.json` | only the tracks classified as drone / near |
| `tracks.json` | every confirmed track (tracker output) |
| `tracked_dets.json` | track-integrated per-frame detections (the headline-metric input) |
| `dets.json` | raw per-frame detections (original coords) + fps + stage timings in `meta` |

**Alarm semantics:** a track is announced as a drone once it accumulates 8 verifier-confirmed detections ([`dronedet/trackclass.py`](../../dronedet/trackclass.py)); measured latency from track birth is 7 frames for PC-MAX and 10 frames for EDGE-RT (~0.25–0.35 s at 30 fps).

> **EDGE-RT note:** the shipped `.engine` is built for the host GPU. On a different machine it falls back to the `.pt` (still correct, just not TRT-fast); rebuild the engine on the target device with [`realtime/tools/export_models.py`](../../realtime/tools/export_models.py).

---

## 2. The research pipeline — `tools/run_best.py`

The round-2 flagship (temporal verifier + expert + tracker) with a friendlier, annotated-video-first output. Good for eyeballing behavior on a clip.

```bash
python tools/run_best.py --video path/to/video.mp4
```

Outputs under `work/infer/<video-stem>/`:

- `tracks_confirmed.mp4` — drone alarms only (the "alarm" view)
- `tracks_all.mp4` — every confirmed flying-object track (drones **and** birds)
- `dets.mp4` — raw detections above `--show-score` (default 0.5)
- `summary.txt` — per-track summary, each track attributed to *drone* or *unconfirmed flying object*

Reading the annotated video: a solid `id1` label means the target was detected this frame; `id1?` (yellow) means a coasted / re-acquired estimate.

Swap in other weights with `--weights` (verifier) and `--full-weights` (expert).

---

## 3. Any method in the toolbox — `python -m dronedet`

Every method in the [registry](methods.md) is runnable directly, and the stages compose:

```bash
# 1. detect (methods: yolo-640/-1280/-sahi, motion-median/-mog2, hybrid,
#    yolo-ft*, moe2-hybrid, moe3-stacked — see docs/guides/methods.md)
python -m dronedet detect --video data/videos/07_05.mp4 --method moe3-stacked --out work/det/moe3.json

# 2. score against ground truth (optionally a frame range, e.g. the hard val split)
python -m dronedet eval --gt work/gt_user.json --dets work/det/*.json --frames 342:571

# 3. track, then render
python -m dronedet track  --video data/videos/07_05.mp4 --dets work/det/moe3.json --out work/tracks/moe3.json --min-score 0.2
python -m dronedet render --video data/videos/07_05.mp4 --dets work/det/moe3.json --out work/vis/moe3.mp4
```

To build a mixture-of-experts / fine-tuned variant, pass `--weights` and `--method-kw` (JSON) — see [`tools/final_round.py`](../../tools/final_round.py) / [`final_round2.py`](../../tools/final_round2.py) for the exact winning configurations.

---

## Compare against a plain detector — `tools/run_baseline.py`

Same visualization for any single-frame detector, so you can see the motion pipeline's gap over appearance-only:

```bash
python tools/run_baseline.py --weights baseline/yolo26n-new-data_full__2026_Jan_19.pt --video path/to/video.mp4
```

Outputs under `work/infer/<stem>_baseline/` (dets + annotated video + summary), at `--conf 0.02` by default.

---

## Video conventions

- Source videos live in [`data/videos/`](../../data/videos/): `07_05.mp4` (train/val) and `10_06.mp4` (**test only — never train on it**).
- Both files hide their opening seconds behind an MP4 **edit list** that every decoder silently honours, so "readable" frames < the container count. [`tools/recover_full_video.py`](../../tools/recover_full_video.py) remuxes losslessly to recover the full length (`10_06.mp4` is really 591 frames, not 361). The showcase videos in [`docs/media/`](../media/) are full-length runs.
- Always read sequentially (`dronedet.video.frames`) — never seek.
