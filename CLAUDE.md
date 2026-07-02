# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computer-vision pipeline for detecting and tracking tiny drones (3-14 px) in RGB video from a (slightly drifting) drone-mounted camera. The `dronedet/` package implements competing detection methods behind one interface, a center-distance evaluation harness, and a camera-motion-compensated Kalman tracker. Two rounds of results: `REPORT.md` (pipeline build + tiny-object training saga, vs semi-automatic GT) and `REPORT2.md` (vs the user's manual labels in `work/gt_user.json` — **the authoritative GT**; the auto-GT's second half had tracked a bird). Round-2 winner: `moe3-stacked` — dual motion proposals (lagged-background slow-mover detector ∪ MOG2) verified by a **temporal 2-class specialist** (3 stacked stabilized grays: t-12/t-6/t) + ft1 full-frame expert; with tracker feedback (`tracked-moe3`) it reaches F1 0.938 / R 1.0 on the hardest segment.

## Environment & Commands

- Python 3.12 venv at `.venv/`; deps in `requirements.txt`. Torch needs the cu128 index (RTX 5070 is Blackwell/sm_120): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
- **Gotcha**: `~/.config/Ultralytics/settings.json` redirects `runs_dir` to `~/PycharmProjects/TheAgency_workspace/runs`, and relative `project=` paths get nested under it — ultralytics training output does NOT land in this repo. Pass an absolute `project` path or look there.
- The video container metadata claims 741 frames; only 571 are readable. Always read sequentially (`dronedet.video.frames`), never seek.

```bash
# run a detector over the video (methods: yolo-640, yolo-1280, yolo-sahi,
# motion-median, motion-mog2, hybrid, yolo-ft, yolo-ft-hybrid, yolo-ft-sahi;
# pass --weights and --method-kw to build fine-tuned / mixture-of-experts
# variants -- see tools/final_round.py for the exact winning configuration)
.venv/bin/python -m dronedet detect --video 07_05.mp4 --method hybrid --out work/det/hybrid.json

# score detection JSONs against GT (optionally --frames 342:571 for the val split)
.venv/bin/python -m dronedet eval --gt work/gt.json --dets work/det/*.json

# tracker + track scoring + annotated videos
.venv/bin/python -m dronedet track --video 07_05.mp4 --dets work/det/hybrid.json --out work/tracks/hybrid.json --min-score 0.2
.venv/bin/python tools/eval_tracks.py --tracks work/tracks/hybrid.json
.venv/bin/python -m dronedet render --video 07_05.mp4 --dets work/det/hybrid.json --out work/vis/dets.mp4

# ground truth (semi-automatic seeded tracking + visual verification sheets)
.venv/bin/python tools/build_gt.py

# manual labeling UI (browser canvas: wheel-zoom + loupe, drag/move/resize
# bboxes, per-frame autosave). Seed from an existing GT to correct it.
.venv/bin/python tools/label.py [--from-gt work/gt.json]      # -> work/labels.json
.venv/bin/python tools/label.py --export-gt work/gt.json      # labels.json -> GT format

# fine-tune ft1 (near/big expert): full frames @1280
.venv/bin/python tools/make_dataset.py && .venv/bin/python tools/train_yolo.py
# round-2 specialists on the manual labels (gt_user.json):
.venv/bin/python tools/build_gt_user.py     # user labels -> gt_user.json + agreement stats
.venv/bin/python tools/make_dataset_ft6.py  # 2-class single-frame slices
.venv/bin/python tools/make_dataset_ft7.py  # 2-class TEMPORAL slices (stacked stabilized grays)
.venv/bin/python tools/train_yolo.py --data work/dataset_ft7/data.yaml --imgsz 640 \
    --batch 16 --name ft7-p2-640-stacked
# round-2 finale: moe2/moe3/sr detections, trackers, tables, painted videos
.venv/bin/python tools/final_round2.py
```

There is no test suite; the eval harness + verification contact sheets in `work/gt_verify/` are the correctness checks.

## Architecture

Everything works in two coordinate frames: **original** frame pixels (all stored artifacts: detections, GT, tracks) and **stabilized** reference-frame coordinates (internal to motion detection/tracking). `dronedet/stabilize.py` estimates the per-frame global transform (phase correlation against frame 0; affine LK+RANSAC variant available); per-frame shifts are stored in every detection JSON's `meta.shifts` so downstream stages never recompute them.

- `dronedet/motion.py` — the classical detector: temporal-median background + per-pixel MAD noise (a pixel is foreground at `k_sigma` × its own historical noise) + **flicker map** (EMA of past foreground rate raises the threshold where motion recurs — kills wind-blown foliage, which oscillates in place, while a transiting drone passes).
- `dronedet/methods/` — registry (`build_method`) of detectors implementing `process(idx, frame, m_stab) -> [Detection]`. `hybrid.py` is the flagship: motion proposals → zoomed/native crops → YOLO verification, unioned with a full-frame YOLO pass (catches the static/landed drone that motion can't see). It supports **mixture-of-experts**: `full_weights`/`full_classes` load a second model for the full-frame pass (winning config: ft5 tiny-specialist verifies crops at 1:1 scale, ft1 covers the full frame). Pretrained variants use COCO classes airplane/bird/kite at very low conf; fine-tuned variants pass `drone_classes=None`.
- `dronedet/evaluate.py` — matching by **center distance** (`max(tau, 0.5*sqrt(gt_area))`), never IoU (meaningless at 4 px). GT `meta.exclude_frames` (frames 291-335: unverifiable fast dash) are skipped entirely.
- `dronedet/track.py` — Kalman CV tracker in stabilized coords; Hungarian on center distance; coasting; **strict local re-acquisition** (unique-peak test + max consecutive uses — a loose version latches onto foliage and makes clutter tracks immortal); post-filters: **directedness** (some 40-real-detection window with net/path ≥ 0.55 and net ≥ 25 px — foliage jitter is zero-mean, flying objects sustain direction) + duplicate-track merging. Birds pass the kinematic filter by design; drone-vs-bird is the appearance layer's job (fine-tuned detector confidence per track).
- `work/gt.json` — per-frame boxes for `far` (the moving target, 3-14 px) and `near` (landed drone, static box mapped by camera shift). Built by `tools/build_gt.py`: seeded continuity tracking in a *non-causal* windowed-median diff, verified visually via contact sheets.
- `tools/label.py` — manual labeling UI (stdlib HTTP server + `cv2`, no extra deps). Decodes the whole video sequentially once (never seeks) into in-memory JPEGs, serves a canvas SPA. Its own store is per-frame corner boxes in **original coords** (`work/labels.json`: `frames[i] = [{x1,y1,x2,y2,label}]`); it seeds from (`--from-gt`) and exports to (`--export-gt` / in-UI button) the `gt.json` object format, grouping boxes by `label` and preserving existing `meta` (shifts). Autosaves per frame; the tab-close beacon POSTs (sendBeacon can't PUT) to the same save route.

Test video ground truth story (verified): the far drone descends from top-right sky, skims low over bushes leftward (frames 0-290), dashes right/up (291-335, unverifiable → excluded), U-turns near the right edge, then cruises left across the sky shrinking to ~3 px (336-570). The scene contains several real birds (including one flying parallel to the drone) — they are legitimate flying-object tracks, not clutter.

**The tiny-object training lesson** (details in REPORT.md): fine-tuning YOLO on frames that contain both the 180 px near drone and the <10 px far drone produces a model with *zero* recall on the tiny target — even on its own training frames — because task-aligned assignment scales tiny GTs' loss contribution by their (near-zero) predicted IoU while the big object dominates. A controlled paste-only probe learns the same targets at R≈0.99. Hence the tiny specialist (`tools/make_dataset_ft5.py`): near drone erased from backgrounds, crisp multi-scale pastes with haze augmentation, fixed 24 px inflated labels (everything downstream scores by center distance, so box size is irrelevant). Never mix large and sub-10 px instances naively; verify a model fits its *training* tiny instances before trusting val numbers.

## Reference Documents

- `drone_detection.md` and `compass_artifact_wf-*.md` — the research surveys that drove the design (SAHI tiling, P2 heads, NWD loss, motion-first pipelines, evidence-backed impact rankings). The whole of `drone_detection.md` is a single line of text; line-based tools are awkward on it.
- `זיהוי רחפנים ואובייקטים זעירים.docx` — Hebrew-language source document, same topic.

Evaluation principle from the research (implemented here): use size-binned, center-distance metrics and detection/false-alarm rates, not plain mAP@0.5 — IoU metrics are unreliable at few-pixel scale. Priority ranking when trading off effort: motion/temporal cues > resolution preservation > tiling > tiny-aware loss (NWD, not yet implemented) > P2 head.
