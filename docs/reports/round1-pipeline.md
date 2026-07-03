# Tiny-Drone Detection & Tracking — Method Comparison Report

Video: `data/videos/07_05.mp4` — 1280×720 @ 30 fps, 571 readable frames (container metadata claims 741; read sequentially). Camera on a hovering drone; measured drift is sub-pixel per second (max ~1.2 px/s); stabilization applied throughout anyway.

## TL;DR — what won

**A layered pipeline, not a single model.** Stabilize → classical motion proposals → fine-tuned *tiny-specialist* YOLO verifies each proposal at native scale → a second *near/big expert* YOLO covers the full frame → motion-compensated Kalman tracker with strict re-acquisition → kinematic + appearance track filtering.

On the held-out hardest segment (the drone is a 3–5 px hazy speck):

- best single-frame method: **moe-hybrid** — AP(far) **0.401**, best-F1 **0.793**, near-drone recall **100%**, 0.44 false positives/frame;
- pure appearance (ft5-sahi) alone reaches far-recall **0.73** — but only after the tiny-object training recipe below; a *naively* fine-tuned YOLO gets **exactly 0.000** far recall even on its own training frames;
- tracking with detector-confirmed detections only: far-drone coverage **81.7%** (91.4% if unconfirmed motion tracks are allowed), near drone a single unbroken 526-frame track, median localization error **1.4–1.7 px**.

## The scene (established by verified ground truth)

Two drones plus wildlife:

- **near**: a landed quadcopter, bottom center (~180×105 px), static all video.
- **far** (the target): starts top-right at ~13 px, descends across the treeline, skims low over bushes leftward shrinking to ~5 px against moving vegetation (frames 0–290), makes a fast right-and-up dash (291–335 — unverifiable even offline; excluded from all scoring), U-turns near the right edge, then cruises left across the sky at **3–5 px** to the end (336–570).
- **Distractors**: several real birds (four crossing the treeline early; one flying parallel to the far drone 10–20 px below it), wind-blown foliage over the lower third, sub-pixel flicker on high-contrast edges.

GT was built semi-automatically (seeded continuity tracking in a non-causal windowed-median difference image) and verified visually on contact sheets (`work/gt_verify/`). Matching is by **center distance** (τ=12 px; scaled for the big near box) — IoU is meaningless at 4 px. A correction UI exists (`tools/label.py`) for future manual fixes.

## Detection: methods compared

| method | idea |
|---|---|
| `yolo-640` | pretrained YOLO11s, full frame at 640 ("just run YOLO") |
| `yolo-1280` | pretrained YOLO11s at native 1280 |
| `yolo-sahi` | pretrained YOLO11s on overlapping 640 tiles (SAHI) |
| `motion-median` | stabilized median background + per-pixel MAD noise + flicker suppression |
| `motion-mog2` | stabilized MOG2 background subtraction |
| `hybrid` | motion proposals → ×4-zoom crops → pretrained-YOLO verify, ∪ full-frame pass |
| `yolo-ft` | fine-tuned YOLOv8s-P2 @1280 full frame (ft1) |
| `*-hybrid` | hybrid with a fine-tuned verifier (ft1/ft2/ft3 …) |
| `ft5-sahi` | **tiny-specialist** (see recipe) on 640 tiles, appearance only |
| `moe-hybrid` | motion → **ft5** verification at 1:1 scale, ∪ **ft1** full-frame (mixture of experts) |

### Held-out comparison (val = frames 342–570; the 3–5 px cruise; ft models trained only on frames < 342)

| method | AP(far) | AP(all) | best-F1 | R(far) | R(near) | P | FP/frame | med-err px | fps |
|---|---|---|---|---|---|---|---|---|---|
| **moe-hybrid** | **0.401** | 0.727 | **0.793** | 0.603 | 1.000 | 0.784 | 0.44 | 1.1 | 8.3 |
| ft5-sahi | 0.400 | 0.323 | 0.402 | **0.729** | 0.262 | 0.339 | 1.93 | 1.3 | 12.3 |
| hybrid (pretrained) | 0.293 | 0.146 | 0.356 | 0.616 | 0.000 | 0.421 | 0.85 | 1.2 | 10.3 |
| yolo-ft-hybrid (ft1) | 0.280 | 0.735 | 0.729 | 0.603 | 1.000 | 0.668 | 0.79 | 1.1 | 10.7 |
| motion-median | 0.280 | 0.140 | 0.357 | 0.616 | 0.000 | 0.423 | 0.84 | 1.2 | 14.9 |
| yolo-sahi | 0.026 | 0.013 | 0.049 | 0.057 | 0.000 | 0.176 | 0.27 | 0.9 | 15.7 |
| motion-mog2 | 0.020 | 0.035 | 0.074 | 0.825 | 0.834 | 0.039 | 41.3 | 1.2 | 25.8 |
| yolo-1280 | 0.014 | 0.007 | 0.021 | 0.022 | 0.000 | 0.455 | 0.03 | 1.0 | 25.5 |
| yolo-640 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | – | 28.0 |
| yolo-ft / ft2-sahi / ft3-sahi | 0.000 | 0.500 | 0.667 | 0.000 | 1.000 | 1.000 | 0.00 | – | 13–23 |

(Full-video table in `work/eval_full.md`; same ordering, moe-hybrid leads with AP(far) 0.320 / AP(all) 0.670.)

Readings — each reproduces a finding from the research surveys on this video:

1. **Resolution is destiny**: 640 → native (1280/SAHI) triples pretrained AP(far) (0.06 → 0.14–0.16). At 720p, SAHI ≈ native-1280; SAHI becomes *the* tool at 4K.
2. **Pretrained appearance can't do this task** — best pretrained-YOLO far-recall on the cruise is 5.7%.
3. **Motion sees everything, knows nothing**: MOG2 hits 83% raw recall at 41 FP/frame. The median+MAD+flicker detector trades to 62% at 0.85 FP/frame.
4. **Fusion + specialization wins**: on the val segment, 138/145 of moe-hybrid's far hits are appearance-confirmed by ft5 at median confidence 0.93 (before ft5: zero confirmed — all raw motion at 0.4).
5. The **landed drone is invisible** to motion (static) *and* to pretrained YOLO (no drone class); any fine-tune fixes it trivially (recall 1.000, conf ~0.98).

## The tiny-object training saga (most valuable finding for the heavy fine-tune)

Getting a detector to *learn* a 4–8 px target took four attempts; each failure mode is directly relevant to the production fine-tune:

- **ft1** — YOLOv8s-P2, full frames @1280: learns the near drone perfectly, far drone recall **0.000 — including on its own training frames** (verified at conf 0.001: nothing above the 2·10⁻³ grid-noise floor at the drone's location).
- **ft2** — slicing-aided fine-tuning (640 native tiles, 3×/frame): same failure.
- **ft3** — + copy-paste of 71 clear drone patches, multi-scale: same failure. Labels verified correct and present in the mosaic'd training batches; P2 head verified (strides 4/8/16/32).
- **Controlled probe** — same trainer, same patches, pasted on drone-free sky-only images: **P 0.999 / R 0.992** in 30 epochs.

Diagnosis: **scale-mixture starvation**. In YOLOv8's task-aligned assignment the classification target and box-loss weight of a GT scale with the predicted IoU; a 180 px object in the same images dominates the normalized loss and holds tiny targets' effective gradient near zero indefinitely (initial IoU for an 8 px box ≈ 0.02 → its own targets stay ≈ 0.02 — a self-locking loop the near drone never lets it escape). This is precisely the failure class NWD/RFLA address.

- **ft5 (the fix)** — *tiny-specialist* trained on paste-only slices: the static near drone is **erased** from every frame (feathered dirt patch — a 640 window cannot geometrically avoid it in a 720p frame); positives are crisp patches pasted at scales 0.35–1.2 with **atmospheric-haze augmentation** (alpha-blend toward local background — the held-out speck is a pale hazy blob, the training patches are dark) and **fixed 24 px inflated labels** (the surveys' point-annotation advice; box size is irrelevant downstream — everything scores by center distance). Result: paste-val mAP50 0.951, and on the real held-out cruise: far-recall 0.73 alone, 0.93-confidence confirmations in the hybrid.

**Consequences for the heavy fine-tune**: (1) don't naively mix 100 px+ and <10 px instances — group by scale, tile, or adopt NWD/RFLA assignment; (2) inflate sub-10 px labels to a fixed ~24 px; (3) copy-paste with scale + haze jitter is the highest-leverage augmentation and nearly free; (4) validate that the model fits its *training* tiny instances before trusting any val number — "low loss" can mean "never asked".

## Tracking

SORT-style tracker in **stabilized coordinates** (Kalman constant-velocity + Hungarian on center distance) with: coasting (≤45 frames); **strict local re-acquisition** while coasting (±18 px window in a short-term median-background SNR image, unique-peak test, ≤12 consecutive — a loose version latched onto foliage and made clutter tracks immortal); **kinematic clutter filter** (needs one 40-detection window with net/path ≥ 0.55 and net ≥ 25 px — foliage jitter is zero-mean, flying objects sustain direction) with an **appearance exemption** (mean detector confidence ≥ 0.55 — so the hovering/landed drone is never filtered); duplicate-track merging.

On `moe-hybrid` detections:

| mode | far coverage | far IDs (switches) | longest streak | near coverage | flying-object tracks | clutter tracks |
|---|---|---|---|---|---|---|
| all objects (score ≥ 0.2) | **91.4%** | 4 (9) | 170 | 100% (1 track, 526 fr, 0.57 px err) | 10 total (incl. birds) | 0 |
| drone-confirmed (≥ 0.55) | 81.7% | 3 (5) | 154 | 100% | 5 | 0 (2 bird tracks partly confirmed) |

The two modes are the operating knob: *surveillance* mode tracks every flying object (birds shown as unconfirmed), *alarm* mode surfaces only appearance-confirmed drones. Two bird tracks still get occasional ft5 confirmations — at 4 px a bird and a drone genuinely look alike; the production answer is a track-level classifier on kinematics + appearance across the whole track (flap signature), plus more bird data in training.

## Recommended production pipeline (implemented, end to end)

1. **Stabilize** — phase-correlation translation (affine LK+RANSAC available for stronger ego-motion).
2. **Motion proposals** — median background + per-pixel MAD noise + flicker suppression: high recall on anything that moves, including 3 px specks, at a handful of FP/frame.
3. **Verify** each proposal with the **tiny-specialist** P2 YOLO on native-scale 640 crops; **union** with the **near/big-expert** full-frame pass. (`moe-hybrid`, 8.3 fps end-to-end single-thread on a laptop GPU without batching/optimization.)
4. **Track** — compensated Kalman + strict re-acquisition; coast through fades.
5. **Classify tracks** — detector-confirmed → drone alarm; directed-but-unconfirmed → bird/unknown; everything else never surfaces.

## What to do at the heavy fine-tune stage

- **Data**: Anti-UAV, Drone-vs-Bird, NPS-Drones, ARD100, AI-TOD; add your own backgrounds with copy-paste (scale + haze jitter) — it carried this whole exercise.
- **Loss/assignment**: adopt **NWD or RFLA** for <16 px boxes (the saga above is the empirical case for it); keep label inflation until then.
- **Scale handling**: either separate experts per scale band (as here) or tile so all targets land in a comparable relative-size regime.
- **Birds**: track-level drone-vs-bird classifier (kinematic + appearance features over the track).
- **4K inputs**: switch the full-frame pass to SAHI tiling (implemented; validated at 720p).
- **Evaluate** with this harness: center-distance AP per size bin + track coverage/ID switches/false tracks — never bare mAP@0.5.

## Reproduce

```bash
.venv/bin/python tools/build_gt.py                  # GT (verify sheets in work/gt_verify/)
.venv/bin/python -m dronedet detect --video data/videos/07_05.mp4 --method <name> --out work/det/<name>.json
.venv/bin/python -m dronedet eval --gt work/gt.json --dets work/det/*.json [--frames 342:571]
.venv/bin/python tools/make_dataset.py && .venv/bin/python tools/train_yolo.py            # ft1
.venv/bin/python tools/make_dataset_ft5.py && .venv/bin/python tools/train_yolo.py \
    --data work/dataset_ft5/data.yaml --imgsz 640 --batch 16 --name ft5-p2-640-tiny       # ft5
.venv/bin/python tools/final_round.py               # moe detections + tables + tracks + videos
```

## Artifacts

- `work/eval_val.md`, `work/eval_full.md` — comparison tables
- `work/vis/final_dets.mp4`, `work/vis/final_tracks.mp4` (+ earlier `hybrid_*.mp4`) — annotated videos
- `work/gt.json`, `work/gt_verify/*.png` — ground truth + verification sheets
- `work/models/yolo-ft-best.pt` (near/big expert), `work/models/yolo-ft5-best.pt` (tiny specialist), `work/models/ft*_results.csv` — weights + training curves
- `work/tracks/*.json` — tracker outputs; score with `tools/eval_tracks.py`

## Caveats

- One 24-second video: all fine-tuned numbers are scene-tuned; the *pipeline shape* and the *training-failure analysis* are the transferable results. The time-split (train 0–341 / val 342–570) keeps the headline numbers honest — the val segment's appearance (pale 3–5 px speck) never appears in training.
- GT is semi-automatic (motion-seeded, visually verified); frames 291–335 are excluded as unverifiable. GT construction slightly favors motion methods in localization.
- Birds at 4 px partially fool the appearance verifier; see track-classifier next step.
- fps figures are unoptimized single-thread Python; the motion detector and YOLO passes batch/parallelize trivially.
