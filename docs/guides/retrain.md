# Retrain, relabel & reproduce

Everything here trains on `data/videos/07_05.mp4` only, against the authoritative hand labels in [`work/gt_user.json`](../../work/gt_user.json). `10_06.mp4` is the test set and is **never** trained on.

The full four-step loop is: **label → build datasets → train → reproduce/evaluate**.

---

## Where the dataset is, and what it consists of

Datasets are **regenerable and gitignored** — they are rebuilt from the two inputs (the video + the GT JSON) by the builder scripts, so they are never committed. The current (v3) recipe, [`tools/make_datasets_v3.py`](../../tools/make_datasets_v3.py), does one stabilization pass and emits three YOLO datasets:

| dataset dir | what it is | trains |
|---|---|---|
| `work/dsv3_full_temporal/` | full-frame 3-channel stabilized temporal stacks @1280 | EDGE-RT (rt-c/d) + the PC full-frame pass |
| `work/dsv3_crop640/` | 640 px temporal slices | the PC verifier (`ft7-v3`) |
| `work/dsv3_crop256/` | 256 px temporal slices | the edge crop verifier |

Each is an ordinary Ultralytics dataset (`images/{train,val}/`, `labels/{train,val}/`, `data.yaml`), **2-class: drone / bird**. A `_all` suffix (e.g. `work/dsv3_full_temporal_all/`) means "no val split — trained on all labeled frames," which is what the shipped models use.

What's *in* each image is the whole tiny-object recipe (details in the [reports](../reports/)):

- **temporal stacking** — three stabilized grayscale moments as R/G/B, so movers show as colored trails;
- **copy-paste augmentation** — patches pasted **per channel along a simulated velocity** (rigid for drones incl. hover; faster + size-jittered for birds to mimic wing flap);
- **atmospheric-haze jitter** — distant targets fade toward the background;
- a **dim bush-phase patch bank** + **hard negatives** (v3), and **sub-pixel trails** for slow drifters;
- **fixed 24 px inflated labels** — IoU-based label assignment starves true-size sub-10 px boxes to *zero* recall; downstream scoring is by center distance, so box size is irrelevant;
- **never mixing** the 180 px landed drone with sub-10 px targets in one set — the big object's alignment scores dilute the tiny gradient to nothing (the landed drone is erased from verifier backgrounds and handled by the separate expert).

---

## 1. Label a new clip

Browser labeling UI (stdlib HTTP server + OpenCV, no extra deps): wheel-zoom + loupe, drag/move/resize boxes, per-frame autosave. Seed it from an existing GT to correct rather than start blank.

```bash
python tools/label.py                          # -> work/labels.json
python tools/label.py --from-gt work/gt.json   # seed from an existing GT to correct it
```

Then canonicalize the corner-box labels into the GT format the pipeline expects, with an agreement report:

```bash
python tools/build_gt_user.py                  # work/labels.json -> work/gt_user.json (+ agreement stats)
```

---

## 2. Build the training datasets

```bash
# current recipe — three datasets in one pass (add --split-at 548 --suffix _all to train on every labeled frame)
python tools/make_datasets_v3.py

# older single-purpose builders (still useful / referenced by the reports):
python tools/make_dataset_ft7.py     # round-2 2-class TEMPORAL 640px slices
python tools/make_dataset_ft6.py     # round-2 2-class single-frame slices
python tools/make_dataset_ft5.py     # round-1 tiny specialist (paste-only, big object erased)
```

---

## 3. Train

[`tools/train_yolo.py`](../../tools/train_yolo.py) wraps the Ultralytics recipe (P2 head, tiny-object settings). Key defaults: `--model yolov8s-p2.yaml`, `--imgsz 1280`, `--epochs 60`, `--batch 6`.

```bash
# PC temporal verifier (round-2 flagship recipe)
python tools/train_yolo.py --data work/dsv3_crop640_all/data.yaml --imgsz 640 --batch 16 --name ft7-v3-640

# edge full-frame nano (EDGE-RT)
python tools/train_yolo.py --data work/dsv3_full_temporal_all/data.yaml \
    --model yolov8n-p2.yaml --imgsz 1280 --batch 8 --name edge-n1280
```

> ⚠️ **Ultralytics runs-dir gotcha.** `~/.config/Ultralytics/settings.json` redirects `runs_dir` out of this repo, and relative `project=` paths get nested under it — training output does **not** land here by default. Pass an absolute `project=` or look under the redirected runs dir. (RTX 5070 / Blackwell needs the cu128 torch wheels — see [requirements.txt](../../requirements.txt).)

**Where the result lands, and what is in git.** Training runs land under `work/runs/<name>/weights/`,
which is gitignored — a fresh clone has no rounds-4–7 checkpoints and every command that names one
fails until you have run the training above. The weights that *do* ship are the rounds-1–3 ones
(`final/`, `work/models/`, `realtime/work/models/`, `baseline/`) plus the two round-7 fusion models
force-added under `work/runs/combined-fusion-{m,s}-p2-2/`. Copy a checkpoint you intend to keep into
`work/models/` — that directory is re-allowed by name in `.gitignore`, so a new weights location is
otherwise silently untracked. Full map: [methods.md §3](methods.md#weights).

---

## 4. Export edge engines & reproduce a full round

```bash
# TensorRT FP16 + ONNX exports for the edge nets (rebuild on the TARGET device — engines are arch-specific)
python realtime/tools/export_models.py

# reproduce an entire round (runs every method, tracks, evaluates, benches, paints videos):
python tools/final_round2.py            # round 2 (PC)
python tools/final_round3.py            # round 3 (PC deliverables)
python realtime/tools/run_all.py        # the six edge pipelines (or: run_all.py rt-c-full1280 for a subset)
python realtime/tools/run_round3.py     # round-3 edge: v3 engines, runs, tracked+classified, eval, bench
```

Comparison tables land in `work/eval_*.md` and `realtime/work/*.md`; benches in `realtime/work/bench*.md`
— all of those are committed, so the rounds-1–3 numbers can be re-read without re-running anything.

Rounds 4–7 train on public datasets instead (ARD-MAV, NPS), which are downloaded rather than
redistributed and are built into `work/ext_datasets/` — a gitignored directory that a fresh clone
does not have. Their dataset builders and training commands are in each round's *Reproduce* block:
[round 4](../reports/round4-external-datasets.md#7-reproduce) (single-dataset tiles),
[round 5](../reports/round5-moving-camera-multidataset.md#9-reproduce) (the combined corpus),
[round 6](../reports/round6-max-pipeline.md#reproduce) (the temporal expert),
[round 7](../reports/round7-fusion.md#reproduce) (the 4-channel RGB+motion tiles and NWD).

---

## Production fine-tune checklist (evidence-based)

From the [reports](../reports/), in priority order: **temporal input channels first** → NWD/RFLA assignment for tiny boxes → copy-paste with scale + haze + trajectory jitter → scale-separated experts → **center-distance evaluation, never bare mAP@0.5**.
