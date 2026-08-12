# INFRA — from a fresh clone to a defensible number

You have just cloned the `sota-benchmark-infra` branch onto a machine with a bigger GPU than the
8 GB RTX 5070 Laptop this project was built on. This is the runbook. Follow it top to bottom; every
command is copy-pasteable from the **repo root**, and every number quoted here was measured, not
estimated, unless it says ESTIMATE.

Read [PLAN.md](PLAN.md) once for *why* this order. Read
[verified-measurements-2026-08.md](verified-measurements-2026-08.md) before you claim anything.
This file is the *how*.

**The one rule the tooling enforces, and the reason this infrastructure exists:**

> You may only subtract two numbers that share a protocol. Against a published scalar there is no
> significance test at all — only an interval and a coverage statement.

---

## 0. Tool status — check this first

This branch is under active construction. Run this before you start, so you know what is already
wired up on the copy you have:

```bash
ls tools/fetch_data.py tools/prepare_data.py tools/train.py tools/evaluate.py \
   tools/compare.py tools/dataset_stats.py tools/make_dataset_external.py \
   benchmarks/scorecard.py benchmarks/adapters configs/experiments
```

| stage | tool | status |
|---|---|---|
| fetch | `tools/fetch_data.py` | **built** — gate-aware, idempotent, refuses to fake a gated download |
| characterise | `tools/dataset_stats.py` | **built** |
| prepare | `tools/prepare_data.py` → `benchmarks/adapters/` | **built**, but ⚠ **only four adapters exist** (§3.2) |
| prepare (legacy) | `tools/make_dataset_external.py` | **built** — the only route for NPS and the combined multi-dataset corpus, and what the ARD-MAV experiment configs actually call |
| train | `tools/train.py` → `configs/experiments/` | **built** — named configs, seeds, a manifest written before the first batch |
| train (raw) | `tools/train_yolo.py` | **built**, but seedless and untraceable. Do not drive it by hand (§4) |
| evaluate | `tools/evaluate.py` → `benchmarks/scorecard.py` | **built** |
| compare | `tools/compare.py` | **built** |
| registries | `benchmarks/{catalog,protocol,published}.py` | **built**, stdlib-only, imported by CI |

The two gaps you will hit: **five of the ten datasets have no `prepare_data` adapter** (§3.2), and
the **bird experiment is blocked** on a dataset builder that does not exist and a Halmstad split that
nobody has defined (§4).

---

## 1. Setup

### 1.1 Clone and create the venv

```bash
git clone https://github.com/NadavCherry/SpeckLock.git
cd SpeckLock
git checkout sota-benchmark-infra
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

Python ≥ 3.11 (`pyproject.toml`); 3.12 is what CI runs.

### 1.2 Install torch **before** the package — the cu128 caveat

This is the single most common way a fresh machine goes wrong. `pip install specklock[train]`
resolves **CPU wheels** from PyPI. Blackwell (sm_120: RTX 5070/5080/5090, RTX PRO 6000) needs the
CUDA 12.8 build, and on those cards a CPU or cu121 wheel does not fail loudly — it either falls back
to CPU (training looks like it works and takes forty times as long) or dies with
`no kernel image is available for execution on the device`.

Install torch first, from the cu128 index, at the pinned versions. Because the pins match what
`pyproject.toml` declares, the next step then leaves it alone:

```bash
.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e ".[train,test]"
```

If your GPU is **not** Blackwell (Ampere/Ada: A100, RTX 3090/4090, L40S), a cu121 or cu124 build of
torch 2.11 is fine and usually faster to fetch — but keep the torchvision minor in step, and record
which build you used in the run notes, because it changes nothing about accuracy and everything
about throughput numbers.

Verify the card is actually visible:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 1.3 Verify the install

```bash
.venv/bin/python -m pytest -q
```

**Expect `801 passed` in roughly 35 s** — measured 2026-08-12 on this branch, of which `pursuit/tests`
contributes 540 and the rest are `dronedet/tests`. Treat it as a **floor, not an equality**: the
branch is under active construction and the count only grows. CI enforces floors of 600 total / 540
pursuit / 50 dronedet, so a *smaller* number means something collapsed — usually a `testpaths` entry
pointing at a directory that is not in the index, a mistake that once kept the build green for weeks
while collecting nothing.

Two constraints the suite is protecting, and which you will break if you are not careful:

* **The scoring path must import without torch or ultralytics.** CI installs four wheels
  (numpy, scipy, opencv-headless, pytest) and then imports `dronedet.metrics`, `dronedet.stats`,
  `dronedet.gt`, `dronedet.detections`, `benchmarks.*` and the pursuit core from a directory that is
  *not* the checkout, failing the build if any of them pulls in torch. Keep every torch/ultralytics
  import lazy, inside the function that needs it. That is the existing pattern everywhere.
* **Every link in the docs must resolve to a git-*tracked* file.** After editing any `.md`:

```bash
.venv/bin/python tools/check_docs.py
```

It also checks that scripts named in fenced `bash` blocks exist and that every `--flag` you type
against a repo script is really in that script's argparse. A command that only works on your machine
is caught here rather than by a reader.

---

## 2. Get the data

### 2.1 Day 0 — fire the two human requests before you download a byte

Both have multi-day latency and cost nothing to start. Do them now, in this order, then come back
and keep reading:

1. **Email `wosdetc@googlegroups.com`** asking for the **Drone-vs-Bird (DDS)** data-usage agreement.
   Sign and return it; the download link comes back by email. **Budget a week, possibly more.** This
   is the name-recognition benchmark and the long pole of the whole plan.
2. **Complete the ExtremeTrack / VISTAC-2 access form** at <https://sites.google.com/view/vistac-2>.
   188 videos (96 hazy + 92 rainy), splits 128/20/40 — the *only* real adverse-weather video with
   per-frame tracking boxes found in the entire 2026 sweep. The submission phase closed 2026-03-29;
   you want the data, not the entry.

Optional third, worth the email because it is also a collaboration hook: ask the **LRDDv3** authors
(Drexel iMaPLe, <https://research.coe.drexel.edu/ece/imaple/lrddv3/>) for the **source clips rather
than the public 5 FPS frame sampling**. At 5 FPS a 12-frame lag is 2.4 s of motion and the temporal
stack is meaningless, which is exactly why nobody has run a temporal method on the only dataset that
has birds *and* rain/snow *and* ground-truth range.

### 2.2 What the fetcher will and will not do

`tools/fetch_data.py` dispatches on `Dataset.gate` from `benchmarks/catalog.py`. Gated datasets are
**not attempted**: a fetcher that "tries anyway" writes a 4 KB HTML login page into `ARD100.zip` and
the failure surfaces a week later as a corrupt archive. Gated entries print the exact human action
(including the address to email) and exit **2**. Exit 0 = all good, 1 = a fetch or verification
failed, 2 = a human must act.

```bash
.venv/bin/python tools/fetch_data.py --list
.venv/bin/python tools/fetch_data.py --priority 2 --dry-run
```

Everything lands under `data/external/`. ARD-MAV is aliased to `data/external/ard_mav/` because that
is where it already lives; an existing tree is never deleted and an interrupted HTTP fetch resumes
from `.part`.

### 2.3 Priority order

| # | key | what it buys you | gate | human? | notes |
|---|---|---|---|---|---|
| 1 | `ardmav` | **the headline external claim.** 1080p, genuinely moving camera, targets from 6×3 px, 107,497 frames. Published SOTA is only **AP 0.55 @ IoU 0.25** (MGMD) — the largest headroom of any set here | Google Drive | no (needs `gdown`) | ships a published 15-video test split **and** a difficulty stratification (Ordinary / Complex / Small Objects) — a per-condition table for free |
| 2 | `halmstad` | **the bird claim.** The only set that is simultaneously video, bird/airplane/helicopter-labelled, night-inclusive, and CC0 with no form | open | no | 650 videos, 203,328 frames. Friction: MATLAB-format labels, an `.xlsx` manifest, and **no official split or metric** — you must define *and publish* a split file or your number is not reproducible |
| 2 | `uav_smid` | immediate hard negatives: 13,928 stills, five deliberately balanced classes (3,162–3,440 each) incl. bird and aeroplane | open | no | **stills** — can support the loss/matching ablations and false-positive claims, never the temporal claim |
| 3 | `extremetrack` | **the weather claim.** 188 videos, 96 hazy + 92 rainy, per-frame tracking boxes | form | **yes** | see §2.1 |
| 3 | `tricross` | air-to-air, moving camera, 73.8 % tiny, real + controlled fog — the only set hitting both stated priorities at once | unknown | **yes** | ⚠ `verified=False`: MDPI returned 403 to automated fetch. Confirm the route by hand, then record `download_id=` in the catalogue so it stops being manual |
| 4 | `smot4sb` | 108,192 frames of tiny birds from a moving camera and **zero drones** — a pure false-positive corpus. Plus a permanently-open Codabench leaderboard | open | no | also the peer-reviewed cover for centre-distance scoring (SO-HOTA) |
| 4 | `ard100` | head-to-head with YOLOMG, this project's nearest published relative; the tiniest targets of any drone video set | BaiduYun | **yes** | no mirror. Baidu account required, throttled hard, expect hours |
| 4 | `nps` | a second head-to-head | unknown | **yes** | ⚠ `verified=False`. **No official split** — every paper self-splits, so cross-paper NPS numbers are not comparable, and you must say so whenever you quote one |
| 5 | `dvb` | the reputational benchmark; bar to beat is Laroca et al. **mAP50 0.7390** | signed agreement | **yes** | see §2.1. Birds are present but **unlabelled** — labelling them and publishing the first bird-attributed false-alarm rate in nine editions is a bigger contribution than a leaderboard row |
| 6 | `dut_antiuav` | the easy end (SOTA 0.92–0.96), useful only because the Jul-2026 YOLOv11 edge baseline reports on it | open | no | quote **only** beside ARD-MAV/ARD100 with an explicit "this is the easy end" sentence |

Human-gated, restated so you cannot miss it: **`dvb` (email), `extremetrack` (form), `ard100`
(BaiduYun), `nps` and `tricross` (no verified route)**. Five of ten. Start them on day 0.

### 2.4 Disk budget

Sizes are archive estimates from the 2026-08 sweep except ARD-MAV, which is **measured**. Extracted
trees, 640 px tiles, runs and renders roughly double the raw footprint for anything you train on.

| slot | dataset | GB | running total |
|---|---|---|---|
| 1 | ARD-MAV (**measured 14.6 GB** archive; the briefing's 30–60 GB estimate was high) | 15 | 15 |
| 2 | Halmstad | 50 | 65 |
| 2 | UAV_SMID v2 | 5 | 70 |
| 3 | ExtremeTrack | 40 | 110 |
| 4 | SMOT4SB | 60 | 170 |
| 4 | ARD100 | 150 | 320 |
| 5 | Drone-vs-Bird DDS | 80 | 400 |
| — | working space: tiles, runs, engines, renders | ~150 | **~550** |

**Do not acquire**, and be ready to say why: **AOT** (11 TB full, ~500 GB "partial", grayscale
collision-course aircraft — eats the whole budget), the **4th Anti-UAV challenge set** (1.8 TB,
Zenodo restricted, IR-only, competition closed), any **event-camera** set (no sensor; cite as
convergent evidence), and **VisDrone** (a drone looking *down* at cars — the "static world cancels to
grey" assumption fails over a moving ground plane).

---

## 3. Prepare

### 3.1 Characterise before you train — and read the label-extent trap first

Run this on every dataset before you build anything from it:

```bash
.venv/bin/python tools/dataset_stats.py --dataset ardmav --per-video
.venv/bin/python tools/dataset_stats.py --gt work/gt_user.json --name 07_05
```

It prints AI-TOD size bins on √(w·h) and, per candidate `min_side`, **the best IoU a perfectly
centred prediction could still reach against the true annotation**.

**The trap.** Both dataset builders inflate small boxes so that YOLO's IoU-based label assignment
does not starve few-pixel ground truths of positive anchors —
`tools/make_dataset_external.py --min-side` (default **12**) and `tools/make_datasets_v3.py`'s
module constant `LABEL = 24.0`. That is defensible for *training* and fatal for *evaluation*: a
model trained on inflated labels predicts inflated boxes, and an inflated box cannot reach the IoU
threshold of the true annotation. The ceiling is pure arithmetic — a concentric grown box contains
the true box, so IoU = true area / inflated area — and it caps your score independently of how good
the detector is.

Measured on the real annotations, 2026-08-12:

**ARD-MAV** (106,456 boxes, median √area 11.8 px):

| `--min-side` | boxes grown | median best IoU | can reach IoU 0.5 | can reach IoU 0.25 |
|---|---|---|---|---|
| **8** | 29.3 % | 1.000 | **96.9 %** | 99.9 % |
| 12 (builder default) | 59.0 % | 0.833 | 74.4 % | 95.0 % |
| 16 | 72.9 % | 0.547 | 52.5 % | 78.7 % |
| 24 | 88.0 % | 0.243 | 32.0 % | 49.1 % |

**07_05, our own data** (`LABEL = 24.0`):

| min side | boxes grown | median best IoU | can reach IoU 0.5 | can reach IoU 0.25 |
|---|---|---|---|---|
| 8 | 77.9 % | 0.802 | 90.0 % | 99.8 % |
| 12 | 98.9 % | 0.440 | 38.1 % | 89.1 % |
| 16 | 100.0 % | 0.249 | 4.4 % | 49.5 % |
| **24 (current)** | 100.0 % | **0.110** | **0.0 %** | 2.9 % |

Two conclusions, and they are different:

* **ARD-MAV at 12 is tolerable, not fatal** — 95 % of boxes can still reach MGMD's IoU 0.25 and
  74 % can reach 0.5. Dropping to **8** takes IoU-0.5 reachability to 96.9 % for a 29 % inflation
  rate. Use `--min-side 8` for anything you intend to compare against a paper.
* **Our own data at `LABEL = 24` is arithmetically hopeless.** Median ceiling 0.110, **0.0 % of
  boxes can reach IoU 0.5**, so COCO AP is not low — it is structurally **0.000**. That is confirmed
  from a completely independent direction: measured detection-vs-GT IoU on 10_06 is mean 0.111,
  **max 0.111**. Predicted box width is a constant 24.0 px (range 24.0–24.0) and the 10_06 GT is
  itself a constant 8.0 px, giving a ceiling of (8·8)/(24·24) = 0.111.

This is why §4's headline experiment is **true extents + NWD**: NWD solves by a principled route the
same assignment problem the inflation was hacking around, and if it carries the assignment as
designed, comparability comes back for free.

### 3.2 Build the corpora — `prepare_data.py` where an adapter exists

`tools/prepare_data.py` is the preferred route. One command produces both things every later stage
needs, under `--out`:

```text
gt/<seq>.json                     dronedet ground truth, TRUE extent, one file per sequence
yolo/images/{train,val}/ + labels/ + data.yaml
splits.json                       which sequence went where, and where the split came from
manifest.json                     everything a reader needs to know what these numbers can mean
```

```bash
# Always look before you build — parses, prints the split and the extent report, writes nothing:
.venv/bin/python tools/prepare_data.py ardmav --out work/prepared/ardmav --dry-run

# ARD-MAV, official split, TRUE extent (the default), native-resolution 640 px tiles:
.venv/bin/python tools/prepare_data.py ardmav --out work/prepared/ardmav --tile 640 --stride 4

# Halmstad — the bird/night set. Define and PUBLISH the split; it has no official one:
.venv/bin/python tools/prepare_data.py halmstad --out work/prepared/halmstad

# UAV_SMID stills, every image, as hard negatives:
.venv/bin/python tools/prepare_data.py uav_smid --out work/prepared/uav_smid --stride 1

# Anything already in YOLO layout (a competitor's release, your own export):
.venv/bin/python tools/prepare_data.py yolo_dir --root some/dir --out work/prepared/x
```

Four properties of this tool that you should not work around:

* **`--min-side` defaults to 0 — true extent** — and every run prints the achievable-IoU table for
  the value chosen alongside 8/12/16/24, so the cost of inflating is on screen at build time rather
  than invisible until evaluation months later. Inflation is a training device with an evaluation
  cost; here it is opt-in, printed, and written into `manifest.json`.
* **Ground truth is never inflated, whatever `--min-side` says.** The flag reaches the YOLO labels
  only. An inflated ground truth is not a device, it is a wrong answer.
* **Test sequences are never exported as training images.** The split comes from
  `Dataset.official_test` / `official_val` via the adapter; `--export-test` writes them to
  `images/test`, which `data.yaml` does not reference.
* **Non-target classes survive into the GT as `ignore` objects**, so `dronedet.metrics` scores a hit
  on one as a *distractor* rather than discarding it — which is the only way "N hits on 3,162
  labelled bird instances" becomes a number. The YOLO labels drop those boxes, so a bird image
  trains as a hard negative. This is the mechanism behind the entire §5.4 confuser table.

⚠ **Only four adapters exist: `ardmav`, `halmstad`, `uav_smid`, `yolo_dir`.** NPS, ARD100, SMOT4SB,
Drone-vs-Bird and ExtremeTrack have **no adapter**. Either write one in `benchmarks/adapters/`
(the parse lives there; `prepare_data.py` is only the two builders and the report), or convert to a
YOLO layout first and use `yolo_dir`.

### 3.3 The legacy route — NPS and the combined multi-dataset corpus

`tools/make_dataset_external.py` is what produced rounds 4–7 and is still the only path to the
merged ARD-MAV + NPS + 07_05 corpus:

```bash
.venv/bin/python tools/make_dataset_external.py --task combined-tiled \
    --min-side 8 --tile 640 --stride-train 6
.venv/bin/python tools/make_dataset_external.py --task combined-gt
```

Three things to know about it:

* **`combined_splits()` now honours the published ARD-MAV split** — test =
  `phantom{05,08,09,10,19,30,41,43,46,47,58,63,65,70,86}`, val = `phantom{06,23,45,61,79}`, train =
  the other 40. It previously re-split by position (`test = i % 10 == 0`, 6 of 60), which put most
  of the official test videos into training. A `legacy=True` flag reproduces the old behaviour
  **only** to regenerate old artifacts. Never report a number produced with it. `dronedet/tests/`
  now pins this; it is a claim about what a number means, so it gets a test.
* **Its `--min-side` still defaults to 12, and it always inflates.** Pass `--min-side 8` for anything
  you intend to compare against a paper, or `--min-side 0` for true extent (the inflation is
  `max(side, min_side)`, so zero is a no-op).
* **NPS keeps a positional split either way**, because NPS publishes none. Say so, out loud, every
  time you quote an NPS number.

Both routes emit **native-resolution crops, not resized frames** — a full-frame pass shrinks a 9 px
drone to 8. Rough corpus size, so you can budget: ARD-MAV is ~107 k frames of which nearly all are
positive; at stride 4 that is ~27 k selected frames, and the tiler emits one positive tile per drone
plus one drone-free hard-negative tile → **≈ 54 k tiles**. At stride 6, ≈ 36 k.

**Read video sequentially, never seek.** The builders already do. Both of the repo's own source
videos hide their opening seconds behind an MP4 edit list that every decoder honours silently
(`10_06.mp4` is really 591 frames, not 361); `tools/recover_full_video.py` remuxes losslessly if you
need the head back.

---

## 4. Train

**The headline experiment is `trueextent_ardmav`: true extents + NWD.** Everything else in the
registry is context for it. The hypothesis is precise: the 12/24 px inflation was never about tiny
objects being hard, it was a prosthesis for IoU-based label assignment collapsing on few-pixel
boxes — which is exactly what a Wasserstein assignment metric is for. If it holds, this repo
reports a non-zero COCO AP for the first time. If it does not, the inflation is load-bearing and
every IoU comparison in the project stays capped — **which is also a result**, and is why the
control runs at the same three seeds.

### Run experiments through the registry, not the command line

Do not drive `tools/train_yolo.py` by hand. It sets no seed, records nothing about what produced a
checkpoint, and keeps its hyperparameters in argparse defaults that get overridden from shell
history — six months later `best.pt` is a file with a number attached and no way to establish which
code, which dataset build or which augmentation made it. Use the registry:

```bash
.venv/bin/python tools/train.py --list
.venv/bin/python tools/train.py --config trueextent_ardmav --dry-run
.venv/bin/python tools/train.py --config trueextent_ardmav --seeds 3
.venv/bin/python tools/train.py --config ardmav_headline --seeds 3      # a group: 2 experiments
```

Five things it does that a bare command line does not, each of which you would otherwise discover
too late:

* **The run is a named config** (`configs/experiments/`), so the hyperparameters are reviewable data
  rather than shell history — and `protocol_key` travels with them, so a label-inflated run compared
  against a published IoU number *automatically* reports that the inflation caps the comparison.
* **Seeds are set for python, numpy and torch, and the seed is in the run directory name**, because
  three seeds of one experiment are three different checkpoints, not three copies of one.
* **`work/runs/<name>-s<seed>/MANIFEST.json` is written before the first batch** — git SHA, dirty
  flag, a hash of the uncommitted diff, the fully resolved config, dataset file-listing hashes and
  measured label statistics, package versions, GPU, exact command line. A run that dies in epoch 3
  is still traceable; a manifest written at the end would not have been.
* **The dataset on disk is checked against the config's claim about it.** `min_side` decides what a
  label *means*, and the ARD-MAV builder writes both the inflated and the true-extent build into the
  same directory — so it samples the label files and refuses to start if the inflation on disk is
  not the inflation the experiment says it is testing. This is the single most valuable guard in the
  tool; it makes the headline 2×2 impossible to run backwards.
* **A VRAM guard warns before the run instead of after the OOM** (`--vram-gib`, default 8).

### The registry today

| config | what it varies | protocol | status |
|---|---|---|---|
| `baseline_ardmav` | the recipe rounds 5–7 actually used — P2 head, 640 px native tiles, labels inflated to `min_side 12`, no NWD — re-run against the **official** ARD-MAV split so its number can be compared with MGMD's 0.55 for the first time. Expected COCO AP ≈ 0: a 12 px label on a 6×3 px drone caps achievable IoU at ~0.13 before the detector does anything | `ardmav-official` | ready |
| ⭐ `trueextent_ardmav` | **the headline.** Identical to `baseline_ardmav` except `min_side 0` (true extents) and NWD on | `ardmav-official` | ready |
| `p2_no_p5_ardmav` | the Jul-2026 published edge recipe: add the stride-4 P2 head, delete the stride-32 P5 head, whose anchors can never fire on an 11.8 px median target. Same dataset as the baseline, so architecture is the only variable | `ardmav-official` | ready — but the model yaml is **hand-written and has never been instantiated**, which is why `expected_strides` is set and the tool aborts before the first batch if the head is not (4, 8, 16) |
| `temporal_stack_ablation_single` / `_stack` | the project's founding claim, run as a controlled A/B: RGB of frame *t* versus three ego-aligned grays at t−6/t−3/t, same labels, splits and schedule | `specklock-centre` | ready. Pooled over three datasets including this repo's own clip, so it is an **internal A/B and not comparable to any paper** |
| `birds_2class` | two-class drone/bird on UAV_SMID + Halmstad, so that a bird-attributed false-alarm rate exists to be quoted | `ap50` | ⚠ **BLOCKED** |

Groups: `ardmav_headline` (baseline + trueextent), `ardmav_all` (+ p2_no_p5), `temporal_stack_ablation`.

**`birds_2class` is blocked on three things and must not be run until all three exist**, because its
number would not be a benchmark result: `tools/make_dataset_birds.py` does not exist; nothing in this
repo can read Halmstad's MATLAB (mcos) labels or its `.xlsx` manifest; and **Halmstad publishes no
split**, so you must define *and commit* one or the number is not reproducible by anyone, including
you. There is also an unresolved decision inside it — whether UAV_SMID's helicopter/aeroplane/bomb
classes map to "bird" or are dropped. That is a decision, not an oversight; make it explicitly.

### Building what a config expects

Each config carries its own `build_command`, and `--dry-run` prints it. For the ARD-MAV pair that is
`tools/make_dataset_external.py --task ardmav-train-tiled` at `--min-side 0` or `12` — note that
this is the legacy builder, not `prepare_data.py`, and that both builds land in
`work/ext_datasets/ardmav_yolo_tiled/`, which is exactly why the on-disk inflation check exists.
Rebuild between the two arms of the 2×2, or the guard will stop you.

Three traps that will each cost you a day:

* **On our own 07_05/10_06 data there is no `min_side` flag at all**: `tools/make_datasets_v3.py`
  hard-codes `LABEL = 24.0` at module scope and writes it into every label line. Extending the
  headline experiment to our own clips requires editing that constant, and the edit must go in the
  run notes — it is the variable the whole experiment turns on.
* **The temporal channels are not colours.** A stack's three channels are stabilised grays at
  t−2Δ/t−Δ/t, so hue/saturation jitter does not change how a drone *looks*, it remixes *when* it
  was. The temporal configs use `NO_PHOTOMETRIC_AUG` and `ExperimentConfig.validate()` enforces it
  instead of trusting the CLI. Applying it to the single-frame arm too mildly handicaps that arm
  relative to how it would normally be trained — say so when you report the number.
* **Pass an absolute `project=`** if you ever fall back to `train_yolo.py`.
  `~/.config/Ultralytics/settings.json` redirects `runs_dir` out of the repo and nests relative
  paths under it, silently scattering runs on a fresh machine.

`dronedet/nwd.py` monkeypatches two ultralytics internals in place — `TaskAlignedAssigner.
iou_calculation` (the big lever: this is what gives tiny GTs positive anchors) and `BboxLoss.forward`
(smaller: keeps regression gradients alive at few-pixel scale). Both are *blends*, not replacements,
because this is a generalist spanning 3→100 px. Assigner boxes are in pixels (`nwd_assign_c` ≈ object
px size, default 16); loss boxes are stride-normalised (`nwd_loss_c` ≈ a few cells, default 2).

### Order, and time estimates

Run `ardmav_headline --seeds 3` first: it is the headline plus its control, at the sample size that
lets you report mean ± std instead of a point estimate. Then `p2_no_p5_ardmav`, then
`temporal_stack_ablation`, then unblock `birds_2class`. The three seeds are not optional — the
detection half has reported single-run point estimates while the pursuit half reported Wilson
intervals, and any Δ smaller than seed noise is not a result.

Time is an **ESTIMATE**, deliberately expressed as a procedure, because no wall-clock was ever
recorded for these runs and your GPU is not the one they ran on. Ultralytics prints the epoch time
after epoch 1 — take it and multiply by `epochs` (60 for every ARD-MAV config). As a sanity frame: a
`yolov8s-p2` epoch over ≈ 54 k 640 px tiles at batch 8 on an 8 GB laptop card is a coffee-to-lunch
unit, which made these overnight jobs. On a 24–48 GB card raise `--batch` to 32–64 for a roughly
linear win. Budget `ardmav_headline --seeds 3` (six runs) as most of a GPU-week at 8 GB, and about a
GPU-day at 48 GB.

If your first epoch is dramatically slower than that, check `torch.cuda.is_available()` before you
check anything else (§1.2).

---

## 5. Evaluate

Scoring is deliberately separated from inference. `tools/evaluate.py` takes **detection JSONs, not
weights**: producing detections is GPU-bound and model-specific; scoring them is cheap,
deterministic and needs nothing but numpy. That split is what lets you score a **rival's released
weights** under your own protocol without their training stack ever touching this code — which is
the whole answer to "you have never run the baseline".

### 5.1 Produce detections, then a scorecard

```bash
# 1. detections, one JSON per sequence (any producer: dronedet, tools/run_max.py, a rival's script)
.venv/bin/python -m dronedet detect \
    --video data/external/ard_mav/ARD-MAV/videos/phantom05.mp4 \
    --method moe3-stacked --out work/det/ardmav/phantom05.json

# 2. score them into a scorecard
.venv/bin/python tools/evaluate.py --dataset ardmav --model trueextent_ardmav \
    --gt work/prepared/ardmav/gt --dets work/det/ardmav \
    --official-split --weights work/runs/trueextent_ardmav-s0/weights/best.pt \
    --seed 0 --out work/scorecards/trueextent-s0.json
```

Sequences pair by **filename stem**. A GT file with no matching detection file is scored as a
**total miss, not skipped** — skipping would silently improve the number, which is exactly the kind
of quiet favour that makes a benchmark result untrustworthy. It warns on stderr when this happens.

The protocol defaults to the dataset's official one from the catalogue (ARD-MAV → IoU 0.25 on the
official 15) and falls back to `specklock-centre` otherwise; override with `--protocol`. Pass
`--targets` when some GT objects are distractors rather than positives.

Note the asymmetry, and keep it: **evaluation GT comes from `prepare_data.py` (true extent, always),
while the ARD-MAV training tiles come from the legacy builder** at whatever `min_side` the experiment
config declares. That is the correct arrangement — inflation is a training device, and an inflated
ground truth is not a device but a wrong answer.

Every strong detector here is **temporal and stateful over consecutive frames**, so a cold
single-frame call returns nothing on a frame the warm detector is 0.83 confident about. Produce
detections by running whole sequences, and read video sequentially.

### 5.2 What is in a scorecard, and why it is shaped that way

`benchmarks/scorecard.py`. The shape is the point — a scorecard that stored only summary numbers
would make every later question unanswerable without re-running the model.

| field | why it exists |
|---|---|
| `sequences: [SequenceResult]` | **per-sequence, never pooled.** The resampling unit for a paired comparison is the sequence: frames within one flight are strongly correlated. Pooling into one AP is how this project ended up reporting AP 1.000 with no spread from what was really a single flight |
| `detections: [(score, outcome)]` | **per-detection, not per-threshold summaries.** AP, P/R at *any* threshold and the bird false-alarm count at *any* threshold are all recomputable from this. Storing "bird hits at 0.55" freezes a choice that later turns out wrong |
| `outcome ∈ {tp, fp, distractor:<name>}` | a distractor is a real thing the detector must not call a drone. It never counts toward recall, and hits on it are **counted and reported** rather than discarded — `dronedet/evaluate.py` used to drop them silently, hiding the single result this project most wants to claim |
| `conditions` on each sequence | "works at night" is a claim about a *subset of sequences*, so subset membership has to be in the artifact, not reconstructed from memory later |
| `n_gt`, `n_frames`, `target_px_median` | sample size and difficulty travel with the number. A dataset-level AP next to a 39 px median target is a different claim from one next to 12 px |
| `git_sha`, `git_dirty`, `weights_sha256`, `command`, `created`, `seed` | provenance. Without these a scorecard is an assertion, not a measurement. **A scorecard with `git_dirty: true` is a draft** |
| `schema_version` | `load()` refuses a mismatched version rather than reading old data with new semantics |

### 5.3 The quick table, when you just want to look

`python -m dronedet bench` prints a markdown comparison directly from detection JSONs — centre AP,
AP by AI-TOD size bin, COCO AP, AP50, P, R, FP/frame, **confuser hits**, median centre error:

```bash
.venv/bin/python -m dronedet bench --gt work/gt_user.json \
    --dets work/det/pcmax.json work/det/moe3.json \
    --targets far --confusers bird --ci --out work/eval_bench.md
```

It is a look, not a result: it writes no scorecard, so nothing downstream can pair against it, and
with `--threshold` omitted it picks the best-F1 point **on the set it is reporting**, which is an
oracle rather than an achievable operating point. Take the threshold from a val run and pass it in.

### 5.4 Condition-stratified and bird-confusion output

Both come out of `tools/compare.py` reading scorecards (§6), because both are statements about a
*subset*, and the subset has to be in the artifact.

```bash
.venv/bin/python tools/compare.py --scorecards work/scorecards/*.json \
    --baseline baseline_ardmav --by-condition --threshold 0.55
```

**By condition** — `| model | condition | sequences | AP | recall | precision | n_gt |`. Read the
sequence count *before* the AP: a condition row with three sequences is a hint, not a result. If the
table says "no condition labels on these sequences", that is a **data gap, not a result** — the
adapter did not supply conditions, so nothing can be said about night/rain/fog, and you must not
imply otherwise. Conditions come from `--conditions` on `tools/evaluate.py` (per-sequence) or the
dataset's `Condition` tuple in the catalogue.

**Confuser rejection** — `| model | confuser hits | confuser instances | hits/instance | recall on
drone | 95 % CI on rate |`, defaulting to prefixes `bird plane airplane helicopter`. Two rules for
reading it:

* **Read it beside the recall column.** A model with zero confuser hits and 0.2 recall has not
  solved anything; it has stopped detecting.
* **The Wilson interval is on the hit rate**, and with a few hundred confuser instances it stays
  wide. That width is the honest state of the evidence, not a presentation problem.

If it reports no distractors, either the dataset has no labelled birds or the adapter dropped them —
and **no false-alarm claim is measurable** until that is fixed. That is precisely the Drone-vs-Bird
situation: birds present, unlabelled, for nine editions.

---

## 6. Compare

```bash
# ours vs ours — paired, on the same sequences, under the same protocol
.venv/bin/python tools/compare.py --scorecards work/scorecards/baseline-s*.json work/scorecards/trueextent-s*.json \
    --baseline baseline_ardmav --threshold 0.55 --out work/compare_extent.md

# ours vs published — no p-value is printed and none is possible
.venv/bin/python tools/compare.py --scorecards work/scorecards/trueextent-s0.json \
    --vs-published --dataset ardmav --out work/sota_ardmav.md
```

### 6.1 Ours vs ours — the only comparison that supports the word "significant"

Columns: `| model | AP | Δ vs baseline | 95 % CI on Δ | p (boot) | p (perm) | p (Holm) | seq wins |
verdict |`.

* The **resampling unit is the sequence**, and indices are resampled *jointly* for both methods —
  that is what makes it paired, and it removes the between-sequence variance the two methods share,
  so an easy video helping both does not count as evidence for either.
* **Two p-values on purpose.** The bootstrap inverts an interval; the permutation test attacks a
  null directly. When they disagree, that disagreement is information about how few sequences you
  have — report both.
* **Holm correction across the whole table**, because an ablation with eight rows is eight
  hypotheses and the best of eight raw p-values is not a finding.
* **`seq wins`** (McNemar on per-sequence wins) answers "is it better more *often*", which disagrees
  with AP exactly when one long sequence carries the result. Worth knowing before you write a
  sentence about it.
* A verdict of **better/worse** requires the CI to exclude zero *and* the Holm-adjusted p < 0.05.
  Anything else prints **no difference**.

### 6.2 Ours vs published — the rule, and what is refused

Columns: `| method | their number | their protocol | comparable? | verdict |`.

`Protocol.mismatches_with` **derives** comparability from the two protocol objects — matcher
(`iou`/`centre`), AP style (`coco`/`ap50`/`ap25`/`voc-all-point`), IoU threshold or centre radius,
split name, and label inflation. It is not remembered by whoever writes the table, because that is
how every apples-to-oranges comparison in this repo's history happened.

* If the protocols differ the row is marked **❌ NOT COMPARABLE and the difference is not shown at
  all.** Not shown with a caveat — not shown.
* If a split is missing on either side, that alone makes the comparison **unverifiable** and the row
  fails. A pooled number and a single-clip number are not the same measurement even under an
  identical matcher.
* An IoU comparison where either side used inflated labels is flagged regardless of anything else,
  because the inflation caps achievable IoU independently of detector quality (§3.1).
* **No p-value appears and none can.** A published AP is one scalar with no distribution behind it.
  What is printed is: our interval, their point estimate, whether ours covers it, and every protocol
  difference. **"Indistinguishable" means our interval covers their point estimate — not that a test
  was run.**

The bars, from `benchmarks/published.py`, each stored with its protocol:

| dataset | method | number | protocol |
|---|---|---|---|
| ARD-MAV | **MGMD / GLAD** | AP@0.25 **0.55** | IoU 0.25, official 15-video test split |
| ARD100 | YOLOMG-1280 | AP@0.5 0.85 (0.78 @640) | IoU 0.5 |
| NPS | TransVisDrone / YOLOMG | AP@0.5 0.95 | IoU 0.5, **no official split** |
| Drone-vs-Bird | Laroca et al. (1st, 8th DvB) | mean mAP@0.5 0.7390 | IoU 0.5, self-chosen 7-video val |
| DUT Anti-UAV | Lightweight YOLOv11 (P2, no P5) | mAP@0.5 0.922 | IoU 0.5 |

Entries carrying `reported_by_competitor=True` — notably **TransVisDrone 0.15 on ARD100, which comes
from YOLOMG's authors, not from TransVisDrone's** — print with a ⚠ and are excluded from
`best_for_dataset()`. They are the least reliable class of number in this literature. This repo once
reproduced that 0.15 in bold as evidence of a rival collapsing. Do not repeat that.

---

## 7. What NOT to claim

Blunt list. Each item is something the repository has previously asserted or could plausibly assert,
and each is refuted by a measurement in [verified-measurements-2026-08.md](verified-measurements-2026-08.md)
or the [internal audit](internal-audit-2026-08.md).

1. **Any ARD-MAV number from rounds 5–7.** `combined_splits()` ignored the official split it defined
   168 lines above and re-split by position (`test = i % 10 == 0`, 6 of 60), so most of the official
   15 test videos were in training. The 0.836 cannot sit in a table beside MGMD's 0.55. Fixed on this
   branch and pinned by a test; anything produced before the fix, or with `legacy=True`, is void.

2. **Anything at all from `phantom16`.** Median target **39.1 px** — rank 57 of 59 videos, the 97th
   percentile, against a dataset median of 11.2 px and an official-test median of 12.1 px. That is
   3.5× the benchmark median and lands in the AI-TOD *medium* bin. "ARD-MAV AP 0.994" is the
   third-easiest video in the dataset and does not test this project's thesis at all. The cherry-pick
   is provable from the report's own baseline column: the round-5 checkpoint already scored 0.992 on
   phantom16 versus 0.836 pooled, so ~0.156 of the 0.84 → 0.994 "improvement" is clip choice.

3. **Any COCO AP or IoU AP from a 24 px-label model.** Not "low" — **structurally 0.000**. Max
   achievable IoU is 0.111 by arithmetic and 0.111 by measurement. Report centre-distance AP with
   the τ stated, and only quote an IoU number from a true-extent model (§3.1, `trueextent_ardmav`).

4. **"24/24 intercepted, 0 buildings hit"** without naming the sensor. That is
   `detector: "oracle"` — the simulator's own bounding box, zero degradation, zero latency. The same
   mission with the shipped seeker is **0/3, all three buildings struck**. On the chase suite it is
   **33/33 oracle vs 28/33 real**. Both numbers are true; the sensor belongs in the claim everywhere
   the number appears.

5. **"Temporal track evidence separates birds from drones", stated as a result.** The measurement is
   real and strong — 99.8 % drone recall with **0 hits on 934 bird instances** at 0.002 FP/frame,
   where the raw detector takes 151 bird hits at 95 % recall — and it rests on **8 bird tracks, one
   flock, one video, one afternoon.** That is an existence proof. Halmstad turns it into a result.
   Two further scopes: say **rotary-wing** (flapping-wing drones defeat the cue by construction), and
   do not claim the mechanism is unexploited — a Jun-2026 NCAA paper independently measures +22 %
   frame-wise accuracy from trajectory features and a Jan-2026 optical method reports 99.47 %. What
   is genuinely unclaimed is doing it *inside the detector's input representation* and *publishing
   the bird-attributed false-alarm rate*.

6. **"AP 1.000 with zero false positives."** The AP survives audit and reproduces. The operational
   claim does not: the PC-MAX run emits **three additional sustained `[drone]` tracks**, each ~150
   frames (5 s), at median distances of 133–212 px from the real drone. AP is 1.000 anyway because
   their scores fall below the best-F1 threshold and the metric drops them. Report tracked results as
   coverage plus false-track count; reserve AP for the per-frame detector.

7. **"Unseen test video."** No dataset builder ever reads `10_06.mp4` — the weights are genuinely
   held out, and the crude leakage charge is false. But six constants in `dronedet/trackclass.py`
   were set against it, and the augmentation policy in `make_datasets_v3.py` explicitly targets "the
   10_06 foliage-crossing misses". The honest phrase is **development test set**.

8. **"From a moving camera", for 07_05 or 10_06.** Measured with the repo's own stabiliser: 10_06
   drifts **0.76 px in x and 1.07 px in y across the entire clip**. That is a fixed rig, static to
   sub-pixel. The moving-camera evidence is the ARD-MAV / NPS row and nothing else.

9. **"74 fps, edge."** That is an RTX 5070 **Laptop** discrete GPU with a TensorRT FP16 engine. The
   only Jetson figure that exists is a **projection** of 10–15 fps @1280, explicitly labelled as not
   a measurement. EDGE-RT has never run on an edge device.

10. **A single AP as a benchmark result.** The whole detection half rests on **2 videos, 1 drone
    each, 885 scored instances, 2 independent flights**. A 30-frame block bootstrap on 10_06 gives
    ~12 independent looks and a CI of [1.000, 1.000], which means "perfectly consistent within this
    one flight" and says nothing whatever about a second flight. Put *n*, the sensor and the hardware
    on every headline number.

11. **Cross-paper NPS comparisons**, and **any Drone-vs-Bird number presented as "the" benchmark
    score.** NPS publishes no official split; DvB withholds test annotations, so every published DvB
    number in nine editions is on a self-chosen validation split. Three such numbers side by side are
    three different experiments.

12. **"Significantly better than [published method]."** There is no second sample. No test exists.
    Say: our interval, their point estimate, whether ours covers it, and every protocol difference.

Negative results already measured — do not spend GPU time re-deriving them: super-resolution on crops
(no gain over bicubic), plain fine-tuning without the tiny-object recipe (AP 0.023), INT8 on a
desktop GPU (slower *and* worse than FP16), DT=9 temporal stacks, a full-frame edge stabiliser, and
feeding this repo's shipped temporal detectors to the pursuit renderer (recall 0.50 with 11 k false
positives — a chaser translating at 14 m/s has no static background to cancel).

---

## Appendix — the loop, in one screen

```bash
.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e ".[train,test]"
.venv/bin/python -m pytest -q                                   # expect at least 801 passed
.venv/bin/python tools/fetch_data.py --priority 2               # exit 2 = a human must act
.venv/bin/python tools/dataset_stats.py --dataset ardmav --per-video
.venv/bin/python tools/prepare_data.py ardmav --out work/prepared/ardmav --tile 640 --stride 4
.venv/bin/python tools/train.py --config ardmav_headline --dry-run   # prints each build_command
.venv/bin/python tools/train.py --config ardmav_headline --seeds 3
.venv/bin/python tools/evaluate.py --dataset ardmav --model trueextent_ardmav \
    --gt work/prepared/ardmav/gt --dets work/det/ardmav --official-split \
    --out work/scorecards/trueextent-s0.json
.venv/bin/python tools/compare.py --scorecards work/scorecards/trueextent-s0.json \
    --vs-published --dataset ardmav
.venv/bin/python tools/check_docs.py                            # before you commit any prose
```
