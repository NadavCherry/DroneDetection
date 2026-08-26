# Why the paper reports 0.95 on NPS-Drones and we measure 0.527

**Status: investigation in progress.** Sections 1-6 are complete and measured. Section 7
(the leak-split retrain) is running; its result is the last quantitative piece and this
document will not claim a total until it lands.

---

## The question

YOLOMG (arXiv:2503.07115) reports **AP 0.95** on NPS-Drones. Trained by us from their
code, on their recipe, and scored by our unified evaluator, the same method reaches
**0.527**. Our own method reaches 0.487 on the same protocol.

A 0.42 gap is not a detail. It decides whether our README may place our 0.487 next to
their published 0.95 at all — and the answer, established below, is that it may not,
because those two numbers are not measurements of the same thing.

The investigation was run as a sequence of eliminations. Each one is a measurement, not an
argument, and one of them refuted the hypothesis this project started with.

---

## 1. Ruled out: annotation version

The paper states it uses the *clean version* annotations released by Dogfight — the same
annotations we use. Not a source of difference.

## 2. Ruled out: truncated or missing ground truth

Dogfight and Purdue v1 annotations agree to within 3 boxes over 5 clips (4,244 vs 4,241).
The gaps are **interior** (304, 265, 472, 431, 281 frames) rather than trailing: the drone
genuinely leaves frame and returns. Not a labelling artefact.

## 3. Ruled out, and this refuted our leading hypothesis: the evaluator

The hypothesis this investigation began with was that the two numbers differ because the
metrics differ. `tools/protocol_sweep.py` tests it properly: it takes **one fixed set of
detections** and re-scores it under every combination of five protocol axes, so each
choice's contribution is measured rather than asserted. Nothing about the model changes
between rows; only the counting rules do.

| axis | our value | their value |
|---|---|---|
| frames scored | every frame of the test video | annotated frames only |
| AP integration | all-point | 101-point COCO interpolation |
| matcher | IoU | IoU |
| confidence floor | 0.001 | 0.001 |
| aggregation | pooled over sequences | pooled over sequences |

Our 101-point implementation reproduces YOLOv5's `utils/metrics.compute_ap` bit-exactly
(verified against it on random PR curves), so this is their arithmetic, not an
approximation of it.

**Result — the total protocol effect, over 6 model/seed combinations:**

| run | our evaluator | their protocol | delta |
|---|---|---|---|
| YOLOMG NPS seed 0 | 0.4965 | 0.5047 | **+0.0082** |
| YOLOMG NPS seed 1 | 0.5351 | 0.5401 | **+0.0050** |
| YOLOMG NPS seed 2 | 0.5485 | 0.5536 | **+0.0052** |
| ours NPS seed 0 | 0.4819 | 0.4938 | **+0.0118** |
| ours NPS seed 1 | 0.5441 | 0.5542 | **+0.0101** |
| ours NPS seed 2 | 0.4347 | 0.4492 | **+0.0145** |
| YOLOMG ARD-MAV seed 0 | 0.8425 | 0.8431 | **+0.0007** |

**The evaluator explains about +0.01 of a +0.42 gap.** The hypothesis is refuted. Whatever
produces the paper's number, it is not the choice of AP definition, the frame set, the
matcher, the confidence floor, or the aggregation rule.

Two side observations from the full grid, recorded because they are large enough to matter
elsewhere even though they are not their protocol:

* **Per-video aggregation instead of pooling is worth +0.095** (0.4965 → 0.5913 on YOLOMG
  seed 0), and +0.127 when combined with scoring annotated frames only. Averaging per-video
  APs weights a 300-frame clip like an 1,800-frame one. Their val.py pools, so this is not
  the explanation here — but it is a bigger lever than every axis they do differ on
  combined, and any future comparison must state which it uses.
* On **ARD-MAV the protocol effect is +0.0007**, and our trained YOLOMG scores 0.8425 there
  against GLAD's published 0.80. Our evaluator is therefore not generally harsh, and our
  YOLOMG training is not generally weak. **The problem is specific to NPS.**

## 4. Their own training run does not reach 0.95 either

The most direct check available, and it needs no evaluator of ours at all: what did
YOLOMG's **own** training loop report, using its **own** `val.py` and its **own** 101-point
AP, on a video-disjoint validation split?

| run | epochs | best val mAP@0.5 | mAP@0.5:0.95 | P | R |
|---|---|---|---|---|---|
| YOLOMG NPS seed 0 | 100 | **0.790** | 0.374 | 0.755 | 0.900 |
| YOLOMG NPS seed 1 | 100 | **0.788** | 0.382 | 0.714 | 0.875 |
| YOLOMG NPS seed 2 | 100 | **0.809** | 0.401 | 0.773 | 0.865 |
| YOLOMG ARD-MAV seed 0 | 100 | 0.876 | 0.548 | 0.912 | 0.790 |
| YOLOMG ARD-MAV seed 1 | 100 | 0.887 | 0.564 | 0.911 | 0.790 |
| YOLOMG ARD-MAV seed 2 | 100 | 0.878 | 0.554 | 0.902 | 0.780 |

Under their code and their metric, a **video-disjoint** NPS split yields **0.79**, not 0.95
— and 0.79 is itself an optimistic figure, because it is the best epoch selected by
maximising that very number.

## 5. Which videos you hold out is worth more than every metric choice combined

Same weights, same code, same metric — only the held-out videos differ:

| held-out set | clips | their metric |
|---|---|---|
| validation | 37-40 | **0.790** |
| test | 41-50 | **0.505** |

**+0.285 from the choice of held-out videos alone.** The NPS test clips are substantially
harder than the validation clips. This is not misconduct by anyone; it is a property of the
dataset that any single-number comparison hides, and it is 28× larger than the entire
protocol effect measured in section 3.

## 6. Their pipeline has no working held-out test path at all

Two independent structural findings, both verified by execution rather than reading.

### 6a. Their split script cannot produce a test set

`third_party/YOLOMG/data/split_train_val.py`:

```python
trainval_percent = 1.0
train_percent    = 0.85
total_xml = os.listdir(xmlfilepath)          # a FLAT list of every frame image
num  = len(total_xml)
tv   = int(num * trainval_percent)           # == num
trainval = random.sample(list_index, tv)     # therefore EVERY index
...
for i in list_index:
    if i in trainval:   ...                  # always true
    else:               file_test.write(name)   # never reached
```

Run unmodified on our 9,087 NPS frames (`tools/sota/demo_their_split.py`):

```
test.txt : 0 lines          <- their held-out test set
train.txt: 7723 lines  from 50 distinct videos
val.txt  : 1364 lines  from 50 distinct videos   <- the SAME 50
```

Two facts, both confirmed by running their code:

1. **`test.txt` is empty.** `trainval_percent = 1.0` sends every index into `trainval`, so
   the branch that writes the test list never executes. Any number their pipeline reports
   is the 15 % `val.txt` slice.
2. **The partition is per-frame, not per-video.** `total_xml` is a flat `os.listdir` over
   every frame of every video, and `random.sample` shuffles it. Train and validation draw
   from the same 50 flights: frame *t* can train while frame *t+1* validates. On NPS, where
   the camera and target move a few pixels between sampled frames, a validation frame is
   very nearly a training frame with noise.

### 6b. Their `val.py` cannot evaluate a test split even if one existed

`val.py` takes two streams — RGB and motion mask — selected by `task` and `task2`:

```python
task2 = task2 if task2 in ('train2', 'val2', 'test') else 'val2'
dataloader = create_dataloader(data[task], data[task2], ...)
```

`task2` defaults to `'val2'` in `run()`'s signature, and **there is no `--task2` command
line flag** (`grep -c "add_argument.*task2" val.py` → `0`). `main()` calls
`run(**vars(opt))`, so the default always applies. Consequently `val.py --task test` reads
RGB frames from the **test** split and motion masks from the **validation** split.

This is not theoretical. Running it crashes:

```
RuntimeError: Sizes of tensors must match except in dimension 1.
Expected size 992 but got size 736 for tensor number 1 in the list.
```

— the two streams are letterboxed to different shapes because they are different videos at
different resolutions. There is no path through their code that evaluates a held-out test
set.

## 7. The decisive experiment: does the per-frame split reproduce 0.95?

**Running now.** Three arms, each identical to a run already on disk in every respect —
same images, same labels, same masks, same architecture, same 100 epochs at 1280 px, same
initial weights — differing **only** in which frames are called train and which val:

| arm | pool | train / val | clips on both sides |
|---|---|---|---|
| `yolomg_leak_ctl` | clips 1-40 only | 7,080 / 1,250 | 40 / 40 |
| `yolomg_leak_all` | all 50 clips (their script exactly) | 7,723 / 1,364 | 50 / 50 |
| `ours_leak_ctl` | clips 1-40, grouped by frame | 12,850 / 2,267 | 40 / 40 |

`yolomg_leak_ctl` is the controlled arm: it re-partitions **only the clips the disjoint run
already had**, so it sees no new footage and in fact trains on ~12 % *fewer* images (7,080
vs 8,019). If it still scores higher, the partition rule is the only thing left to credit.

`ours_leak_ctl` applies the same manipulation to our own method, so the finding cannot be
read as "the competitor cheats" when what it actually shows is "this protocol inflates
everyone". Our tiles are grouped by source frame before shuffling, so both arms measure
exactly one thing: frame-level leakage.

Reference points to compare against, all measured, all from the same weights and code:

| | their metric |
|---|---|
| video-disjoint test (clips 41-50) | 0.505 |
| video-disjoint val (clips 37-40) | 0.790 |
| per-frame random val (leak) | *pending* |
| paper | 0.95 |

### 7a. The first arm landed, and it does NOT support a large leak effect

`ours_leak_ctl` finished first (it is a 640 px model, not a 1280 px dual-stream one). Its
own trainer's validation metric, against the same metric on the video-disjoint runs:

| arm | validation set | epochs | val mAP50 |
|---|---|---|---|
| ours, **video-disjoint** | clips 37-40 | 30 | 0.932 |
| ours, **video-disjoint** | clips 37-40 | 100 | 0.941 |
| ours, **leak** (random 15 % of clips 1-40) | mixed | 30 | **0.965** |

**At matched epochs the leak is worth +0.034.** Not +0.4. For our method, on tiles, a
per-frame partition is a small effect, and the leading hypothesis is again not carrying the
weight put on it.

Two cautions, both of which cut against reading more into this than it says:

* This arm ran **30 epochs, not 100** — `cluster/leak_train.sbatch` omits `--epochs` for the
  `ours` branch, so the registered config's default applied. The leak arm therefore had
  *less* training than the 100-epoch baseline and still scored higher, which is the
  conservative direction, but the matched row is the 30-epoch one.
* Our tiles are cropped around targets and grouped by source frame, so a "random split"
  here leaks less than a random split of *whole frames* in a full-frame pipeline. YOLOMG is
  full-frame at 1280 px, and its leak arms are still training. **They are the decisive
  test, not this one.**

### 7b. The gap this exposed instead: our own val metric does not predict our test result

| ours, 100 epochs | value |
|---|---|
| trainer's val mAP50, 640 px tiles, clips 37-40 | **0.941** |
| our evaluator's AP, full-frame tiled inference, clips 41-50 | **0.487** |

These are different quantities — a tile-level metric on one set of clips against a
full-frame metric on another — so the difference is not a bug, and neither number is
wrong. It is a warning about reading either one as "the" accuracy. A tile-level validation
score near 0.94 sits beside a full-frame test AP near 0.49 for the same weights, and any
paper reporting only the first would look like a much stronger result than the same system
scored the second way. **That is worth knowing regardless of how the YOLOMG arms resolve**,
and it is a mechanism no protocol sweep would have found.

---

### 7c. The decision rule, fixed before the data arrives

The YOLOMG leak arms are still training. This section is written **now**, at epoch ~24 of
100, so that the verdict is not chosen after seeing where the curve lands. What follows is
the rule; the next revision of this document reports the outcome against it and nothing
else.

**The comparison.** Best validation mAP@0.5, their own trainer, their own metric:

| arm | pool | seeds | best val mAP@0.5 |
|---|---|---|---|
| video-disjoint | train 1-36, val 37-40 | 0, 1, 2 | 0.790 / 0.788 / 0.809 |
| leak, controlled | random 85/15 over clips 1-40 | 0 | *pending* |
| leak, faithful | random 85/15 over clips 1-50 | 0 | *pending* |

The disjoint band is **0.788-0.809**, a spread of 0.021 across three seeds. The leak arms
are one seed each and are two different pools, not two seeds, so they are two independent
data points rather than a sample with a variance.

**The rule.**

* If both leak arms land **inside 0.788-0.809**, per-frame leakage contributes
  approximately nothing, and we say so. The gap stays unexplained by leakage.
* If either lands **above 0.83** — more than one disjoint spread clear of the band — the
  leak is a real contributor and we quantify it as `leak_best - 0.796` (the disjoint mean).
* **Between 0.809 and 0.83** is the honest inconclusive zone: one seed cannot separate a
  0.02 effect from seed noise, and we report it as such rather than rounding it into a
  conclusion in either direction.
* Reaching **~0.95** would mean the leak explains the published number outright.

**Two things that would invalidate the comparison**, to be checked before reading it:

* an arm that stops short of 100 epochs — the disjoint numbers are 100-epoch bests, and
  our own `ours_leak_ctl` arm already ran 30 epochs instead of 100 because the sbatch
  omitted `--epochs`. Check `results.csv` line counts first.
* a best epoch at or near the final epoch, which would mean the run had not converged and
  the ceiling is unmeasured rather than measured.

**The early read, recorded so it cannot be quietly forgotten.** At matched epoch 20 the
leak arms sit *inside* the disjoint band: 0.6950 (ctl) and 0.6799 (all) against disjoint
0.6934 / 0.5834 / 0.7166. That points at the first outcome. It is not the verdict —
memorisation can appear late in training, which is exactly why the arms are being run to
100 epochs instead of being called here.

## Running total

| reason | measured impact | status |
|---|---|---|
| annotation version | 0 | ruled out |
| truncated ground truth | 0 | ruled out |
| AP definition, frame set, matcher, conf floor, aggregation | **+0.010** | measured, refutes the leading hypothesis |
| which videos are held out (test 41-50 vs val 37-40) | **+0.285** | measured |
| whether videos are held out at all (per-frame leak) | *pending* | running |
| *(not their protocol, but larger than all of the above: per-video instead of pooled aggregation)* | *+0.095* | measured |

## What this already licenses us to say

Regardless of how section 7 resolves, three things are established:

1. **Our 0.487 and their published 0.95 are not the same quantity and must never be
   subtracted.** `dronedet.Protocol.mismatches_with` already refuses this in code; this
   document is the evidence for why that guard exists.
2. **Our reproduction of YOLOMG is competent, not crippled.** It beats GLAD's published
   ARD-MAV number (0.842 vs 0.80) under a protocol whose total distortion there is +0.0007.
   Where we report YOLOMG beating us, we are reporting a genuine loss to a working baseline.
3. **The published NPS protocol has no held-out test set, by two independent mechanisms** —
   an empty `test.txt` and a `val.py` with no `--task2` flag. A number produced under it
   describes performance on frames interleaved with training frames from the same flights.

## Reproducing this

```bash
PYTHONPATH=. python tools/protocol_sweep.py \
    --gt work/ext_datasets/gt/nps --dets work/det/nps/yolomg_nps_seed0 \
    --label "YOLOMG seed 0"

PYTHONPATH=. python tools/sota/demo_their_split.py \
    --names-from work/ext_datasets/yolomg_nps

sbatch cluster/splitfix.sbatch          # builds the leak splits (CPU)
sbatch cluster/leak_train.sbatch        # the three arms of section 7 (3 GPUs)
```
