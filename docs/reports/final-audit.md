# Final audit — scientific status, and a hostile re-read before publication

This is item #12: the closing report on the twelve-item programme that took this repository
from research state to a publishable one. It states what the project has established, what it
has not, and what a hostile reviewer finds when every claim is checked against the artefact
that produced it.

**Verdict: the science is in better shape than the prose. The launch package must not be
published as drafted.** Five specific things are wrong with it, listed in §2. None of them is
a measurement error — every point estimate in this repository reproduced. They are claims
that outrun their evidence, and a priority claim that belongs to someone else.

---

## 1 · How this audit was run, and how far to trust it

Nine claim areas, each audited against its artefacts, then attacked by two adversarial
reviewers with different lenses (statistics/design, and protocol/wording), then every
surviving objection independently checked by two verifiers instructed to default to refuting.
82 agents, 2,629 tool calls, zero errors.

**A caveat about that process, stated because it matters.** Of 54 objections raised, the top 3
per area (27) went to verification and **27 of 27 were upheld, 0 refuted**. A 100 % survival
rate through an adversarial stage is a warning sign, not a triumph — it suggests the verifiers
were insufficiently sceptical. So the load-bearing findings below were re-checked by hand,
directly against the artefacts, and each one marked **✔ verified independently** was confirmed
that way rather than taken from an agent. The unmarked ones should be treated as strong leads,
not settled facts.

The 27 objections were also capped at 3 per area by severity; 27 more were raised and never
verified. This audit is not exhaustive.

---

## 2 · What blocks publication

### 2.1 The core mechanism is prior art, and this repository knows it ✔ verified independently

`docs/research/datasets-and-benchmarks-2026.md:414` records:

> The closest published analogue is **Temporal-YOLOv8** (TNO, Sensors 2024): *"instead of using
> a single video frame as input, multiple frames are stacked from different time steps"* — the
> identical mechanism, published **two years earlier**, with a near-identical jump (**0.465 →
> 0.839 mAP**). … **This must be cited.**

It is cited in exactly two internal research files and **on no reader-facing surface**: not
`README.md`, not `docs/index.html`, not `CITATION.cff`, not the launch package. The README has
no related-work section at all.

The launch package's headline — *"Stack three moments as colour channels and it isn't"* — claims
the contribution is the stacking. On the evidence in this repo, it is not. The same file names
what is actually defensible: **ego-motion stabilisation before stacking, the specific t−12/t−6/t
lags, centre-distance scoring at few-pixel scale, and the track-level announce rule.** That is a
narrower and still real contribution, and it is the one the post should make.

### 2.2 The headline ablation is confounded ✔ verified independently

`README.md:71`, `work/ablation/REPORT.md:9` and the post all say the 0.159 → 0.895 pair holds
**"same training corpus"**. It does not. `realtime/tools/make_datasets_rt.py:239` gates the
copy-paste augmentation block behind `if temporal:`, and `:242` pastes 4 drone + 3 bird extra
labelled instances into every *training* frame. The single-frame arm never enters that branch.

The repository credits that same augmentation with most of the gain — `realtime/README.md:57`:
*"added copy-paste with per-channel velocity trails to the full-frame dataset (mAP50 0.48 →
0.70) → test AP 0.64."*

Everything else about the control is sound, and stronger than claimed: both checkpoints decode
to identical `train_args` — same `yolov8n-p2.yaml`, epochs, batch, imgsz, seed, augmentation
and pretrained weights, differing only in the `data:` path. The number reproduces byte-identically
from `tools/ablation_temporal.py`. **The two arms differ in input representation and in training
data, so the gap cannot be attributed to the representation alone.** A velocity trail cannot be
pasted into a single RGB frame — but a static drone can, so the control was runnable and was not
run. The honest sentence names the confound.

### 2.3 The bird result is measured on the training video ✔ verified independently

`configs/experiments/local_video.py:1` — *"The project's own two videos: **train on 07_05**,
test on 10_06."* All 934 labelled bird instances are on 07_05.

`docs/launch/linkedin.md:56` says: *"**The test video** carries eight hand-labelled bird tracks."*
That is false. It is the training video, and it is the only video that supplies bird supervision.
`README.md:181` names the video without stating its role, which is better but still not enough
for a 🟢 demonstrated mark.

### 2.4 The moving-camera framing is not supported ✔ verified independently

`README.md:5`, the repository's first sentence, and the post's framing paragraph both rest on
the camera moving. `docs/research/datasets-and-benchmarks-2026.md:416`:

> **The moving-camera framing is not supported by the two flagship numbers.** `10_06` drifts
> **0.76 px in x, 1.07 px in y across the entire clip**… That is a **static rig to sub-pixel
> precision**… *Say this plainly before someone else does.*

It was never said plainly. The word "moving" appears once in the README, bolded, at line 5;
"near-static" appears zero times there, in `docs/index.html`, or in the post — while four
internal round reports classify both local videos as near-static. The genuinely moving-camera
corpus is ARD-MAV/NPS, which is precisely where the thesis stops separating (§3).

### 2.5 The retraction never reached the images or the shipped package ✔ verified independently

The project retracted *"AP/F1 1.000 on unseen real video; 24/24 intercepted"*. That reached
`README.md` and the `<meta>`/`og:` tags. It did not reach:

| surface | still says |
|---|---|
| `docs/media/social_card.svg:21` → the `og:image` a shared link renders | "1.000 … unseen real video"; "24 / 24" |
| `docs/index.html:243` (hero stat) | "1.000 … never trained on and never used to pick a model. Zero false positives." |
| `docs/gallery.html:116-117` (linked from `README.md:14`) | "on the **unseen test video** … AP/F1 = 1.000 · zero false positives" |
| `docs/guides/methods.md:93,:100` | "1.000, zero false positives on the unseen test video" |
| `final/README.md:17`, `:33`, `:49` — the **shipped package** | "on the unseen test video: tracked AP/F1/R/P = 1.000, zero false positives" |
| `docs/index.html:491` | "**74 fps** … **107–122 fps**" — the literal 100+ fps figure `README.md:171` retracts with *"Nothing reached three figures."* |
| `docs/media/architecture_{system,pcmax,edgert}.svg` | "AP / F1 = 1.000"; "24 / 24 intercepted", no sensor named |
| `docs/media/chart_cpa.svg` + `tools/make_result_charts.py:113` | "24 / 24 intercepted, from every direction" |
| `NOTICE.md:56-62` (the licence file) | "the test video never trained on and never used for model selection" |

Against `README.md:412` — *"`10_06` is a development set, not an unseen one"* — and
`work/reports/tracks/pcmax_1006.json`, which records `track_precision` 0.2 with four sustained
false drone tracks.

The `chart_cpa` case is the worst of them: the sentence is **hard-coded in the generator**, so
`README.md:369`'s own documented command regenerates the retracted claim.

---

## 3 · Scientific status

### What is established

- **A representation change large enough to matter on this corpus.** AP 0.159 → 0.895 on
  `10_06`, reproducing byte-identically, on identical network/hyperparameters/seed — subject to
  the augmentation confound in §2.2 and the sample size (1 video, 1 drone, 337 boxes).
- **Competitive standing against a retrained competitor**, scored by one evaluator on our splits:
  ARD-MAV 0.809 vs YOLOMG 0.834; NPS 0.487 vs 0.527.
- **A size-dependent crossover in the means.** Below 10 px we lead (+0.083 at <8 px, +0.095 at
  8–10 px); above 16 px YOLOMG leads and *that* side is significant on 3/3 seeds.
- **Track-level rejection of birds**: 0 raised over 934 labelled instances, from 440 detections
  that land on them — on the training video (§2.3).
- **Speed measured jointly with accuracy**: 58.9 fps at AP 0.876, TensorRT engine @1280 on an
  RTX 4090; 35.2 fps for the `.pt` fallback a fresh clone actually runs.
- **Three negative results held at full strength**: dt = 6 is not established on held-out test;
  the 100+ fps model did not reproduce; ~0.109 of the NPS discrepancy is unexplained.

### What is not established — and is missing from §6

The direct test of the project's thesis was run, and **`README.md` §6 does not contain it.**
From `work/reports/SUMMARY.md`, temporal vs. this project's own single-frame control:

| corpus | rows | result |
|---|---|---|
| ARD-MAV (both budgets, 3 seeds) | 6 | **6 × "no difference"** |
| NPS (both budgets, 3 seeds) | 6 | **2 × "worse", 4 × "no difference", 0 × better** |

So: *the representation's advantage is demonstrated on the static-rig corpus; on both
moving-camera benchmarks the paired tests cannot separate it from a single frame, and on NPS it
is twice significantly worse.* This is stated correctly at
`docs/reports/round8-sota-campaign.md:155`, and nowhere a reader will meet it — not §6, not §12,
not the site, not the post.

Related: `work/reports/size_curve/ardmav_mission.json` carries an `ours-single` arm, but all 15
of its `paired` rows compare `ours` vs `yolomg`. **README §4's third column has no paired test
behind it at all.** ✔ verified independently

### A claim that is wrong in the project's own disfavour ✔ verified independently

`README.md:121` — *"They lead on both public benchmarks, and **their lead is the one that
reaches significance**."* The overall paired tests say otherwise: **0 of 6 ARD-MAV rows** reach
significance (p_perm 0.17 / 0.58 / 0.23 at 100 ep), and 1 of 6 on NPS. That sentence is true of
the **size bins** (16–25 px and >25 px, 3/3 seeds) but sits directly beneath the overall table.

By the project's own both-tests-must-agree rule, overstating a defeat is the same error as
overstating a win. The correct statement: *they lead on point estimates; the difference does not
reach significance on ARD-MAV, and does on 1 of 3 NPS seeds.*

### The matching rule contradicts the stated convention ✔ verified independently

`README.md:32` declares, as a convention the whole file depends on: *"Matching is by centre
distance (τ = 12 px), not IoU… our numbers and theirs are **not the same quantity** and are never
subtracted."* `README.md:127` repeats it: *"ours are centre-distance."*

But `benchmarks/protocol.py:130` defines `ARDMAV_GLAD` as `matcher="iou", ap_style="ap50",
iou_threshold=0.5`, and `work/reports/size_curve/*.json` record `rule='iou', iou=0.5`. **Every
number in §4, §5 and §6 is IoU@0.5.**

The truth is *better* than the claim — the protocol's own note says producing IoU is "what makes
our number and theirs the same kind of number" — but §1 governs the whole file and is wrong
about three of its sections.

---

## 4 · The YOLOMG / NPS discrepancy

The attribution stands as an approach and is the strongest investigation in the project: 0.505
test → +0.291 video selection → +0.045 leakage → +0.010 convention, ~78 % accounted for, ~0.109
left explicitly unexplained. Two defects found:

- The **+0.291** term subtracts a one-seed number from a three-seed mean. The seed-matched value
  computable from data already in the document is **+0.263**.
- The **+0.045** leakage term is called a real contributor on the strength of a single run
  crossing a hand-set threshold by 0.011 — no bootstrap, no permutation test, no interval. It is
  the one headline claim in the repository exempt from its own two-test standard.

Neither changes the conclusion — video selection dominates by roughly six-fold either way.

---

## 5 · Multiple comparisons

`docs/research/INFRA.md:624` states a Holm correction rule for families of hypotheses.
`tools/make_summary.py` and `tools/size_curve.py:108` never apply it, across families of 15–27
tests. Applying the repo's own `holm()` reportedly removes all three "worse" verdicts and one
third of one "significant on all three seeds" claim. **Not independently re-verified** — but it
is the highest-value single check remaining, because it cuts *toward* the project on the NPS
"worse" rows and *against* it on a significance claim.

---

## 6 · Repository state

Real strengths: a large and genuinely load-bearing suite; `SUMMARY.md` self-describing since the freeze; scorecards carry
`git_sha`, `weights_sha256` and the exact command; `Protocol.mismatches_with` refuses
cross-protocol subtraction in code; `check_docs.py` passes over 85 documents.

Gaps found: scorecards record no matching rule (the one setting §3 shows is being
misdescribed); recall/precision columns throughout §3, §8 and the edge report are F1-oracle
operating points chosen on the set they are reported on — the practice `dronedet/metrics.py`
forbids in writing; the edge benchmark carries no interval or test anywhere; the "31/31 perfect
sensor" control arm has **no artefact in the repository**; and the test count has four live
values across README (950), INFRA/CI (943), round 8 (941) and sota-methods (540). `INFRA.md:103`
names the three files to keep in step and they are not in step; the four new tests added with the
launch checker make every one of them stale. The true count should be generated, not typed.

---

## 7 · What to do, in order

1. **Do not post.** Fix §2.1–§2.3 in the launch package first: cite Temporal-YOLOv8 and narrow
   the novelty claim; drop "same training data"; change "the test video" to "the training video"
   for the bird result.
2. **Regenerate the images and fix the generators** — `make_social_card.py`, `make_result_charts.py`,
   the three architecture SVGs — then `NOTICE.md`, `final/README.md`, `gallery.html`,
   `methods.md`, `index.html:243`.
3. **Add the null result to README §6 and §12**: temporal vs single-frame is 6 × "no difference"
   on ARD-MAV and 2 × worse on NPS.
4. **Fix `README.md:121`** to say what the tests say, and **`README.md:32`/`:127`** to say IoU.
5. **Run the Holm correction** and republish whatever verdicts change.
6. Re-run `tools/check_launch_claims.py`, and extend it — it checks the post's numbers but not
   its *provenance words*, which is where every §2 defect lives.

The project's habit of publishing its own negative results is real and unusual, and it is why
this audit had artefacts to check. The failure mode it has not yet solved is narrower: a claim
gets corrected in the file where it was found, and left standing in the four files that repeat it.
