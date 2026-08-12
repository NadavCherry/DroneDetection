# Plan — from "amateurish" to a defensible SOTA claim

Written 2026-08-12 from four inputs, all in this directory:
[the SOTA review](sota-methods-2026.md), [the data & benchmark briefing](datasets-and-benchmarks-2026.md),
the [internal audit](internal-audit-2026-08.md), [measurements I re-ran here](verified-measurements-2026-08.md),
and [the addendum](addendum-2026-08.md) that closed four gaps a completeness critic found.

---

## The diagnosis in one paragraph

The mechanisms are not amateurish. The stabilised temporal stack won CVPR 2025 Anti-UAV Track 1 and
was independently published in Sensors 2024 with a near-identical gain (0.465 → 0.839 against this
repo's 0.06 → 0.83). Refusing IoU at 4 px is backed by TPAMI 2025 (same model: IoU-AP 10.9 vs
SAFit-AP 24.2) and by MVA 2025 adopting a centre-distance metric for the same reason. The NWD
implementation here is *more* correct than a 2026 arXiv paper's. And the parameter-scaling lane is
closed for sub-16 px targets — a 3-billion-parameter model reaches only 16.2 APvt on AI-TOD-v2,
*below a 2026 method* at a fraction of the size, while DINOv3-7b-sat scores 9.2, level with 2022's
RFLA — so priors, not scale, are the open lane, which is exactly where this project lives.

What is amateurish is the **evidence packaging**: a headline of 1.000 on one video, a comparison
table that varies metric *and* dataset *and* split simultaneously, no seeds or intervals on the
detection half, no competitor ever run in-house, and — the one genuine methodological fault — an
ARD-MAV number computed on a home-made split by a model trained on most of the official test set.

## The three things that must be fixed before any new claim

**1. The ARD-MAV split leak.** `tools/make_dataset_external.py:41` defines the official split
(15 test videos, Guo et al.); `combined_splits()` at line 209 ignores it and re-splits by position
(`test = i % 10 == 0`, 6 of 60). Most of the official test videos are therefore in training. The
0.836 cannot sit in a table beside MGMD's 0.55. Re-acquire, retrain on the official 45, score the
official 15 at IoU 0.25 *and* centre distance.

**2. The oracle/real conflation in the pursuit headline.** `work/pursuit/city/results.json` is 24
episodes, all `"intercept"`, with `detector: "oracle"`. `work/pursuit/city_pipe/results.json` is 3
episodes, all `"target_struck"`. Both numbers are true; only the first reaches the README's line 42,
while the honest version sits at line 314. Put the sensor in the claim everywhere it appears.

**3. Box extent — the reason no number here is comparable.** `tools/make_datasets_v3.py:50` sets
`LABEL = 24.0`, so every training label on our own data is a 24 px square whatever the drone's real
size; predictions inherit it (measured: detection width is a constant 24.0 px); and the 10_06 ground
truth is itself a constant 8.0 px. Measured max achievable IoU is **0.111**, so COCO AP is
structurally **0.000** — not low, *impossible*.

**Measured on the real ARD-MAV annotations** (downloaded and parsed 2026-08-12, 106,456 boxes):
`--min-side 12` grows 59 % of boxes but still leaves **95 % able to reach MGMD's IoU 0.25** and 74 %
able to reach 0.5 — tolerable, not disqualifying. Dropping it to 8 takes IoU-0.5 reachability to
96.9 %. So the external path is fixable with one flag; it is *our own* data at `LABEL = 24` that is
arithmetically hopeless. Full tables in
[verified-measurements §6b](verified-measurements-2026-08.md).

**And the headlined ARD-MAV clip is the third-easiest video in the dataset.** `phantom16`, the single
clip behind "ARD-MAV AP 0.994", has a median target of **39.1 px** — rank 57 of 59, the 97th
percentile — against a dataset median of 11.2 px and an official-test-split median of 12.1 px. At
39 px the target is in the AI-TOD *medium* bin, so that number does not exercise this project's
thesis at all. This is a stronger version of the audit's cherry-picking finding and it is
independently verified from the annotations.

The inflation existed to stop IoU-based label assignment starving tiny ground truths of positive
anchors — which is precisely what `dronedet/nwd.py` now solves by a principled route. The experiment
is therefore: **true extents + NWD, versus inflated extents**, scored both ways. If NWD carries the
assignment as designed, comparability comes back for free.

## What is genuinely novel and worth building on

Three facts line up into one opportunity nobody has taken:

- The team that **won** the 2025 Drone-vs-Bird challenge tried single-frame bird rejection, it
  failed, and their stated future work is *"a classifier that analyzes multi-frame image patches to
  accurately differentiate drones from similar objects, such as birds"*. That sentence describes the
  track classifier already in this repo.
- Trajectory **alone** reaches 92.0 % bird/drone accuracy (LAT-BirdDrone 2025); track-level context
  is worth +73 % on birds over per-frame (OBSS). Appearance at 10 px carries almost nothing.
- **No benchmark measures it.** Drone-vs-Bird has birds but leaves them *unlabelled*; ARD100 has no
  bird class; YOLOBirDrone publishes no confusion matrix. In eight editions of the challenge, no
  bird-specific false-alarm rate has ever been published.

Measured here today, and currently unreported: **99.8 % drone recall with 0 hits on 934 labelled
bird instances at 0.002 FP/frame**, where the raw detector takes 151 bird hits at 95 % recall. The
mechanism claim — *temporal track evidence, not appearance, is what separates a 6 px bird from an
8 px drone* — is real. It rests on 8 bird tracks in one flock on one afternoon, which is an
existence proof, not a result. Halmstad turns it into one.

⚠ **Scope this claim carefully — the recency sweep changed it.** A June 2026 paper
([NCAA](https://doi.org/10.1007/s00521-026-12080-5)) independently measures **+22 % frame-wise
accuracy from trajectory features over appearance alone** on few-pixel high-motion targets, and a
January 2026 optical method reports 99.47 % on bird/drone separation from temporal features. The
mechanism is confirmed, not unexploited. What is still unclaimed by anyone: doing it **inside the
detector's input representation** rather than as a post-hoc stage on an extracted track, and
**publishing the bird-attributed false-alarm rate**. Claim those two. And say *rotary-wing*:
flapping-wing drones defeat the cue by construction.

The physical argument for weather is equally cheap and equally untested: haze is
`I = J·t + A(1−t)` with `t = exp(−βd)`, a low-frequency field, so wherever depth is locally constant
the veil **subtracts out of a stabilised temporal difference for free** — no dehazing network, none
of the artefacts HazyDet measured as harmful (3 of 9 dehazers made detection *worse*; training on
degraded data beat the best dehazer 48.7 vs 44.8). If the single-frame/temporal gap *widens* with
fog, that is a novel result and it converts the biggest measurement gap into the strongest claim.

## Order of work

Cheap and measurement-only first — these answer the damaging criticisms without a GPU-week.

| # | Work | Why now | Cost |
|---|---|---|---|
| 1 | Re-score every existing result with `python -m dronedet bench` — IoU **and** centre-distance, size-binned, with bird hits and bootstrap CIs | Kills "you invented your own metric" by reporting both; already built and tested | done / hours |
| 2 | Put *n*, the sensor, and the hardware on every headline number; demote 1.000-on-one-video to a case study and lead with the generalist result | The audit's top 8 fixes; seven are hours and none needs retraining | hours |
| 3 | ⚠️ **CORRECTED 2026-08-12 — YOLOMG releases NO weights.** Its only release asset is a 60 MB visualisation MP4 and its README trains from stock `yolov5s.pt` (checked: releases API, rendered README, raw README). So this item is not "download and run". **Train** YOLOMG's architecture (GPL-3.0, compatible with our AGPL-3.0) on **our** ARD-MAV official split — which is the better experiment anyway, because a rival trained on the same split gives a *paired* comparison and therefore a real p-value, where its own weights on its own data could only ever give an interval-vs-point-estimate. `p2_no_p5_ardmav` (the Jul-2026 YOLOv11 edge recipe, already implemented here) is the cheap second baseline | "You have never run the baseline" is unanswerable — and it is the only route to a significance test, since a published scalar has no distribution | **days**, not hours |
| 4 | **`local_extent` group** — `baseline_local` then `trueextent_local`. `tools/make_datasets_v3.py --label-px 0` now writes true extents (the flag exists and is tested); this is the run that uses it. **This is the only item that touches the fatal case** — `trueextent_ardmav` addresses ARD-MAV, where inflation is merely tolerable | Restores comparability where it is actually broken: 0 % of 07_05's boxes can reach IoU 0.5 at `LABEL = 24`, against 74 % for ARD-MAV at min_side 12. Testable entirely on data already on disk | days |
| 5 | Re-acquire ARD-MAV, retrain on the official 45, score the official 15 | Removes the one real methodological fault | days |
| 6 | Halmstad (CC0, video, labelled birds, night, 203 k frames, no agreement) → publish the drone-vs-bird confusion matrix stratified by target pixel size. Add **UAV_SMID v2** (Mendeley, CC BY 4.0, direct download, 5 balanced classes with 3,162–3,440 bird objects) as immediate hard negatives — most anti-UAV training sets have never shown a detector a bird | Nobody has published this table; it is the owner's stated priority | days |
| 7 | Fog ladder on our own clips via the HazyDet ASM (`A ~ N(0.8,0.05²)`, `β ~ N(0.045,0.02²)`); report detection rate **and** stabiliser inlier count per β. Then validate on **real** weather: ExtremeTrack (188 videos, 96 hazy + 92 rainy, tracking boxes) and **TriCross-D2D** (air-to-air, moving camera, 73.8 % tiny, real + controlled fog) | Tests the physics claim above; two curves, one figure — and TriCross-D2D is the only set hitting both priorities at once | days |
| 8 | 3 seeds on the headline config, report mean ± std | The pursuit half already reports Wilson intervals; the detection half must match | 3× train |

Day 0, in parallel, because they have human latency: email `wosdetc@googlegroups.com` for the
Drone-vs-Bird data agreement, and email the LRDDv3 authors for source clips rather than their 5 FPS
frame sampling (at 5 FPS a 12-frame lag is 2.4 s and the stack is meaningless).

## Where we could actually be measured against others

Live and open today, verified 2026-08-12:

| Venue | Status | Note |
|---|---|---|
| [LTS Multi-UAV Tracking](https://www.codabench.org/competitions/16223/) | open to 2030 | Top HOTA 0.9316. Scenario mix is this project's hard-condition list verbatim: 39 cloud-background videos, 68 tree-background. Baseline author used a 6 GB RTX 4050 — weaker than this machine. |
| [MVA 2025 SMOT4SB post-comp](https://www.codabench.org/competitions/5101/) | open, no end date | 108 k frames of tiny birds from a moving camera and **zero drones** — the perfect pure false-positive corpus. Also the peer-reviewed cover for centre-distance scoring. |
| [MVA 2023 SOD4SB post-comp](https://codalab.lisn.upsaclay.fr/competitions/9594) | end date "Never" | Single-frame twin of the above — a controlled ablation. |
| [MaCVi 2026 Thermal Detection](https://macvi.org/dataset) | still accepting | Judged on accuracy **and** embedded feasibility — the argument `realtime/` already makes. |
| [HazyDet](https://github.com/GrokCV/HazyDet) | rolling table | **No temporal method appears on it anywhere.** |

## What not to do

Scaling the backbone (measured not to help below 16 px), dehazing or super-resolving as a front end
(published negative result, and this repo independently measured SR no-gain), VisDrone (a drone
looking *down* at cars — the static-world-cancels assumption fails over a moving ground plane), and
any event-camera benchmark (no sensor; cite as convergent evidence instead). AOT is 11 TB and would
eat the entire disk budget for grayscale collision-course aircraft.
