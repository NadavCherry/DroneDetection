# Addendum — closing four gaps in the research briefings

A completeness critic re-checked [the methods briefing](sota-methods-2026.md) and
[the data briefing](datasets-and-benchmarks-2026.md) and found four holes. This closes them.
Two entries are **corrections**: one claim in the briefings was overstated, one was under-verified
and has now been confirmed against a specific commit.

Written 2026-08-12. ✅ = I opened the source. ⚠ = unverified, quoted from an abstract or a record.

> **Tooling note, recorded because it bounds this document.** The session's web-search budget
> (200/200) was exhausted partway through, and arXiv's export API was rate-limited from this IP.
> The agents worked around it with OpenAlex, Crossref, HuggingFace, Zenodo, arXiv's search UI over
> WebFetch, `curl` for full text, and — for the code question — `git clone` and direct execution,
> which is *better* than search for that task. All four gaps were closed. The one systematic
> weakness that remains: **MDPI returned HTTP 403 to every automated fetch**, so numbers from MDPI
> journals (Drones, Sensors, Applied Sciences, Electronics) come from publisher-deposited Crossref
> metadata rather than a rendered page. They are marked where they appear.

---

## 1. CORRECTION — the MLLM / open-vocabulary family, which the original sweep never searched

The first sweep never ran a single query on language-guided, open-vocabulary, or MLLM detection.
That was a real hole: it is the one family with a plausible *semantic* route to drone-vs-bird, and
the briefing recorded neither a result nor a negative.

**The answer is that they collapse below roughly 8–12 px, and as of 2026 this is measured in at
least six independent places — four of them published this year.** This is no longer an unwritten
defence; it is a citable one.

| Source | What it measures | The number |
|---|---|---|
| [FineSightBench](https://arxiv.org/abs/2606.07861) (Jun 2026) ✅ | The pixel size at which a VLM stops seeing at all, on clean synthetic targets | Best model (Gemma-4-31B) needs **7.3 px** to reach 50 % accuracy; GPT-4o **14.0 px**; InternVL3.5-38B **16.4 px**. At 4 px "most others fail almost completely" — misclassification 22.9 % |
| [Scale-biased confidence in OVD](https://arxiv.org/abs/2607.10993) (Jul 2026) ✅ | Whether open-vocab confidence is usable at small scale | Mean confidence small vs large: **GroundingDINO 0.180 / 0.520**, OWL-ViT 0.085 / 0.279, YOLO-World 0.145 / 0.615. Threshold for 50 % recall: **0.129 small vs 0.476 large (3.7×)**. Authors call it "structurally inevitable from CLIP's image-level pretraining"; oracle per-scale thresholds recover ΔF1 **+0.001** for small |
| [DroneEyes / SkyAnchor](https://arxiv.org/abs/2607.19857) (Jul 2026) ✅ | Frontier MLLMs on tiny aerial targets, 2,140 videos | J&F out of 100: **Claude-Sonnet-4.5 0.95, GPT-4o 1.39, GPT-5 1.75, Gemini-3.0 2.76**. A purpose-built 3B model reaches 38.1 |
| [SOUBench](https://arxiv.org/abs/2604.22884) (Apr 2026) ✅ | 15 MLLMs on COCO-small VQA | Best GPT-5.2 **58.97 %** vs human 84.92 %; **aerial is the worst** of three domains; object-localisation sub-task 26–46 % vs human 100 |
| [ATA benchmark](https://doi.org/10.3390/drones10060429) (Jun 2026) ✅ | Air-to-air tiny-drone tracking, with and without language | Best is **vision-only** SeqTrack **44.55** AUC; best vision-language MMTrack **38.07**. Authors: "vision-language trackers do not always outperform purely visual trackers after introducing language prompts" |
| [OWL-ViT + SAHI on VisDrone](https://doi.org/10.15625/1813-9663/21899) (May 2026) ✅ | Size-stratified open-vocab detection on drone imagery | Small-object AP50 **10.7**, large 45.2. With SAHI tiling small rises to 17.5 — still beaten by a plain closed-set TOOD |
| [AeroPinWorld](https://doi.org/10.3390/electronics15071364) (Mar 2026) ✅ | Zero-shot open-vocab on VisDrone | YOLO-World v2-S **0.112 mAP / 0.054 AP_small**; the entire published zero-shot field sits between **0.105 and 0.135 mAP** |

**The sentence to use.** *SpeckLock operates at 3–14 px. The best vision-language model yet measured
needs 7.3 px to reach even coin-flip accuracy on a clean synthetic target, frontier models need
13–16 px, and on real tiny aerial targets GPT-5, Gemini-3.0 and Claude-Sonnet-4.5 score below 3 out
of 100. Language priors do not reach this scale; motion and geometry priors do.*

Two further points, both of which corroborate rather than threaten the architecture:

* DroneEyes' own ablation isolates their **temporal memory bank** at +5.1 (29.89 → 35.00) — the
  largest single term in their model. Temporal aggregation is what carries tiny targets, which is
  SpeckLock's thesis arriving from an entirely different direction.
* AeroPinWorld's whole contribution is removing **early stride-2 downsampling** because it is the
  transfer-critical step for tiny aerial targets — the same reasoning behind SpeckLock's P2 head.

**Caveat to state honestly:** FineSightBench is clean synthetic imagery with no clutter, motion, or
cloud, so its numbers are an *upper bound* on VLM ability, not a like-for-like comparison.

---

## 2. CORRECTION — the UAV-DETR NWD rebuttal is **verified true**, with three refinements

The methods briefing claimed, from one code read, that [UAV-DETR](https://arxiv.org/abs/2603.22841)
applies NWD's pixel-unit constant to normalised coordinates. It was flagged pending a second reader
because it is a public claim about someone else's work. A second agent cloned the repository
(`git clone --depth 1 https://github.com/wd-sir/UAVDETR.git`, HEAD **e51d1d81**, 2026-08-01), traced
the path, and re-executed the function under torch.

**Verdict: the claim holds.** Every link confirmed — predictions are sigmoid-bounded
(`ultralytics/nn/modules/transformer.py:387`), ground truth arrives normalised
(`dataset.py:147` → `augment.py:885`), nothing denormalises in between, and `constant=12.8` is never
overridden. The scale mismatch is exactly `imgsz`.

Three refinements before this is said publicly:

1. **The factor is `imgsz`, not a universal "640×".** Say "640× at their configured `imgsz=640`",
   citing `cfg/default.yaml:13`.
2. **Anchor the units in the original paper.** [NWD (arXiv 2110.13389)](https://arxiv.org/abs/2110.13389)
   sets *C* to "the average absolute size of AI-TOD" — which is where the literal 12.8 comes from —
   so C's units are pixels **by the method's own authors**. That is what makes this a units error
   rather than a tuning preference. Pre-empt the escape hatch: the paper also says "C is robust in a
   certain range", which invites the reply that 12.8 is just a hyperparameter. It is not, because
   the value was *derived from a pixel statistic*.
3. **A second, independent defect exists — mention it at most in a footnote.** The `xywh=True`
   branch at `ultralytics/utils/metrics.py:533-547` rebinds `w1, h1` to half-extents and then
   recovers the centre as if they were full extents, so the computed centre is off by w/4, h/4 and
   the size difference leaks into the centre term. Verified numerically: a box (cx=100, cy=100,
   w=40, h=20) returns centre (90.0, 95.0). It is a silent variable rebind rather than a conceptual
   error, and mixing it with the units claim muddies a clean argument.

**Strategic caution, and it is worth taking.** UAV-DETR is an unreviewed arXiv preprint. Publicly
dismantling one is a low-value and slightly unsporting use of this finding. The better use is
internal: it is evidence that `dronedet/nwd.py` is right, and a reason to state our own C in pixels
explicitly and test the sensitivity — not a stick.

---

## 3. CORRECTION — the wingbeat claim is overclaimed, and the space is more occupied than thought

The methods briefing's §5c has already been rewritten (see the ⚠ block there). The gap sweep
strengthens the correction: this is not merely radar prior art, there is **recent optical work
solving the same problem, and a patent family**.

| Source | What it does | The number |
|---|---|---|
| [Non-Appearance-Based Discrimination of UAVs and Birds in Optical Remote Sensing](https://www.mdpi.com/) (Jan 2026) ✅ | Bird/drone separation from **temporal features only**, no appearance | **99.47 % accuracy, F1 99.51 %** (stacking ensemble); time-frequency signal-variation features alone 98.40 %; trajectory-only and wavelet baselines lower |
| Sarcos patent family, "early detection of UAVs" ✅ | Wingbeat-band discrimination, claimed | Bird bands **2–5 Hz** (small) and **6–10 Hz** (large); propeller **30–300 Hz**. Priority 2020-04-01, granted US member listed 2024 ⚠ |
| [K/W-band micro-Doppler of drones and birds](https://doi.org/10.1049/rsn2.12060) (2018) ✅ | The radar baseline | Measured wingbeats: Hawk Owl ~6 Hz, Harris Hawk ~4 Hz, Tawny Eagle ~4 Hz |
| [Multimodal fusion of flapping-wing targets](https://doi.org/10.3390/) (Oct 2025) ✅ | Flapping-wing *drones* — the counterexample class | Micro-Doppler **alone** only 66.7 % (RF/SVM), 33.3 % (MLP); needs polarisation + IR fusion to reach 99.8 % |

Three consequences:

* **A 2026 optical paper already reports 99.47 % on this task.** Any SpeckLock bird claim must be
  positioned against it, not presented as first. Its likely weakness is target scale — check whether
  it operates anywhere near 6–8 px before conceding the ground.
* **The defensible claim narrows to: doing it at few-pixel scale, in RGB, inside the detector's input
  representation**, rather than as a downstream signal-processing stage on an already-extracted
  track. That is genuinely different and worth measuring.
* **Flapping-wing drones defeat the cue by construction**, and micro-Doppler alone only reaches
  66.7 % on them. Any result must say it is about **rotary-wing** intruders.

---

## 4. NEW — adverse-weather **video**, the table the data briefing lacked

Only video can carry a temporal method, and the original hard-conditions table was mostly stills and
dashcam imagery. These are the sets a stabilised stack can actually run on.

| Dataset | Content | Weather | Annotations | Access |
|---|---|---|---|---|
| **ExtremeTrack / VISTAC-2** (ICPR 2026) ✅ | **188 videos** (96 hazy + 92 rainy), ~85,000 frames | real haze + rain | per-frame tracking boxes; splits **128 train / 20 val / 40 test** | Google Drive via [challenge site](https://sites.google.com/view/vistac-2). ⚠ The discrepancy is **resolved**: the ICPR page's "199" is wrong; the challenge site's **188** is authoritative. Submission closed 29 Mar 2026 — take the data, not the entry |
| **UAV-AWID** ✅ | 9,725/2,767/2,528 train/val/test + a 19,446-image augmented set | rain (drizzle→torrential), motion blur, noise, 3 severities each | boxes; baselines across YOLOv5/v8, Faster R-CNN, RetinaNet, YOLO-NAS | [github](https://github.com/AdnanMunir338/UAV-AWID). ⚠ Degradations are **synthetic**, so never call it "real adverse weather" |
| **VisioDECT** ✅ | 20,924 annotated images, 852×480, six drone models, with range and altitude labels | sunny / cloudy / evening | boxes | IEEE DataPort; Data in Brief 2026 |
| **SWUAV** (SWUAV-DANet) ✅ | 18,195 aerial images, 236,392 instances | **12 conditions** incl. blizzard, freezing rain, dense fog, backlit night | boxes, 5 vehicle classes | [Sensors 26(9):2793](https://doi.org/10.3390/s26092793). Aerial-of-**ground**, so wrong target class — value is the weather taxonomy |
| **AVisT** ✅ | 120 sequences, ~79,653 frames, 42 categories | 18 scenarios across 5 adverse-visibility groups | tracking boxes | [arXiv 2208.06888](https://arxiv.org/abs/2208.06888), site live |

**On "a drone entering a cloud": still nothing.** No dataset anywhere annotates cloud-entry or
cloud-exit events for an air target. [UAV-CB](https://arxiv.org/html/2603.17492v1) has *clouds* as a
background category, not an occlusion event. This remains an unoccupied niche and a cheap dataset
contribution — our own footage plus the HazyDet ASM ladder is the fastest route to a first
measurement.

---

## 5. NEW — the June–August 2026 window

The critic's first gap was a six-week blind spot at the newest end. It has now been swept: 55 items,
45 verified, using arXiv's search UI, OpenAlex (1,849 in-window works scored locally), Crossref,
HuggingFace and Zenodo, after WebSearch and the arXiv API were both exhausted. Nothing here
overturns the plan; three items change what we should claim.

### 5a. The bird claim now has an independent, published confirmation — and loses its novelty

[**Improving visual differentiation of drones and birds using trajectory features**](https://doi.org/10.1007/s00521-026-12080-5)
(Neural Computing and Applications, Jun 2026) combines a trajectory-feature network with an
appearance network for targets that are only a few pixels, and measures **up to +22 % frame-wise
classification accuracy from adding trajectory features over appearance alone** on small,
high-motion objects.

That is SpeckLock's central bet, confirmed independently, in vision, at few-pixel scale, three
months ago. Two consequences, and they pull in opposite directions:

* **It corroborates the mechanism.** Combined with the 92.0 % trajectory-only result already in the
  briefing, the "motion history separates a 6 px bird from a 6 px drone" claim is now supported by
  at least three independent groups.
* **It removes "nobody has exploited this".** The remaining defensible novelty is narrow and
  specific: doing it **inside the detector's input representation** rather than as a post-hoc stage
  on an already-extracted track, and **publishing the bird-attributed false-alarm rate**, which
  still nobody has. Claim those two, not the mechanism. ⚠ Paywalled (Springer); in-house dataset,
  no public release — read the full text before positioning against it.

### 5b. A directly comparable edge detector, published last month

[**Lightweight Small-UAV Detection via Synergistically Enhanced YOLOv11**](https://doi.org/10.3390/app16157423)
(24 Jul 2026) is almost exactly SpeckLock's edge recipe, independently derived: **add a P2 branch,
delete the P5 large-object head**, DySample upsampling, WIoU v3, deployed to a Jetson Orin Nano Super
with TensorRT FP16.

| | number |
|---|---|
| DUT Anti-UAV | **92.2 % mAP@0.5**, 61.5 % mAP@0.5:0.95 |
| params | 2.11 M (−18.2 % vs YOLOv11n) |
| cross-dataset | validated on Det-Fly and LRDDv2 |

This is the head-to-head the audit says we have never run, on hardware weaker than ours, with an
IoU-based number. "Add P2, delete P5" is also a concrete, cheap ablation SpeckLock has not tried.
Two more architectural competitors in the same window:
[Gaze-DETR](https://arxiv.org/abs/2607.19040) (Anti-UAV410 **87.06 mAP50 / 90.90 F1**), which learns
*where to look* before running the expensive head — the same insight as proposal→verify from the
other end; and [S-Drone-YOLO](https://doi.org/10.3390/app16125854), whose quality-map-rescales-logits
head is a drop-in candidate. ⚠ All three are MDPI/arXiv; MDPI landing pages returned 403 to
automated fetch, so numbers come from publisher-deposited metadata, not rendered pages.

### 5c. Two new datasets that hit both priorities at once

| Dataset | Why it matters | Access |
|---|---|---|
| [**TriCross-D2D**](https://doi.org/10.3390/drones10060459) (12 Jun 2026) | Air-to-air drone-to-drone with **three coupled domain shifts — scene, viewpoint and weather** (real flight video plus controlled synthetic fog). 13 sequences, 23,403 raw frames, 7,045 benchmark images, 9,771 instances, **73.8 % extremely-tiny/tiny/small**. Moving camera, tiny targets, fog — the exact intersection of both stated priorities | MDPI open access; ⚠ download route unverified (403) |
| [**UAV_SMID v2**](https://data.mendeley.com/datasets/3k3hjc7rkt/2) (3 Aug 2026) | 13,928 images, 16,229 objects, **five deliberately balanced classes: helicopter, bomb, drone, bird, aeroplane** (3,162–3,440 each). Most anti-UAV sets are drone-only, so a detector trained on them has never been shown a hard negative | Mendeley, **CC BY 4.0, direct download** — no form, no agreement |

Also worth noting: [**SkyEV**](https://arxiv.org/abs/2607.18747) (21 Jul 2026) is RGB+event UAV
detection recorded with deliberate camera ego-motion and **uncompressed** frames — the authors made
the same call SpeckLock did about compression destroying a tiny target's local contrast, which is a
citable corroboration of the raw-frames decision in `pursuit_proto.py`. And a
[3D LiDAR UAV-and-bird set](https://doi.org/10.5281/zenodo.21413097) (4 Aug 2026) pairs *named bird
species* with *named drone models* plus trajectories — no camera, but the trajectory statistics are
directly reusable for the kinematic gate.

### 5d. One idea worth stealing

[**AEGIS**](https://doi.org/10.5281/zenodo.21141573) (2 Jul 2026) is a counter-UAS pipeline built to
**abstain rather than guess**: evidential fusion with a containment guarantee, and runtime
verification that declares when the evidence conflicts. For a system whose stated priority is false
positives, "refuse to answer" is a design axis SpeckLock does not currently have — the track
classifier outputs drone/other with no third state. Zenodo, CC BY 4.0, code on GitHub,
bit-reproducible under a fixed seed.

⚠ **Do not quote [DroneShield-AI](https://arxiv.org/abs/2606.11687)** (96.1 % detection at 3.2 %
false-alarm) without heavy qualification: single author, self-reported, no independent benchmark.
It is the highest-profile new counter-UAS system paper in the window and its false-alarm number
*will* be quoted at us, so it is listed here to be ready for, not to cite.

### 5e. What the window does **not** contain

No method claiming to beat YOLOMG on ARD100. No fifth Anti-UAV challenge. No ninth Drone-vs-Bird
edition. MGMD's **AP 0.55** on ARD-MAV still stands as the bar. The head-to-head targets chosen in
the plan are still the right ones.

---

## What changes in the plan

Very little in the ordering, and one addition.

* **No reordering.** [PLAN.md](PLAN.md)'s measurement-first sequence stands. Nothing here touches
  the ARD-MAV split leak, the oracle/real conflation, or the box-extent ceiling — all three are
  internal-validity problems immune to any literature gap.
* **The MLLM evidence is a free win.** It costs nothing, needs no GPU, and pre-empts the most
  fashionable objection ("why not just use a foundation model?"). Add it to the related-work
  section as a measured negative, not an opinion.
* **The bird claim needs repositioning before it is made**, against a Jan 2026 optical method at
  99.47 % — and it must be scoped to rotary-wing intruders.
* **ExtremeTrack joins the acquisition list** as the only real adverse-weather *video* with tracking
  boxes. It is small, it is downloadable now, and it is the only external set on which the
  fog-versus-temporal-stack hypothesis can be tested on real weather rather than synthesised haze.
