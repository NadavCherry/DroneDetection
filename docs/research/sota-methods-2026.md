# SpeckLock — Methods Briefing

**Scope.** The strongest published methods for detecting a 3–14 px drone, what the field currently
believes are the winning ingredients, where SpeckLock stands against them, and how to hybridise.

**Reading conventions.**

| Mark | Meaning |
|---|---|
| ✅ | Verified against the primary source (paper text, released code, or measured on this machine) |
| ⚠️ | **Unverified** — reported in a secondary source, a search snippet, or an abstract only |
| ❌ | Claim contradicted by the authors' own released code |

**Hardware baseline for all feasibility calls:** one RTX 5070 Laptop (8151 MiB, sm_120 Blackwell),
32 CPU cores, 30 GB RAM, ~600 GB disk, torch 2.11 + cu128, ultralytics.

---

## 1. The current state of the art

### 1a. Drone-specific detection

| Method | Year | Venue | Datasets | Headline metric | Code | 8 GB? |
|---|---|---|---|---|---|---|
| [YOLOMG](https://arxiv.org/abs/2503.07115) | 2025 | arXiv (no venue) | ARD100, NPS-Drones | AP@.5 **0.85** ARD100 / **0.95** NPS (1280px) ✅ | [GPL-3.0 + weights](https://github.com/Irisky123/YOLOMG) | ⚠️ 640 yes (bs 2–4); 1280 no |
| [Laroca et al.](https://arxiv.org/abs/2504.19347) — DvB 1st | 2025 | IJCNN 2025 | Drone-vs-Bird (DDS) | mAP50 **0.7390** avg, 7-video val ✅ | ✗ none | ✅ reimplement (~1 day) |
| [WRN-YOLO](https://github.com/yjwong1999/IJCNN2025-DvB) — DvB 3rd | 2025 | IJCNN 2025 | Drone-vs-Bird (DDS) | **no number published** ✅ | [full code](https://github.com/yjwong1999/IJCNN2025-DvB) | ✅ bs 4 + accum, 11–14 h/run ✅measured |
| [Frame Dynamics](https://arxiv.org/abs/2505.04917) | 2025 | CVPRW 2025 — **1st, Anti-UAV Trk 1** | Anti-UAV410 (TIR) | AOA **73.23** ✅ | [public](https://github.com/facias914/A-Simple-Detector-is-a-Strong-Tracker) | ✅ trivially |
| [MGMD / GLAD](https://arxiv.org/abs/2410.10527) | 2024 | IEEE T-ITS | ARD-MAV | AP **0.55**, F1 0.69, 28 fps ✅ | [demo only](https://github.com/WestlakeIntelligentRobotics/Global-Local-MAV-Detection) | ⚠️ inference only (TRT rebuild) |
| [UAV-DETR](https://arxiv.org/abs/2603.22841) | 2026 | arXiv | DUT Anti-UAV + private | mAP50:95 **67.15** DUT ✅ | [public](https://github.com/wd-sir/UAVDETR) | ✅ bs 4 = 5.14 GiB ✅measured |
| [TransVisDrone](https://arxiv.org/abs/2210.08423) | 2023 | ICRA 2023 | NPS, FL-Drones, AOT | AP@.5 **0.95** NPS ✅ / **0.15** ARD100 ✅ | [public](https://github.com/tusharsangam/TransVisDrone) | ❌ 42 GB @ bs4 |
| [Dogfight](https://arxiv.org/abs/2103.17242) | 2021 | CVPR 2021 | NPS, FL-Drones | AP@.5 **0.89** NPS, ~1 fps ✅ | [public](https://github.com/mwaseema/Drone-Detection) | ❌ TF 1.12 / CUDA 9 — dead on sm_120 |
| [SDD-YOLO](https://arxiv.org/abs/2603.25218) | 2026 | arXiv | private DroneSOD-30K only | mAP50 **0.860** ⚠️ | ✗ none | n/a — unreproducible |
| [YOLOBirDrone](https://arxiv.org/abs/2601.08319) | 2026 | arXiv | private BirDrone | mAP50 **0.948** ✅(paper) | [stub, 838 KB](https://github.com/dapinderk-2408/YOLOBirDrone) | ❌ dataset never released |

### 1b. General tiny-object detection (transferable losses / assigners / heads)

| Method | Year | Venue | Datasets | Headline metric | Code | 8 GB? |
|---|---|---|---|---|---|---|
| [ScaleBridge-Det](https://arxiv.org/abs/2512.01665) | 2025 | arXiv | AI-TOD-v2 | AP **35.7** (3.0B params) ✅ | ✗ | ❌ 3B params |
| [D³R-DETR](https://arxiv.org/abs/2601.02747) | 2026 | arXiv | AI-TOD-v2 | AP **31.3** / APvt 16.6 ✅ | ✗ | ⚠️ likely |
| [DQ-DETR](https://arxiv.org/abs/2404.03507) | 2024 | ECCV 2024 | AI-TOD-v2 | AP **30.2** / AP50 68.6 ✅ | [public](https://github.com/hoiliu-0801/DQ-DETR) | ✅ best reproducible baseline |
| [Dome-DETR](https://arxiv.org/abs/2505.05741) | 2025 | ACM MM 2025 | AI-TOD-v2, VisDrone | AP **28.7** ✅ | [public](https://github.com/RicePasteM/Dome-DETR) | ✅ |
| [NWD-RKA](https://arxiv.org/abs/2206.13996) | 2022 | ISPRS J. | AI-TOD-v2 | AP **24.7** / AP50 57.2 ✅ | [mmdet](https://github.com/Chasel-Tsui/mmdet-aitod) | ⚠️ mmcv blocked on sm_120 |
| [RFLA](https://arxiv.org/abs/2208.08738) | 2022 | ECCV 2022 | AI-TOD-v2 | AP **25.7** / **APvt 9.2** ✅ | [mmdet](https://github.com/Chasel-Tsui/mmdet-rfla) | ⚠️ mmcv blocked |
| [SAFit / RGBT-Tiny](https://arxiv.org/abs/2406.14482) | 2025 | TPAMI (author-claimed ⚠️) | RGBT-Tiny | ATSS **IoU-AP 10.9 vs SAFit-AP 24.2** ✅ | [gated](https://github.com/XinyiYing/RGBT-Tiny) | ✅ metric is ~30 lines |
| [DEIMv2 / DINOv3](https://arxiv.org/abs/2509.20787) | 2025 | arXiv | COCO | 57.8 AP; **"small objects largely unchanged"** ✅ | [public](https://github.com/Intellindust-AI-Lab/DEIMv2) | ✅ (S/Pico) |

### 1c. Multi-frame / temporal (the family SpeckLock belongs to)

| Method | Year | Venue | Datasets | Headline metric | Code | 8 GB? |
|---|---|---|---|---|---|---|
| [Temporal-YOLOv8](https://pmc.ncbi.nlm.nih.gov/articles/PMC11598073/) | 2024 | Sensors 24(22):7387 | Nano-VID (private) | mAP **0.465 → 0.839** by stacking 3 grey frames ✅ | ✗ | ✅ reimplement |
| [DeepPro](https://arxiv.org/abs/2506.12766) | 2026 | **IEEE TPAMI** | NUDT-MIRSDT, RGBT-Tiny | Pd **98.50** / Fa 0.72, **0.197M params, 184 fps** ✅ | [public](https://github.com/TinaLRJ/DeepPro) | ✅ trivially |
| [TDCNet](https://arxiv.org/abs/2511.09352) | 2026 | **AAAI 2026** | IRSTD-UAV, IRDST | F1 **97.12** / AP50 93.83 ✅ | [public](https://github.com/IVPLabs/TDCNet) | ⚠️ likely (18.5 fps) |
| [Dual-Interval Motion Cues](https://arxiv.org/abs/2605.22605) | 2026 | arXiv | VisDrone-VID | mAP50 **27.4** vs 23.3 baseline ✅ | ✗ | ✅ |
| [LVNet](https://arxiv.org/abs/2503.02220) | 2025 | IEEE JSTARS | NUDT-MIRSDT, IRDST | IoU **91.66**, 1.77M params ✅ | [public](https://github.com/ZhihuaShen/LVNet) | ✅ |
| [XS-VID / YOLOFT](https://arxiv.org/abs/2407.18137) | 2024 | arXiv | XS-VID | AP 36.4 vs 33.6 YOLOv8-L; **APes +3.4** ✅ | [public](https://github.com/gjhhust/YOLOFT) | ✅ |
| [CHAL](https://github.com/UESTC-nnLab/CHAL) | 2026 | **CVPR 2026** | DAUB-H, NUDT-MIRSDT | ⚠️ numbers unread (CVF 403) | [public](https://github.com/UESTC-nnLab/CHAL) | ⚠️ unknown |

**The single most important row in this section:** Temporal-YOLOv8 (Sensors 2024) published
SpeckLock's exact core mechanism — replace the 3 RGB channels with 3 grayscale frames from different
timesteps — two years earlier, and measured **0.465 → 0.839 mAP**, nearly identical in shape to
SpeckLock's 0.06 → 0.83. Its ablation also tested the variants SpeckLock did not: 9 channels (3 full
RGB frames) = 0.743 and 11 grayscale channels = 0.781, **both worse** than the plain 3-grey stack at
0.839. This is simultaneously the strongest validation of SpeckLock's design and the clearest prior
art it must cite.

---

## 2. Winning ingredients, ranked by evidence strength

Ranked by *quality and independence* of the supporting measurement, not by size of the gain.

| # | Ingredient | Strength | The number | Source |
|---|---|---|---|---|
| 1 | **Temporal stacking at the input** | **Very strong** — 4 independent groups, 1 competition win | 0.465→0.839 mAP; 0.06→0.83 (SpeckLock); RGB+RGB 0.33 vs motion 0.78; AOA 73.23 = 1st place | [Sensors 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11598073/), repo, [YOLOMG T.III](https://arxiv.org/abs/2503.07115), [CVPRW25](https://arxiv.org/abs/2505.04917) |
| 2 | **Do not score tiny boxes with IoU** | **Very strong** — TPAMI + a competition metric | Same ATSS model: **IoU-AP 10.9 vs SAFit-AP 24.2** (2.2×) | [SAFit](https://arxiv.org/abs/2406.14482), [SO-HOTA](https://arxiv.org/abs/2507.12832) |
| 3 | **Ego-motion compensation before differencing** | **Strong** — the enabling precondition | Dual-interval GMC: 23.3→27.4 mAP50; DeCoDet ego-decoupling: 63.15→79.13 mAP50 | [2605.22605](https://arxiv.org/abs/2605.22605), [2606.15286](https://arxiv.org/abs/2606.15286) |
| 4 | **Multiple temporal lags, not one** | **Strong** — direct ablation | short-only 24.4, long-only 22.4, **both 27.4**; YOLOMG 2-frame 0.73 → 3-frame 0.78 | [2605.22605 T.II](https://arxiv.org/abs/2605.22605), [YOLOMG T.III](https://arxiv.org/abs/2503.07115) |
| 5 | **Native-scale crops / P2 head, never full-frame downscale** | **Strong** | LRDDv3 640px mAP50 0.543 → 1920px **0.822**; YOLOMG stride-2 head worth +0.02 AP | [2605.25942](https://arxiv.org/abs/2605.25942), [YOLOMG T.III](https://arxiv.org/abs/2503.07115) |
| 6 | **Tiny-object loss/assigner (NWD family)** | **Strong** | SimD +4.1 AP on very-tiny; SAFit rescues Sparse R-CNN 8.1→21.4 AP | [SimD](https://arxiv.org/abs/2407.02394), [SAFit T.I](https://arxiv.org/abs/2406.14482) |
| 7 | **Track-level integration before declaring** | **Moderate** — widely believed, rarely measured end-to-end | SpeckLock: per-frame AP ~0.9 → 1.000 tracked; DvB winner's single-frame rejector **failed** | repo, [2504.19347](https://arxiv.org/abs/2504.19347) |
| 8 | **Physics/kinematic gate on candidates** | **Moderate** — strong where measured, few replications | SpeckLock: 3.8%→80.7% on target, 3.1%→**0%** clutter; ASUMOT Fa 0.56e-4 @ Pd 89.9% | repo, [ASUMOT](https://arxiv.org/html/2607.11303v1) |
| 9 | **Longer temporal aperture than 3 frames** | **Moderate but contested** | DeepPro saturates at **T=40**; but LVNet optimal at T=2 (IRDST) / T=8 (NUDT) | [DeepPro](https://arxiv.org/abs/2506.12766), [LVNet](https://arxiv.org/abs/2503.02220) |
| 10 | **Learned differencing > hand-made differencing** | **Moderate** — one clean ablation | Hand TD F1 92.25 → learned TDC F1 **96.76** (+4.5) | [TDCNet](https://arxiv.org/abs/2511.09352) |
| 11 | **Train on degraded data; do NOT bolt on restoration** | **Strong (negative result)** | Best dehazer 44.8 vs **48.7** for training directly on hazy; 3 of 9 dehazers made it worse | [HazyDet T.V](https://arxiv.org/abs/2409.19833) |
| 12 | **Scaling the backbone** | **Strong evidence it does NOT work** | DINOv3 gains "on medium and large objects, small largely unchanged"; DINOv3-7b-sat **APvt 9.2** — worse than 2022 RFLA | [DEIMv2](https://arxiv.org/abs/2509.20787), [ScaleBridge T.](https://arxiv.org/abs/2512.01665) |

**Ingredient 12 is the strategic finding.** The parameter-scaling lane is closed for sub-16 px
objects. Stated precisely, from the AI-TOD-v2 numbers in
[the data briefing §4.9](datasets-and-benchmarks-2026.md):

| model | year | params | AP | APvt |
|---|---|---|---|---|
| RFLA | 2022 | ~R50 | 25.7 | **9.2** |
| NWD-RKA + DetectoRS | 2022 | ~R50 | 24.7 | 9.7 |
| DQ-DETR | 2024 | — | 30.2 | 15.3 |
| **D³R-DETR** | **2026** | — | 31.3 | **16.6** |
| ScaleBridge-Det | 2025 | **3.0 B** | **35.7** | 16.2 |
| DINOv3-7b-sat | 2025 | **7 B** | — | **9.2** |

Two separate facts, and they must not be merged: a **3-billion**-parameter model reaches only
16.2 APvt, *below a 2026 method* at a fraction of the size; and **DINOv3-7b-sat scores 9.2 APvt,
level with 2022's RFLA**. Four years and seven billion parameters moved very-tiny AP from 9.2 to
about 16.6, and the model that got there is not the biggest one. The open lane is priors: motion,
geometry, physics. That is precisely where SpeckLock operates.

> ⚠ **An earlier draft of this paragraph said the 3B model scored "below a 2022 method". That was
> wrong** — 16.2 is well above RFLA's 9.2 and NWD-RKA's 9.7 — and it was wrong in the direction
> that flattered the argument, which is the failure mode this whole document exists to prevent. It
> also cited "D³Q's 16.5"; the table says D³R-DETR 16.6. Both are corrected here.

---

## 3. SpeckLock vs the field — honest assessment

### 3a. Where SpeckLock already matches or exceeds SOTA

| Capability | SpeckLock | Field | Verdict |
|---|---|---|---|
| Temporal stack at input | 0.06 → 0.83 mAP50 ✅ | Same trick won CVPRW 2025 Anti-UAV Trk 1 | **Matches SOTA mechanism** |
| Ego-motion compensation | Affine/homography, cached in `DetectionSet.meta`, no stage recomputes ✅ | YOLOMG recomputes 2 homographies/frame offline and discards them ✅ | **Ahead on engineering** |
| NWD implementation | `assign_c=16` px (assigner) + `loss_c=2` cells (loss), **separate spaces** ✅ | UAV-DETR reportedly applies C=12.8 to **normalised [0,1]** boxes — a ~640× scale error that would make NWD ~linear — ⚠ **SINGLE CODE READ, NOT YET INDEPENDENTLY VERIFIED. Do not state publicly until a second reader confirms the exact lines.** | Ahead on correctness *if the claim holds* |
| P2 head | `configs/yolov8{s,m}-p2-ch4.yaml` ✅ | Standard practice (SDD-YOLO, LAF-YOLOv10) | **Matches** |
| Multi-dataset generalisation | ARD-MAV 0.994 · NPS 0.801 · black drone tracked, **one set of weights** ✅ | Every published leader is a specialist; survey lists no unified model ✅ | **Ahead — genuinely unique** |
| Closed-loop pursuit | 24/24 city, CPA 0.080 m, closed loop through a renderer + airframe ✅ | P2P is open-loop offline forecasting on a fixed dataset ✅ | **Ahead — nobody closes the loop** |
| Physics gate on clutter | 3.8%→80.7% on target, 3.1%→0% clutter ✅ | ASUMOT/EventRadar do the analogue but need **event cameras** ✅ | **Ahead in RGB** |
| Negative results recorded | SR no-gain, INT8 worse, DT=9, full-frame edge stabiliser ✅ | Almost universally buried | **Ahead on rigour** |
| Test discipline | 540 unit tests + CI + docs-link checker ✅ | Essentially unheard of in this literature | **Ahead on rigour** |

### 3b. Criticisms that **would stick** — ranked by how damaging

| # | Criticism | Why it lands | Fix | Cost |
|---|---|---|---|---|
| **1** | **The headline comparison is not a comparison.** README compares SpeckLock's centre-distance AP on ARD-MAV/NPS against published AP@0.5-IoU on ARD100/NPS. Three variables differ at once: metric, dataset (ARD-**MAV** ≠ ARD**100**), and split. | The README already concedes it ("read this as a *class* comparison, not a leaderboard entry"). A reviewer will simply stop reading there. Metric-shopping is the single most damaging accusation available. | Report **both** columns: IoU-mAP50 *and* centre-distance AP, on **one** split. Cite [SAFit](https://arxiv.org/abs/2406.14482) + [SO-HOTA](https://arxiv.org/abs/2507.12832) so centre distance reads as standard practice, not invention. | Low — re-scoring only, no retrain |
| **2** | **No head-to-head against a competitor's actual weights.** Every rival number is quoted from that rival's own paper on that rival's own split. | "You have never run the baseline" is unanswerable, and it is *cheap to fix*: YOLOMG ships `best.pt` for both 640 and 1280. | Run YOLOMG's released weights and SpeckLock on **one** common split; score both under both metrics. | Low — inference only |
| **3** | **Single-run numbers, no seeds, no variance, no CI on detection.** | Universal in this field, but "everyone does it" is not a defence against *amateurish*. [LER-YOLO](https://arxiv.org/abs/2605.20667) reports mean ± std over 3 runs — that is the bar. | 3 seeds on the headline config; report mean ± std. Pursuit half already reports CI [76.6, 93.3] — apply the same discipline to detection. | Medium — 3× train time |
| **4** | **AP/F1 = 1.000 on one 591-frame video.** | A perfect score on N=1 reads as an easy or leaked test set, not as strength. It actively undermines the far more impressive ARD-MAV/NPS generalist result. | **Demote it.** Lead with ARD-MAV 0.994 / NPS 0.801 one-model. Keep 10_06 as a qualitative case study. | Zero — editorial |
| **5** | **Bird evaluation is 8 hand-labelled tracks on one video.** | The FP claim — the thing the owner most cares about — rests on N=8, single scene, single bird species-group. Round 6 measures exactly 2 bird false tracks. | Enter a real bird benchmark (§5). | High — data access |
| **6** | **Zero adverse-weather numbers.** | Claims about fog/cloud/night currently have no measurement behind them at all. | §6 — synthesise fog with the published ASM recipe and report a degradation ladder. | Medium |
| **7** | **SpeckLock publishes no FP rate at a fixed operating point.** ⚠ *An earlier draft said the field publishes none — that is false and was corrected: the [Drone-vs-Bird review](https://pmc.ncbi.nlm.nih.gov/articles/PMC8072977/) has compared methods on "correct detection rate, false alarm rate, and average precision" for years.* The true gap is narrower and better: **no bird-attributed confusion rate exists anywhere**, because the birds are unlabelled (§5.3 of the data briefing). | Detection AP is not an alarm rate, and operators buy alarm rates. Stating the broad version invites a reviewer to discredit the surrounding argument with one citation. | Report our own FP/frame and FP/hour at a fixed confidence, plus track-level FP; and claim novelty only for the **bird-attributed** rate. | Low |

### 3c. Criticisms that would **NOT** stick — and the citation that kills each

| Criticism | Rebuttal | Source |
|---|---|---|
| "Stacking frames as channels is a hack" | It **won CVPR 2025 Anti-UAV Track 1** (AOA 73.23), and was published in Sensors 2024 with 0.465→0.839 | [2505.04917](https://arxiv.org/abs/2505.04917), [Sensors 24(22):7387](https://pmc.ncbi.nlm.nih.gov/articles/PMC11598073/) |
| "You invented your own metric" | TPAMI 2025 shows the same model scores IoU-AP 10.9 vs SAFit-AP 24.2; MVA 2025 replaced IoU with Dot Distance for the identical reason | [2406.14482](https://arxiv.org/abs/2406.14482), [2507.12832](https://arxiv.org/abs/2507.12832) |
| "You should be using NWD" | Already implemented — and **more correctly than a 2026 arXiv paper**, which applies C in normalised space | `dronedet/nwd.py`; cf. [UAV-DETR](https://arxiv.org/abs/2603.22841) |
| "Add a P2 head" | Already there (`configs/yolov8{s,m}-p2-ch4.yaml`) | repo |
| "Use a modern foundation backbone" | Scaling demonstrably does not help this object size | [DEIMv2](https://arxiv.org/abs/2509.20787), [ScaleBridge](https://arxiv.org/abs/2512.01665) |
| "Dehaze/super-resolve first" | Published negative result: best dehazer 44.8 vs 48.7 training on degraded data; SpeckLock independently measured SR no-gain | [HazyDet T.V](https://arxiv.org/abs/2409.19833), repo |
| "Interception is not real" | The README already marks it 🟡 sim-only and says "There is no flight test here" — this is *better* practice than the field | repo |

### 3d. Where SpeckLock is genuinely behind

| Gap | Field's position | Severity |
|---|---|---|
| Learned vs hand-made differencing | TDCNet: hand TD F1 92.25 → learned TDC **96.76** | Medium — a real +4.5 F1 on the table |
| Temporal aperture | DeepPro saturates at **T=40** frames; SpeckLock uses 3 taps over 13 | Medium — but LVNet contradicts (T=2 optimal on one set), so **measure, don't assume** |
| External benchmark breadth | 2 public datasets (ARD-MAV, NPS). AI-TOD-v2, DDS, Anti-UAV410 all untouched | Medium |
| Bird discrimination | No bird class in training; no confusion matrix | **High** — this is the owner's stated priority |
| Adverse conditions | Nothing measured | **High** — stated priority |
| Statistical hygiene on detection | No seeds/CI (pursuit half has CI; detection does not) | Medium |

---

## 4. Hybridisation proposals

For each: the graft, the failure mode it fixes, and the measurement that decides it.

| # | Host method | Graft from SpeckLock | Failure mode fixed | How to measure it worked | Cost |
|---|---|---|---|---|---|
| **H1** | [YOLOMG](https://github.com/Irisky123/YOLOMG) released weights | Tracker + track classifier (N_CONF=8) on the raw detections. **No retraining.** | Per-frame FPs: YOLOMG's DvB precision is **0.50** — half its detections are wrong | Track-level P/R vs per-frame P/R on ARD100 test + DvB. Success = precision ≫ 0.50 at ≤0.27 s added latency | **Days** |
| **H2** | YOLOMG (same weights) | Re-score under centre distance τ=12 **alongside** IoU@0.5 | Conflates "found it" with "localised to sub-pixel" on 42.18% sub-12px objects | Publish both columns stratified by size bin. Success = a documented reordering of the 640 vs 1280 ranking | **Hours** |
| **H3** | YOLOMG (retrain) | Replace scalar `E_t = (\|I_t−Î_{t−k}\|+\|I_t−Î_{t+k}\|)/2` with the ordered 3-moment stack (already 3-channel — **zero architecture change**) | (a) rectified magnitude destroys direction/order; (b) k=2 = 66 ms is too short for a hovering drone — their own stated failure mode | AP on ARD100 + the hover subset. Success = the "hovering drones ignored" failure disappears | Multi-day |
| **H4** | [WRN-YOLO](https://github.com/yjwong1999/IJCNN2025-DvB) | Temporal stack into the `wide_resnet50_2` stem — conv1 takes 3 channels, so **100.69M params and 585 GFLOPs are untouched** | Strictly single-frame; 30.4% of DDS boxes are ≤14 px and become ≤7 px after the 960 letterbox | mAP50 on their own `idx%2` split, especially `gopro_002` (where the 1st-place team's whole-image variant scored **0.0121**) | bs 4, 11–14 h ✅measured |
| **H5** | [Laroca DvB winner](https://arxiv.org/abs/2504.19347) (reimplemented) | Swap the post-hoc frame-consistency filter for the track classifier; add the stack as input | Their temporal step is a *box interpolator* — it cannot recover a drone the detector never fired on | mAP50 on the 7-video val split vs their 0.7390. Success = beat 0.7390, and specifically lift `gopro_002` (0.4491) and `hillside_cross` (0.4992) | ~1 day + train |
| **H6** | [UAV-DETR](https://github.com/wd-sir/UAVDETR) | (a) NWD into the **HungarianMatcher** (theirs is IoU-only in matching); (b) fix C from 12.8-normalised to pixel space | Assignment collapses to class+L1 for tiny boxes because GIoU is flat below one box width | DUT mAP50:95 vs their 67.15. Success = the "hybrid loss" ablation row (+2.06) grows once NWD actually engages | bs 4 = 5.14 GiB ✅measured |
| **H7** | [DeCoDet / HazyDet](https://github.com/GrokCV/HazyDet) | **Take, don't give:** import PDFT (clear → synthetic → real, staged backbone freeze, lr×0.1) into SpeckLock's sim→real gap | SpeckLock's shipped detectors collapse on the pursuit renderer (recall 0.50, 11k FP) — exactly what PDFT solves | Renderer recall/FP before vs after. Their measured PDFT gain was **+10.0 mAP**, larger than any architecture term in their paper | Medium |
| **H8** | [TDCNet](https://github.com/IVPLabs/TDCNet) | **Take:** replace fixed differencing with re-parameterised temporal-difference convolution | Hand-made TD leaves ~4.5 F1 on the table vs learned TDC | F1/AP50 on IRSTD-UAV with SpeckLock's stabiliser in front (they depend on external GIM registration and never study its failure modes — SpeckLock owns that stage) | Medium |
| **H9** | [YOLOBirDrone](https://arxiv.org/abs/2601.08319) | Temporal stack as the **bird discriminator**, plus the confusion matrix they omit | Their FP column is *background* FP by their own footnote — bird→drone confusion is **never measured** | Publish drone-vs-bird confusion at <32 px. Their headline gap over YOLOv8 is 0.001 mAP50; anything real wins | Reimplement only (dataset unreleased) |
| **H10** | [SAFit](https://arxiv.org/abs/2406.14482) | **Take:** adopt L_SAFit as a reported metric beside centre distance; ablate C (they never do) | Converts a house convention into citable standard practice — directly answers criticism #1 | Report IoU-AP / SAFit-AP / centre-distance AP side by side on every existing result | ~30 lines |
| **H11** | [DeepPro](https://github.com/TinaLRJ/DeepPro) | **Take:** test whether the temporal aperture should grow from 13 frames toward 40 | DeepPro's ablation shows gains to T=40; LVNet says T=2. Unresolved — and SpeckLock already rejected DT=9 | Sweep taps/lags on ARD-MAV + 10_06 at fixed net. Publishes a curve nobody has for RGB moving-camera | Low — sweep only |

**Recommended order.** H2 → H1 → H10 (all cheap, all measurement-only, all directly answer §3b
criticisms 1, 2 and 7) before any GPU-week is spent on H3/H4/H5.

---

## 5. Drone-vs-bird — everything known

### 5a. The field's measured numbers

| Source | Setting | Number | Verified |
|---|---|---|---|
| [YOLOMG T.V](https://arxiv.org/abs/2503.07115) | Zero-shot ARD100→Drone-vs-Bird | **P 0.50** / R 0.47 / AP 0.41 | ✅ |
| [Laroca et al.](https://arxiv.org/abs/2504.19347) | Single-frame FP rejector (MobileNetV3 on crops) | "**filtered out many false positives, [but] also lost many true positives**"; performance declined for **every** classifier tried | ✅ |
| [Laroca et al.](https://arxiv.org/abs/2504.19347) | Two-class drone+bird training | Training drone-only "resulted in slightly worse results" — the bird class helps **even when bird predictions are discarded** | ✅ |
| [YOLOv12-ADBC](https://www.mdpi.com/2504-446X/9/11/732) | Per-class accuracy | drone **96.4%** vs bird **80.0%** — a 16-point gap | ⚠️ MDPI 403; from search snippet |
| [OBSS sequence models](https://arxiv.org/abs/2207.10409) | Track-level (not frame-level) bird classification | bird classification **+73%**, overall F1 **+35%** vs per-frame | ✅ |
| [LAT-BirdDrone](https://pmc.ncbi.nlm.nih.gov/articles/PMC12769828/) | CNN+iTransformer on **trajectory alone** | acc **92.04%**, F1 92.57, AUC 98.21 (665 trajectories, mean 48 frames) | ✅ |
| [STBRNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC12049524/) | CNN+GRU on pre-cropped large targets | P 0.984 / R 0.964; **16 FP, 36 FN** absolute | ✅ |
| [Anti2 dataset](https://github.com/gdpinntit/-anti-interference-and-anti-UAV-dataset) | 4-class balanced | 1,688 UAV / **1,444 bird** / 1,406 helicopter / 1,320 airplane | ✅ |
| [AOD4](https://data.mendeley.com/datasets/cd5z895tr2/1) | 4-class, official split | 22,516 images, ~7,900 annotations/class | ✅ |
| [Halmstad](https://github.com/DroneDetectionThesis/Drone-detection-dataset) | **Video**, 4 classes incl. bird, **CC0 public domain** | 650 videos, 203,328 annotated frames | ✅ |
| [LRDDv3](https://arxiv.org/abs/2605.25942) | Range-labelled, bird class | 93,652 drone / **6,031 bird** / 722 airplane | ✅ |

### 5b. SpeckLock's own bird evidence

| Item | Value | Source |
|---|---|---|
| Hand-labelled bird tracks | **8** (a flock crossing the treeline, frames 2–304) | `work/gt_labeled_all.json`, round 2 |
| A bird corrupted the auto-ground-truth | The auto-GT's "return cruise across the sky" **was a bird**; round-1 winner `moe-hybrid` scored AP 0.004 because it was detecting the bird | round 2 |
| Measured bird false tracks | v1: **0**; v2 (higher recall): **2 (birds)** on 10_06 | round 6 |
| Stated position | "Drone-vs-bird at a few pixels is the unchanged honest frontier" | round 7 |

### 5c. The synthesis

Three facts line up into a single, publishable opportunity:

1. The 1st-place DvB team **tried** single-frame bird rejection and it **failed** — and their stated
   future work is verbatim *"developing a classifier that analyzes multi-frame image patches to
   accurately differentiate drones from similar objects, such as birds"*
   ([2504.19347](https://arxiv.org/abs/2504.19347)). That sentence describes SpeckLock's track
   classifier, written by the team that won.
2. Trajectory **alone** reaches 92.04% ([LAT-BirdDrone](https://pmc.ncbi.nlm.nih.gov/articles/PMC12769828/)),
   and track-level context is worth **+73%** on birds ([2207.10409](https://arxiv.org/abs/2207.10409)).
   Appearance at 10 px carries almost nothing; kinematics carries most of it.
3. **No benchmark measures it.** DDS has birds but leaves them **unlabelled**; ARD100 has no bird
   class; YOLOBirDrone never publishes a confusion matrix; RGBT-Tiny never mentions birds.

**The mechanism — stated defensibly.** A bird's wingbeat is non-rigid and periodic at 3–10 Hz.
Across t−12/t−6/t the three moments of a *flapping* target de-register in aspect ratio and area; a
*rigid* quadrotor body leaves three near-identical shapes on a smooth path. The stack therefore
encodes that cue **into the channel dimension, where a plain conv can read it** — at exactly the
scale where appearance has failed for everyone else.

> ⚠ **This paragraph originally claimed the mechanism was unexploited. That was wrong and would have
> been the most attackable sentence in the briefing.** Wingbeat-based drone/bird discrimination is a
> mature **radar micro-Doppler** literature: [MDPI Signals 4(2):18](https://www.mdpi.com/2624-6120/4/2/18)
> attributes "periodic fluctuations in signal intensity … to changes in body shape due to flapping",
> and [IET RSN 2021](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/rsn2.12060) does
> multi-band drone/bird separation on the same cue.
>
> The correct — and stronger — framing is that a stabilised multi-lag intensity stack is a **low-rate
> optical analogue of micro-Doppler**: same physical discriminant, a sensor that costs nothing, and
> twenty years of radar corroboration that the cue is real. What is genuinely new is doing it **in
> vision at few-pixel scale**, where the target is too small for any appearance model. Claim that,
> cite the radar work, and the argument becomes physically grounded instead of merely novel.
>
> **Named limitation, which must ship with the claim:** flapping-wing / ornithopter-class drones
> defeat a wingbeat discriminator *by construction*, and [SPIE 13471](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13471/134710D)
> models them explicitly. Any bird-vs-drone result built on this cue is a result about
> **rotary-wing** intruders and must say so.

**Cheapest credible experiment:** Halmstad (CC0, video, labelled birds, 203k frames — no data
agreement, no form) → build 3-moment stacks → report the drone-vs-bird confusion matrix stratified by
target pixel size. Nobody has published that table.

---

## 6. Adverse weather / cloud — everything known

### 6a. Measured degradation

| Source | Condition | Number | Verified |
|---|---|---|---|
| [XWOD](https://arxiv.org/html/2605.11521v1) | YOLOv11m under fog | mAP50 **23.45**, precision 66.08, **recall 21.92** | ✅ |
| [XWOD](https://arxiv.org/html/2605.11521v1) | Wildfire smoke | mAP50 17.85, precision 60.57, **recall 15.42** | ✅ |
| [XWOD](https://arxiv.org/html/2605.11521v1) | Best (tornado) vs worst (wildfire) | **51.12 pp spread** | ✅ |
| [YOLOMG T.V](https://arxiv.org/abs/2503.07115) | Zero-shot **night** | YOLOMG **AP 0.84** vs YOLOv9 0.64, YOLOv5 0.62 | ✅ |
| [HazyDet T.V](https://arxiv.org/abs/2409.19833) | Dehaze-then-detect, real fog | Best (RIDCP) **24.2** vs 21.5 baseline; **3 of 9 dehazers made it worse** | ✅ |
| [HazyDet](https://arxiv.org/abs/2409.19833) | Train on hazy vs restore at test | **48.7** (train hazy) vs 44.8 (best dehazer) | ✅ |
| [HazyDet T.IV](https://arxiv.org/abs/2409.19833) | **PDFT** (clear→synthetic→real) | 25.6 → **35.6** real-fog mAP (**+10.0**) — larger than any architecture term in the paper | ✅ |
| [CD-Buffer](https://arxiv.org/abs/2603.26092) | Real ACDC, test-time adaptation | fog 24.45, snow 15.41, rain 13.71, **night 8.92** mAP50 | ✅ |
| [2607.05467](https://arxiv.org/html/2607.05467v1) | UAV under synthetic fog | Fog degrades **primarily through missed detections**; fog-inclusive training beats restoration | ✅ |
| [2502.02027](https://arxiv.org/html/2502.02027v4) | Dehazing on **clear** images | Degrades clear-image detection — the front-end is a net loss most of the time | ⚠️ not fetched |
| [WACV 2025 RWS](https://openaccess.thecvf.com/content/WACV2025W/RWS/html/van_Lier_Evaluation_of_Spatio-Temporal_Small_Object_Detection_in_Real-World_Adverse_Weather_WACVW_2025_paper.html) | Real wind/rain/haze, 16.4 px objects | **Temporal stacking (TYOLOv8) beat both 3-frame-difference and plain YOLOv8 by +0.21 mAP** | ⚠️ CVF 403; per-subset table unread |

### 6b. The two findings that matter most for SpeckLock

**Finding 1 — the temporal stack is the *more* weather-robust choice, not the less.** The single
most directly relevant study (WACV 2025 RWS) tested exactly SpeckLock's architecture family against
real wind, rain and haze on 16.4 px objects and found the frame-stacking detector **won by +0.21
mAP** over both a classical 3-frame-difference detector and plain YOLOv8. ⚠️ Per-subset numbers
unread (CVF returns 403) — retrieve via IEEE Xplore doc 10972625 before quoting.

Mechanistically this is expected and worth stating as a claim SpeckLock can test: haze is
`I = J·t + A(1−t)` with `t = exp(−βd)` — a **low-frequency, slowly-varying** field. Wherever local
depth is roughly constant, the veil term `A(1−t)` **subtracts out of a stabilised temporal
difference** for free, with no dehazing network and none of the artefacts HazyDet measured as
harmful. That is a real, cheap, testable advantage over every single-frame method in §1.

**Finding 2 — the real risk is the stabiliser, not the detector.** Fog, cloud and night are exactly
the low-texture regimes where feature-based homography estimation loses lock. YOLOMG's released code
makes this concrete and unguarded: `motion_compensate` tracks 609 grid points and, if fewer than 15
survive, silently substitutes `H = diag(0.999, 0.999, 1)` — not even identity — and **nothing
downstream is told alignment failed** ✅ (read from released source). SpeckLock owns its stabiliser
end-to-end and can therefore do what nobody has: publish a
**stabilisation-residual-vs-accuracy curve**, and feed alignment confidence into the network as a
channel so the detector learns to discount the motion evidence when the warp is untrustworthy.

### 6c. What "drone enters a cloud" actually needs

| Requirement | Status in the field |
|---|---|
| A dataset with annotated cloud-entry/exit events | **Does not exist.** [UAV-CB](https://arxiv.org/html/2603.17492v1) has "clouds" as a *background category*, not an occlusion *event* ✅ |
| Temporally coherent weather synthesis | [WeatherWeaver](https://arxiv.org/abs/2505.00704) (ICCV 2025, NVIDIA) — video diffusion, explicitly lists **clouds** ✅ |
| Hallucination-safe synthesis | [RealWeather](https://arxiv.org/abs/2608.02953) — scene-fidelity RL specifically to stop the generator inventing/erasing structure ✅ |
| A published fog-degradation ladder | [HazyDet](https://arxiv.org/abs/2409.19833) ASM: `A ~ N(0.8, 0.05²)` in [0.7,0.9], `β ~ N(0.045, 0.02²)` in [0.02,0.16] ✅ |

**Critical caveat on synthesis.** A diffusion model asked for "cloud" over empty sky containing a
4 px drone is far outside its training distribution and will very likely erase or paint over the
target. **Validate before training on it:** run the pre-augmentation ground truth through the
detector and confirm the drone survives; or use cycle-consistency (clear→foggy→clear) as a
correctness gate ([Cyclone](https://arxiv.org/html/2607.13927v1)). Skipping that check produces
mislabelled data and invalidates every number downstream.

### 6d. Recommended weather plan

1. Apply the **HazyDet ASM** to SpeckLock's own clips at 5 β levels. For a sky-dominant scene at
   near-constant depth this degenerates to a principled contrast-and-veil augmentation — cheap,
   physically motivated, no diffusion risk.
2. Report **detection rate and stabilisation inlier count** at each β. Two curves, one figure.
3. Test the §6b Finding-1 hypothesis directly: **single-frame vs temporal stack, as β rises.** If
   the gap widens with fog — which the physics predicts and WACV 2025 corroborates — that is a
   genuinely novel, publishable result and it converts the owner's biggest measurement gap into his
   strongest claim.
4. Import **PDFT** (§4 H7) for the sim→real transfer, where its measured value was **+10.0 mAP**.

---

## 7. Bottom line

SpeckLock's mechanisms are not amateurish — the core trick won a CVPR 2025 challenge track and was
independently published in Sensors 2024, its metric choice is backed by TPAMI 2025 and an MVA 2025
competition metric, its NWD implementation is more correct than a 2026 arXiv paper's, and its
one-model-many-datasets result is something no published method has demonstrated.

What is amateurish is **the evidence packaging**: a headline of 1.000 on one video, comparisons
across three simultaneously-varying axes (metric, dataset, split), no seeds, and no baseline ever run
in-house. Every one of those is fixable with re-scoring and inference runs — no new architecture, no
GPU-weeks — and fixing them is worth far more than any of the hybridisation proposals in §4.
