# SpeckLock — Data & Evaluation Briefing

**Date:** 2026-08-12 · **Hardware assumed:** 1× RTX 5070 Laptop (8 GB VRAM), 32 cores, 30 GB RAM, **610 GB free** (measured: `/dev/mapper/ubuntu--vg-ubuntu--lv 935G 279G 610G 32% /`) · Torch 2.11 cu128.

**Provenance rule for this document.** Everything below is a synthesis of (a) the deduplicated literature sweep supplied to me and (b) measurements I could make locally against this repo. **I did not re-fetch any dataset URL myself.** The `Verified-live?` column therefore reports *what the sweep confirmed*, not what I confirmed. Anything the sweep flagged as snippet-derived, paywalled, or unfetched is marked **UNVERIFIED** and must not be cited without opening the source.

Legend for `Verified-live?`:

| Mark | Meaning |
|---|---|
| **Y** | Download route was fetched/confirmed during the sweep; open or one-click |
| **FORM** | Route exists but is gated: signed agreement, email request, or web form |
| **?** | Listed in the sweep but the *download route itself* was never fetched — assume nothing |
| **N** | Confirmed dead, pending release, or restricted-access |

Size figures marked `(est)` are my arithmetic from frame count × resolution, not published numbers.

---

## 1. Dataset table

Grouped by what each set can actually *prove* for SpeckLock. The columns are constant across groups.

### 1A. RGB air-target video — SpeckLock's actual operating point

These are the only sets where the full pipeline (stabilise → 3-moment stack → track → track-classify) can run end to end on the correct sensor and the correct target class.

| Name | Year | Frames | Resolution | Target px | Camera motion | Birds? | Weather? | Official metric | Download URL | Licence / form | Size GB | Verified-live? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ARD-MAV** (GLAD/MGMD) | 2024 | 60 videos / **106,665–107,497** frames | 1920×1080 @30 | **6×3 → 136×75 px**, mean 0.02 % of frame | **Moving** (onboard MAV, violent) | No | No | AP / P / R / F1 at **IoU 0.25**; fixed 15-video test list | https://github.com/WestlakeIntelligentRobotics/Global-Local-MAV-Detection | Open (MIT repo); Google Drive + Baidu `z1xb` | ~30–60 (est) | **Y** |
| **ARD100** (YOLOMG) | 2025 | 100 videos / **202,467** frames | 1920×1080 @30 | mean area **0.01 % of frame**; 42.18 % < 12×12 px | **Moving** (air-to-air) | No | Low-light + strong-light test videos | **AP@0.5** (IoU), 65 train / 35 test videos | https://github.com/Irisky123/YOLOMG → BaiduYun `pan.baidu.com/s/1ycAoKbzQ1rlzvKr8VRakgw?pwd=1x2z` (code `1x2z`) | GPL-3.0 code; **BaiduYun only, no mirror** | ~80–150 (est) | **Y** (link seen; Baidu friction) |
| **Drone-vs-Bird / DDS** (WOSDETC) | 2025 (8th ed.) | 77 videos / **~104,760** frames, mean ~1,384 f/video | 720×576 → 3840×2160 | ≥15 px, mean object ~34×23 px (~0.1 % frame); size bins 19.9 / 30.1 / 22.8 % in 0–12² / 12²–20² / 20²–32² | **Mixed** — static + moving; the 3 moving-camera sequences are where every method collapses | **Present but NOT annotated** | Weak contrast, long range, reduced visibility | **AP @ IoU 0.5** (recent eds.); 2020 ed. used a penalty metric (see §4) | https://github.com/wosdetc/challenge | **FORM** — email `wosdetc@googlegroups.com`, sign non-commercial usage agreement; budget ~1 week | ~40–80 (est) | **FORM** |
| **NPS-Drones** | 2019 | ~50 clips (exact frame count not in sweep) | not stated in sweep | few-px air-to-air | **Moving** | No | No | AP@0.5 | annotations via https://github.com/mwaseema/Drone-Detection and https://github.com/tusharsangam/TransVisDrone | Open | ~10–30 (est) | **?** |
| **FL-Drones** | 2017 | not stated | low-res | few-px, extreme motion blur | **Moving** | No | No | AP@0.5 — **no canonical split** | via TransVisDrone conversion scripts | Open | small | **?** — sweep marked `verified:false` |
| **MOT-FLY** | 2022 | 16 seq / **11,186** frames (8 train / 8 test) | 1920×1080 | 1–3 UAVs, 5–100 m range, "<5 % of image" (no px histogram published) | **Moving** (DJI Mavic) | No | Morning→evening | **MOTChallenge**: MOTA / IDF1 / HOTA | https://github.com/CZC-123/MOT-FLY | Apache-2.0; Google Drive + Baidu `pe53` | ~5–10 (est) | **Y** |
| **AOT** (Amazon Airborne Object Tracking) | 2021 | 4,943 flights / **5.9 M+** images / 3.3 M+ annotations | 2448×2048 **8-bit GRAYSCALE** | collision-course airborne targets, few px | **Moving** (airborne) | **YES — labelled** (bird / aircraft / drone types) | No | AIcrowd challenge metric (see §4 — UNVERIFIED) | `aws s3 ls --no-sign-request s3://airborne-obj-detection-challenge-training/` | **CDLA-Permissive-1.0** (commercial OK), **no account, no form** | **~11 TB full; ~500 GB `partial=True`** | **Y** |
| **XS-VID** | 2024 | 38 seq / **12,230** frames | 1024×1024 | median **18×18 px**; es(0–12²) 19.3 %, rs(12²–20²) 36.6 % | **Moving** (DJI Air3, 70–90 m alt.) | No | **Day and night** | COCO AP + APes/APrs/APgs size bands | https://github.com/gjhhust/YOLOFT | Open | ~5–15 (est) | **Y** |
| **UAV-Anti-UAV** (MambaSTS) | 2025 | **1,810 videos**, 1 M+ frames | not stated | tiny, dual-dynamic | **Moving** (UAV chasing UAV) | No | 15 attribute labels | SOT metrics (AUC / precision) | https://arxiv.org/abs/2512.07385 | **?** release not confirmed | ? | **?** |
| **USC-Drone** | 2017 | 60 videos / ~20 K frames **(sources conflict: 30 / ~27 K)** | 1920×1080 | — | static-ish | No | No | — | no live canonical URL found | — | small | **N** — do not cite |

### 1B. Bird / false-positive sets — the only places an FP rate can be *measured*

| Name | Year | Frames | Resolution | Target px | Camera motion | Birds? | Weather? | Official metric | Download URL | Licence / form | Size GB | Verified-live? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Halmstad Drone Detection** | 2021 | **650 videos** (365 IR + 285 visible) / **203,328** annotated frames + 90 audio clips | survey table says 640×512 visible / 320×256 IR — **conflicts with expectation, verify on download** | not documented | static (airport rigs) | **YES — 4 classes: drone, BIRD, airplane, helicopter** | **Night** | **None defined** — no official split, no stated metric | https://github.com/DroneDetectionThesis/Drone-detection-dataset · Zenodo DOI `10.5281/zenodo.5500576` | **CC0-1.0 public domain — no form, no agreement** | not published | **Y** |
| **SMOT4SB** (MVA 2025) | 2025 | 211 videos / **108,192** frames / 371,690 instances / 2,240 IDs | 1920×1080 + 3840×2160 @30 | majority **< 32×32 px** | **Moving** (UAV-borne) | **Birds are the TARGET class** (perfect hard-negative corpus for a drone detector) | No | **SO-HOTA** (HOTA with DotD ↔ IoU) | https://github.com/IIM-TTIJ/MVA2025-SMOT4SB | Open; splits 128 / 38 / 45 videos | ~30–60 (est) | **Y** |
| **Anti2** | 2024 | **5,062** stills / 5,828 boxes | not stated | **verify — not confirmed tiny** | static | **YES — 1,444 bird** + 1,406 helicopter + 1,320 airplane + 1,688 UAV | No | none stated (paper used DotD-YOLOv9-C) | https://github.com/gdpinntit/-anti-interference-and-anti-UAV-dataset | Open, no form | <1 | **Y** |
| **AOD4** | 2024 | **22,516** stills, ~7,900 ann./class | mixed | not characterised | static | **YES — 4 classes incl. bird** | No | none stated; split 15,761 / 4,514 / 2,241 | https://data.mendeley.com/datasets/cd5z895tr2/1 · Roboflow mirror | Open, no form | ~5–15 (est) | **Y** |
| **LRDDv3** | 2026 | **102,532** RGB + 29,630 IR stills from 128 clips, **sampled at 5 FPS** | 4K RGB / 640×512 IR | boxes down to 12 px; **per-target ground-truth RANGE 0–200 m** | Dynamic camera movement included | **YES — 6,031 bird + 722 airplane** vs 93,652 drone | **YES — 24,356 rain / 1,211 snow / 41,015 cloudy** | mAP50 / mAP50-95 / F1 | https://research.coe.drexel.edu/ece/imaple/lrddv3/ | **CC BY 4.0**, request form | **~200–400 (est)** for 4K | **FORM** |
| **LRDDv2** | 2025 | **39,516** stills, >8,000 with range | 1080p | majority ≤ 50 px | **Moving sequences included** | not stated | sunny / cloudy / rainy, glare, occlusion | mAP@50 (IoU) | https://research.coe.drexel.edu/ece/imaple/lrddv2/ | request | ~30–60 (est) | **FORM** |
| **VIP Cup 2025** (ICIP) | 2025 | ~64,500 paired RGB+IR stills (45 k / 6.5 k / 13 k) | **320×256** | proportionally large at this resolution | not stated | **YES — ~30 k bird vs ~36 k drone (RGB)** | **fog, mist, motion blur, uneven illumination, AWGN — graded to severity 5/5** | Accuracy / F1 / Precision / Recall (**not IoU-mAP**) | https://www.kaggle.com/competitions/2025-ieee-icip-vip-cup | Kaggle rules | ~5–15 (est) | **Y** |
| **BirDrone** (YOLOBirDrone) | 2026 | **11,495** stills (3,067 lifted from DvB) | mixed | **skewed LARGE**: 14,553 objects >96×96 vs 2,705 <32×32 | static | **YES — 15,867 bird vs 13,881 drone** | No | mAP50 / accuracy / **FP %** | https://github.com/dapinderk-2408/YOLOBirDrone | "Complete data available soon" — **sample folder only** | small | **N** |
| **STBRNN set** | 2025 | **20,925** stills | 640×640 | **YOLOv7-pre-cropped, LARGE well-resolved targets** | — | **YES** | No | P/R/F1 + **raw FP/FN counts** | https://data.mendeley.com/datasets/6ghdz52pd7/5 | Open | ~2–5 (est) | **Y** |
| **LAT-BirdDrone** | 2025 | 65 videos / **33,262** images → **665 trajectories** (350 bird / 315 drone), mean 48 frames | 1920×1080 vis + 640×512 IR | — | not stated | **YES — 4 bird species vs 4 DJI models** | **rain, snow, sandstorm** | trajectory-classification accuracy / F1 / ROC-AUC | https://www.scidb.cn/anonymous/ZkFWRlJ2 | Open | ~5–15 (est) | **Y** |

### 1C. Hard-conditions sets (fog / haze / cloud / night / glare)

| Name | Year | Frames | Resolution | Target px | Camera motion | Birds? | Weather? | Official metric | Download URL | Licence / form | Size GB | Verified-live? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **HazyDet** (+RDDTS) | 2024/26 | **11,600** images / 383 K instances (11,000 synthetic + **600 real**) | drone-view | 4+ octaves, long tail at 64–256 px² | drone-view (looking **down**) | No | **Haze — synthetic + real (RDDTS)** | mAP on synthetic test **and** real RDDTS, both reported | https://github.com/GrokCV/HazyDet | **Apache-2.0**, live leaderboard | ~10–20 (est) | **Y** |
| **Task-Driven Synthetic Fog** (UAV) | 2026 | CfAR 5,834 det. images + DUT + MMAUD | mixed | small | mixed | No | **Depth-aware fog, β ∈ {0.4, 0.8, 1.6, 2.4, 3.6}** | mAP@0.50 + MOTA | Zenodo DOI `10.5281/zenodo.20615750` | Open | ? | **Y** |
| **RGBT-Tiny** | 2025 | 115 seq / **93 K** frames / 1.2 M boxes **with track IDs** | 640×512 @15 | **>81 % < 16×16 px**; 48 % in 8²–16² | **Static** rigs | drone is a class; bird-like classes present | **33.9 % night, ~70 % low-light** | **SAFit** (see §4) | https://github.com/XinyiYing/RGBT-Tiny | **FORM** (Google/MS Forms) | ~20–40 (est) | **FORM** |
| **UAV-CB** | 2026 | **3,442** RGB-T pairs | not stated | most UAVs <5 % of image | static | No | **"clouds" is one of 5 background categories** | AP50 / AP75 / AP@.5:.95 | none given in paper | **N — no URL** | small | **N** |
| **XWOD** | 2026 | **10,010** real images / 42,924 boxes | mixed | mean rel. box area 0.0316 | dashcam | No | **7 weather types incl. fog, haze/sand/dust, wildfire smoke** | mAP50 / P / R per condition | https://www.kaggle.com/datasets/kuantinglai/exwod | Kaggle | ~5–10 (est) | **Y** |
| **DAWN** | 2020 | **1,000** real images / 7,845 boxes | mixed | large vehicles | dashcam | No | fog / snow / rain / sandstorm | COCO mAP | https://ieee-dataport.org/documents/dawn-vehicle-detection-adverse-weather-nature | Open | <1 | **Y** |
| **RTTS** (RESIDE) | 2018 | **4,322** real hazy images | mixed | ordinary-scale | static | No | real haze | mAP50 | RESIDE benchmark | Open | ~1 | **?** — sweep marked `verified:false`, arXiv ID inferred |
| **SynDroneVision** | 2025 | **140,038** synthetic images, **recorded sequentially** | 2560×1489 | drone targets | UE5 + Colosseum | No | sun/sky, time-of-day varied; **no dense fog, no rain particles, no volumetric cloud** | detection mAP | https://zenodo.org/records/13360116 | Open (Zenodo) | ~100–200 (est) | **Y** |
| **Nano-VID / Nano-VID-weather** (TNO) | 2024/25 | 3,968 train / 388 test frames | — | avg **16.42 px** | static | No | **REAL wind / rain / haze subsets** | mAP | — | **N — proprietary, in-house** | — | **N** |

### 1D. Infrared multi-frame small-target sets — where the temporal literature actually lives

SpeckLock's grayscale-moment stack ports to IR unchanged (it only needs single-channel intensity + a stabiliser). These are cheap, small, and have very live 2026 leaderboards in the literature.

| Name | Year | Frames | Resolution | Target px | Camera motion | Birds? | Weather? | Official metric | Download URL | Licence / form | Size GB | Verified-live? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **IRSTD-UAV** (TDCNet) | 2026 | 17 seq / **15,106** frames | not stated | small UAV | **requires GIM registration as preprocessing** | No | **buildings / trees / CLOUDS clutter** | P / R / F1 / AP50 | https://github.com/IVPLabs/TDCNet (Google Drive + Baidu) | Open | ~2–5 (est) | **Y** |
| **DAUB** | — | **16,177** frames, 22 segments, 16,944 targets (8,983 train / 4,795 test) | 256×256 MWIR | **1–10 px** | **Static** | No | No | **5-px centre-distance TP** + mAP50/F1 (see §4) | via IR-small-target repos | Open | <2 | **?** |
| **Anti-UAV410** | 2023 | 410 seq / **438 K** boxes | 640×512 TIR | frame-level attributes, full published list: **Thermal Crossover**, Out-of-View, Scale Variation, Fast Motion, Occlusion, Dynamic Background Clutter, Tiny/Small/Medium/Normal Size ✅ (verified against the repo README 2026-08-12) | static | No | **35 % sudden lighting change, >60 % occlusion**; **Thermal Crossover** is a first-class labelled failure mode — the target's radiance matches the background and it disappears | **State Accuracy** (see §4) | https://github.com/HwangBo94/Anti-UAV410 (Drive + Baidu `a410`) | Open | ~30–60 (est) | **Y** |

> **Why the `Thermal Crossover` attribute is worth knowing even though this project is RGB.** The
> standard reply to "just use a thermal camera and this is trivial" is that the field's own thermal
> benchmark ships a labelled attribute for *the target vanishing against the background*. Thermal
> moves the failure mode rather than removing it. Cite the attribute list above, not a general
> impression.
| **Anti-UAV600** | 2023 | 600 seq / **~723 K** frames | 640×512 TIR | tiny | static | No | — | State Accuracy | ModelScope `ly261666/3rd_Anti-UAV` | Open, slow from outside CN | ~40–80 (est) | **?** |
| **4th Anti-UAV Challenge set** | 2025 | T1/T2: 223 train + 216 test videos; **T3: 300 seq** (0–40+ targets/frame) | 640×512 TIR (Anti-UAV300 adds 1920×1080 RGB) | tiny-scale explicitly added | static | No | dynamic backgrounds | State Accuracy (T1/2), **MOTA** (T3) | Zenodo `15103888` (train) / `16299533` (test) | **N — Zenodo files RESTRICTED**; use Drive/Baidu | **1.8 TB total** | **N** for Zenodo |
| **Anti-MUAV15** | 2025 | 15 seq / **16,269** frames | 640×512 | avg **12–23 px** | static (assumed) | No | buildings/sky/trees | none stated | https://github.com/Shihan0325/Anti-MUAV15 (Drive + Baidu `r72r`) | no licence stated | <2 | **Y** |
| **CST Anti-UAV** | 2025 | 220 seq / **240 K+** boxes; 78,224 objects <10 px diagonal | 640×512 TIR @25 | **the smallest published TIR set** | static | **YES — birds annotated as "Complex Dynamic Background" distractors** | year-long: day/night, sunny/cloudy, wind, **fog** | mSA / P(AUC) / S(AUC) | — | **N — "about to be released"** | ~20–40 (est) | **N** |
| **TDTIV** (MCATrack) | 2025 | **290 K** frames / 280 K boxes | TIR | ≤20×20 px | static | No | heavy clutter | tracking | https://github.com/zhangjiahao02/MCATrack | **N — repo returned 404** | ? | **N** |
| **MIST** | 2026 | 78 train / 42 test seq; **MIST-Hard** = 11 seq at SCR ≤1 and **>7 px/frame target speed** | synthetic IR | tiny | — | No | low SCR | not extractable from README | https://github.com/GR-ray/MIST | Open | ? | **Y** |
| NUDT-MIRSDT / IRDST / ITSDT-15K | 2023–25 | multi-frame IR standards | — | 1–10 px | static | No | — | **IoU / nIoU / Pd / Fa** (segmentation-style) | index: https://github.com/Tianfang-Zhang/awesome-infrared-small-targets | Open | small | **Y** (index) |

### 1E. Generic tiny-object stills — for loss/matching validation only, **not** for the temporal claim

| Name | Year | Frames | Resolution | Target px | Camera motion | Birds? | Weather? | Official metric | Download URL | Licence / form | Size GB | Verified-live? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **AI-TOD-v2** | 2022 | **28,036** images / 752,745 instances (11,214 / 2,804 / 14,018) | 800×800 tiles | **mean 12.7 px ± 5.6** | none (satellite/aerial) | No | No | COCO AP + **APvt / APt / APs / APm** via `cocoapi-aitod` | https://chasel-tsui.github.io/AI-TOD-v2/ (annotations on Drive; base imagery separate) | Open | ~10–20 (est) | **Y** |
| **SODA-A / SODA-D** | 2023 | A: 2,513 img / 872,069 oriented inst.; D: 24,828 img / 278,433 inst. | high-res | defines es/rs/gs size bands | none | No | No | COCO AP + APes bands | https://shaunyuan22.github.io/SODA/ (OneDrive/Baidu, pw `SODA`) | **CC BY-NC 4.0 — academic only** (conflicts with AGPL-3.0 commercial story) | ~20–50 (est) | **Y** |
| **DroneSwarms** | 2024 | **9,109** stills (6,577/2,532) / 241,249 objects | 1920×1080 | **mean 7.9 px**, 99.60 % ≤32×32, **26.59 drones/image** | not stated | No | clear/overcast/**dusk/backlighting** | none stated | https://hiyuur.github.io/ | **FORM** — signed application emailed to `yaohaiyu@tju.edu.cn` | ~5–15 (est) | **FORM** |
| **Det-Fly** | 2021 | **13,271** stills | 3840×2160 | air-to-air Mavic2 | **stills only — no sequences** | No | 4 backgrounds × 3 view angles (factorial) | mAP50 | https://github.com/Jake-WU/Det-Fly | MIT; OneDrive `4dcc` / Baidu `qjyt` | ~35–50 (est) | **Y** |
| **DUT Anti-UAV** | 2022 | 10 K detection images + 20 tracking videos | mixed RGB | **relatively LARGE** | static | No | No | mAP50 | https://github.com/wangdongdut/DUT-Anti-UAV | Open; Drive codes `u955`/`wkzs`/`ik4d`/`oine`/`e8mr` | ~5–15 (est) | **Y** |
| **TIB-Net** | 2020 | **2,860** stills | 1920×1080 | tiny, ~500 m range | **Static ground rig** (clean control for the stabiliser) | No | **day → night** | none stated | https://github.com/kyn0v/TIB-Net (Drive) | licence unstated | <2 | **Y** |
| **UETT4K** | 2025 | **33,601** stills | 4K | not documented | not documented | not mentioned | "challenging conditions" | none stated | https://github.com/mugheessarwarawan/UETT4k-Anti-UAV (SharePoint) | **MIT**; fragile SharePoint link — mirror immediately | ~100–200 (est) | **?** |
| **MIDGARD** | 2020 | ~7,595–9,884 images, 14 environments | mixed (180° and 96° FoV) | **large — does NOT test 3–14 px** | moving | No | sunlight → very low light | none; annotation carries **distance + σ** | https://mrs.fel.cvut.cz/midgard (**note: `mrs.felk.cvut.cz` fails TLS**) | unstated | <5 | **Y** |
| **TinySet-9M** | 2026 | 9 M+ annotations, 6 domains | mixed | mean **20.4 px** | mixed | No | No | AP0.5 (point-prompted protocol) | https://zhuhaoraneis.github.io/TinySet-9M/ | ? | very large | **?** |
| **UAVNet-MS** | 2026 | 15,618 RGB+MSI cubes | — | **93.7 % ≤32×32** | — | No | — | — | no URL in abstract | **N** | ? | **N** |

### 1F. Event-camera sets — **idea sources only, not runnable on this hardware**

| Name | Year | Scale | Target px | Why listed | Verified-live? |
|---|---|---|---|---|---|
| **EV-UAV** | 2025 | 147 seq / 2.3 M event annotations | **6.8 × 5.4 px avg** — dead centre of SpeckLock's band | STC motion-continuity **loss** is transplantable to the temporal stack; reports **Fa** as a first-class metric | **Y** (https://github.com/ChenYichen9527/Ev-UAV) |
| **M2E-UAV** | 2026 | 87,223 train / 21,395 val | tiny | Names SpeckLock's regime **"motion-on-motion"**; ships synchronised **IMU** → IMU-primed homography idea | **?** repo may be empty |
| **FRED** (Florence RGB-Event) | 2025 | 7+ h, 5 drone models | tiny | **Has an RGB stream** + rain/night/adverse lighting → usable half | **?** counts not extracted |
| **SkyEV** | 2026 | uncompressed RGB + event, strong ego-motion | extremely small | Uncompressed frames for the same reason SpeckLock dumps per-frame JPEGs | **?** no URL |

---

## 2. Prioritised acquisition order (610 GB, one 8 GB GPU)

**The strategic frame.** The single strongest available claim is *not* "we beat SOTA on a saturated benchmark". It is:

> On the two public benchmarks where the camera moves and the target is 3–14 px — the only regime where a temporal input representation can matter — the published state of the art is **AP 0.55** (MGMD on ARD-MAV) and **AP 0.78–0.85** (YOLOMG on ARD100), and neither method does ego-motion-compensated multi-lag stacking. Separately, no published work has ever reported a **bird false-alarm rate** on drone-vs-bird data, because the bird boxes were never labelled.

Both halves of that are winnable on this hardware. Acquire in this order.

### Day 0 (before any download): fire the two slow requests

Both of these have multi-day human latency and cost nothing to start:

1. **Email `wosdetc@googlegroups.com`** for the Drone-vs-Bird DDS usage agreement. This is the reputational benchmark and the long pole — budget a week, possibly more.
2. **Email the LRDDv3 authors** (Drexel iMaPLe, via https://research.coe.drexel.edu/ece/imaple/lrddv3/) asking for **the source clips, not the 5 FPS frame sampling**. LRDDv3 is the only set with birds *and* rain/snow *and* ground-truth range, and nobody has run a temporal method on it precisely because the public release is frame-sampled at 5 FPS (a 12-frame lag = 2.4 s of motion, which destroys the stack). That request is also a collaboration hook.

### Priority 1 — **ARD-MAV** (~30–60 GB) — *the headline claim*

- **Why first:** RGB, 1080p, genuinely moving camera, targets from **6×3 px**, 107 K frames, MIT-licensed, direct download, and the published SOTA (MGMD, T-ITS 2024) is only **AP 0.55 / recall 0.59 / F1 0.69**. That is enormous headroom on the exact axis SpeckLock is built for.
- **It ships a published test split and a published difficulty stratification**: test = `phantom{05,08,09,10,19,30,41,43,46,47,58,63,65,70,86}`, stratified as Ordinary {09,10,30,47,70}, Complex {05,08,58,65,86}, Small Objects {19,41,43,46,63}. That gives a per-condition table for free — exactly the format SpeckLock's reports already use.
- **⚠ This repo has already used ARD-MAV and used the WRONG split.** See §5.1 — this is the single most urgent correction in the whole briefing, and re-acquiring the data is the prerequisite for fixing it.
- **Cost check:** ~107 K frames at 1080p; fits trivially. Training tiles at 640 px on an 8 GB card is already a solved path in this repo (`tools/make_dataset_external.py --task combined-tiled`).

### Priority 2 — **Halmstad Drone Detection** (CC0, size unpublished, likely <50 GB) — *the false-positive claim*

- **Why:** it is the **only** dataset in the entire sweep that is simultaneously (a) **video**, so the temporal stack runs natively, (b) carries **labelled bird / airplane / helicopter** distractors, (c) includes **night**, and (d) is **CC0-1.0 public domain with no form, no agreement, no export gate**.
- This is where the audit's strongest local finding becomes a public claim. Measured on this repo today: **99.8 % drone recall with 0 hits on 934 bird instances at 0.002 FP/frame** for the track-integrated PC-MAX pipeline, versus **151 bird hits** for the raw detector at 95 % recall. That is a real, defensible mechanism result — but it currently rests on **8 bird tracks, one flock, one afternoon**. Halmstad turns an existence proof into a measurement.
- **Friction to budget:** labels are MATLAB-format (`mcos-decoder`), the video manifest is an `.xlsx`, and there is **no official split or metric** — you must define and *publish* your own split file, or the number is not reproducible by anyone else.

### Priority 3 — **ARD100** (~80–150 GB) — *the nearest-relative head-to-head*

- **Why:** YOLOMG is SpeckLock's closest published relative — pixel-level motion fused with RGB — with **released GPL-3.0 code**, a published ablation, and the tiniest targets of any drone video set (42.18 % under 12×12 px). Its own ablation is the argument SpeckLock needs: "RGB + RGB" instead of motion collapses **AP 0.78 → 0.33**, and 3-frame differencing beats 2-frame (0.78 vs 0.73).
- **The clean experiment:** YOLOMG needs **1280 px input** to reach 0.85 and drops to 0.78 at 640 — because it does frame *alignment*, not explicit homography stabilisation, and pays for the residual with resolution. Adding SpeckLock's stabiliser in front and showing 640 px reaches 1280 px accuracy is a single-variable, publishable ablation that costs one training run.
- **Friction:** **BaiduYun only**, no mirror. Start the download early and expect pain.

### Priority 4 — **SMOT4SB** (~30–60 GB) — *the metric defence + an open leaderboard*

- **Two payoffs for one download.** (a) It is a **live, permanently-open Codabench leaderboard** (see §3) with baseline SO-HOTA 9.90 and winner 50.59 — a 5.1× gap on moving-camera tiny-target tracking where **no entrant used ego-motion-compensated temporal stacking**. (b) 108,192 frames of nothing but small birds from a moving camera is the **perfect pure false-positive stress corpus**: "our drone detector fires N times across 108 K frames containing zero drones" is a number nobody in this field has published.
- It also supplies the peer-reviewed cover for the centre-distance metric (see §4.4), alongside RGBT-Tiny's SAFit (TPAMI 2025).

### Priority 5 — **Drone-vs-Bird / DDS** (~40–80 GB, whenever the agreement clears)

- The name-recognition benchmark. Beat Laroca's 0.7390 mean mAP50 — and note *where*: `gopro_002` at 0.4491 and `dji_phantom_4_hillside_cross` at 0.4992 are the moving-camera / cluttered cases, and hillside actually **regressed** under their tiling (0.7406 → 0.4992). Those two videos are SpeckLock's design point.
- **The genuinely novel contribution here is not a leaderboard row.** It is: *label the birds* on the public DDS videos and publish the **first bird-specific false-alarm rate** in the challenge's nine-year history. The organisers' own 2021 review leaves bird FPs undefined. That is a dataset contribution, cheap, and unimpeachable.

### Optional 6 — **IRSTD-UAV** (~2–5 GB) — *the cheapest possible external win*

17 sequences, 15 K frames, downloadable today, cloud/tree/building clutter, and the AAAI-2026 SOTA (TDCNet) sits at F1 97.12 / AP50 93.83 with MOCID one point behind. Small enough to train overnight. Its ablation — hand-made temporal difference F1 92.25 → learned re-parameterised TDC F1 96.76 — is the exact "is channel stacking naive?" question, answered on data you can hold.

### Disk budget

| Slot | Dataset | GB (est) | Running total |
|---|---|---|---|
| 1 | ARD-MAV | 60 | 60 |
| 2 | Halmstad | 50 | 110 |
| 3 | ARD100 | 150 | 260 |
| 4 | SMOT4SB | 60 | 320 |
| 5 | Drone-vs-Bird DDS | 80 | 400 |
| 6 | IRSTD-UAV | 5 | 405 |
| — | working space: tiles, runs, engines, renders | ~150 | **~555 / 610** |

### Explicitly **do not** acquire

| Set | Reason |
|---|---|
| **AOT** | 11 TB full / ~500 GB "partial" — eats the entire budget for grayscale collision-course aircraft. Revisit only if the bird-FP claim needs a second corpus, and then stream a subset. |
| **4th Anti-UAV challenge set** | **1.8 TB**, Zenodo restricted, IR-only, and the competition is closed. |
| Any **event-camera** set (EV-UAV, M2E-UAV, SkyEV, FRED-event half) | No sensor. Cite as convergent evidence; do not download. |
| **VisDrone / VisDrone-VID** | Wrong problem — a drone looking *down* at cars. The "static world cancels to grey" assumption fails over a moving ground plane. Say this explicitly when asked why SpeckLock isn't on VisDrone. |
| **Det-Fly**, **DroneSwarms**, **AI-TOD-v2**, **SODA** | Stills. The temporal stack **cannot run**. Use only for the single-frame branch, the NWD/GCD loss ablation, or the matching-rule argument — never as evidence for the temporal claim. |
| **SODA-A/D** if commercialisation ever matters | **CC BY-NC 4.0**, academic-only, against an AGPL-3.0 repo. |
| **DUT Anti-UAV** as a headline | mAP50 0.971 means the targets are large. Reporting it *alone* invites the comparison that makes 0.83 look bad. Report it only alongside ARD-MAV/ARD100 with an explicit "this is the easy end" sentence. |

---

## 3. Live competitions and leaderboards — what can actually be entered today (2026-08-12)

### OPEN NOW

| Venue | Platform / URL | Phase window | Task & metric | Data | Notes |
|---|---|---|---|---|---|
| **Beyond Strong Baseline: Multi-UAV Tracking (LTS)** | https://www.codabench.org/competitions/16223/ | **Main phase 2026-05-10 → 2030-05-10 — status Current** | Thermal-IR multi-UAV detection+tracking; **HOTA / MOTA / IDF1**; submit a zip of MOT-format `.txt` | Train: 200 videos, Zenodo `10.5281/zenodo.15103888`; Test: 98 seq, Zenodo `10.5281/zenodo.16299533` | **The single most actionable venue.** Live board (fetched 2026-08-12): 1) endives-packet HOTA **0.9316**, 2) ciwei6 0.9187, 3) us-ias 0.9187; "Strong Baseline" 0.8808. Scenario mix is SpeckLock's hard-condition list verbatim: **39 cloud-background videos**, 18 cloud-with-fewer-UAVs, 68 tree-background, 11+4+17+8 building-background, takeoff/landing. **No external data allowed** for badge eligibility. Baseline code: https://github.com/wish44165/YOLOv12-BoT-SORT-ReID (author ran it on an RTX 4050 6 GB — strictly weaker than this box) |
| **MVA 2025 SMOT4SB — Post-Competition** | https://www.codabench.org/competitions/5101/ (phase 8582) | **Status Current, no end date** | Small multi-object tracking from a moving UAV; **SO-HOTA / SO-DetA / SO-AssA** | https://github.com/IIM-TTIJ/MVA2025-SMOT4SB | Public post-comp board (fetched 2026-08-12): xl123 **55.462**, abandoned 51.387, s113022 49.320 — i.e. already above the 50.59 private-test winner. Original baseline 9.90. |
| **MVA 2023 SOD4SB — Post-Competition** | https://codalab.lisn.upsaclay.fr/competitions/9594 | Started 2023-04-28, **end date "Never"** | Single-frame small-bird detection, AP-style | https://github.com/IIM-TTIJ/MVA2023SmallObjectDetection4SpottingBirds | Three-way tie at **0.777**. Single-frame by construction — its real use is as the controlled ablation twin of SMOT4SB. |
| **Seasons in Drift / LTDv2** | https://www.codabench.org/competitions/16106/ | **2026-01-01 → 2030-01-01** | Thermal detection **robustness across months**; ranked by **mAP Balance** + coefficient of variation | >1 M frames, single scene | Live board (fetched 2026-08-12, 17 subs): best andreasaa mAP-Balance **0.4832**, CoV 0.1053. Ground-level classes (person/bicycle/motorcycle/vehicle), not drones — value is the *robustness-first ranking* precedent. |
| **MaCVi 2026 Thermal Object Detection** | https://arxiv.org/abs/2604.13244 · https://macvi.org/dataset | **still accepting after the workshop** | COCO AP (AP50 tiebreak) + AP75/AR1/AR10; **3 submissions/day** | Maritime Collision Avoidance Dataset | Judged on accuracy **and embedded real-time feasibility** — the same argument `realtime/` makes. Prize: NVIDIA RTX 5080. Related **SeaDronesSee** eval servers (det / SOT / MOT) are also open at macvi.org/dataset. Note: 2026 dropped the UAV maritime tracking track. |
| **HazyDet leaderboard** | https://github.com/GrokCV/HazyDet | rolling, repo-hosted | mAP on synthetic test **and** real RDDTS | Apache-2.0 | Not a submission server, but a maintained public table with a real-hazy split. **No temporal method appears anywhere on it.** |
| **CSIG UAV Video Arbitrary-Scale SR** | https://www.codabench.org/competitions/17147/ | Register closed 2026-08-15; **testing 2026-08-15 → 2026-08-19** | 0.60·PSNR + 0.25·LPIPS + **0.15·tLP** | 12,000 frames | Restoration, not detection, and effectively domestic-China. Listed only because it is live *and* because its 15 % temporal-consistency term concedes that per-frame SR flickers — corroborating SpeckLock's own "SR on crops: no gain over bicubic" negative result. |

### CLOSED / DORMANT — do not plan around these

| Venue | Status as of 2026-08-12 |
|---|---|
| **Drone-vs-Bird (WOSDETC), 8th ed.** | Closed Feb 2026 cycle (test release 2026-02-10, submission 02-13). **No 9th edition announced anywhere.** The wosdetc GitHub still lists 2023 as latest. Data remains obtainable by agreement; there is no live server. |
| **4th Anti-UAV Challenge (CVPR 2025)** | All three CodaLab servers (21688 / 21690 / 21806) **ended 2025-03-09**. No 5th edition — the full CVPR 2026 workshop list was enumerated and contains no Anti-UAV workshop. Offline evaluation on Anti-UAV410 with the published protocol is the only route. |
| **VIP Cup 2025 (ICIP)** | Closed. Winners: NeuronX (BUET) $5,000. No per-team leaderboard published. Reproduce EGD-YOLO / SpectraSentinel as proxies. |
| **UG2+ 2026 (CVPR)** | Closed 2026-05-13, and none of the three 2026 tracks is detection (restoration / segmentation / turbulence). Watch CVPR 2027. |
| **PBVS 2026 (CVPR)** | Phases closed 2026-03-01. Nothing is a tiny-drone detection track. |
| **URVIS OTRW-MMRGBT (CVPR 2026)** | Closed 2026-03-10. |
| **ACM MM 2026 UAVM / PairUAV** | Closed 2026-07-01. Navigation, not detection. |
| **AERO-HPR (CVPR 2026)** | No challenge at all; archival deadline 2026-03-13, non-archival 2026-04-20, both passed. Human-centric. |
| **LoViF @ ECCV 2026** | Restoration only; paper deadline 2026-07-25 passed. **Signal value: ECCV 2026 workshops are running now — watch its slate for a tiny-object/drone track.** |
| **ICPR 2024 LimitIRSTD** | Closed; no successor announced. ICPR 2026 (Lyon, 2026-08-17→22) competition list **not checked** — worth a manual look this week. |
| **VisDrone / AISKYEYE** | Dormant. 2024 leaderboard still reads "Coming soon". No open server. |
| **Frontex C-UAS Prize** | Closed Nov 2025. TRL ≥ 7 — a deployable system, not a detector. Structurally inaccessible. |
| **DIU PROJ00656 (Yuma)** | Solution briefs closed 2026-02-26. Vendor solicitation, US entity, no public metric. **UNVERIFIED** — the diu.mil deep link 404'd; dates are from news aggregators. |
| **ICMCIS / NATO** | **UNVERIFIED** — page 301-redirects to a NATO host that 403s automated fetch. It is radar/multi-sensor fusion anyway, not video. Kaggle slug: `icmcis-drone-tracking`. |
| **Kaggle "Drone Detection UPC 2025-2026"** | A university course exercise. No metric, no leaderboard reachable. **Ignore.** |
| **AIcrowd AOT challenge** | Status **not verified** in the sweep. The *data* is open on S3 regardless. |

**Bottom line on venues:** exactly **four** places can produce a public, dated, third-party-scored number in the next month — LTS Multi-UAV (Codabench 16223), SMOT4SB post-comp (Codabench 5101), SOD4SB post-comp (CodaLab 9594), and MaCVi Thermal OD. Of those, **LTS Multi-UAV is the one whose data (39 cloud-background sequences) matches the stated interest in hard conditions**, and SMOT4SB is the one whose *metric* vindicates SpeckLock's scoring.

---

## 4. Exact evaluation protocols — so numbers are comparable, not home-made

### 4.1 Drone-vs-Bird / DDS

- **Matching:** a detection is TP **iff IoU with a ground-truth box > 0.5**. Ranked by **AP** computed from the precision–recall curve at IoU 0.5.
- **Historic variant (do not confuse):** the original challenge scored with a *penalty* metric — the area of the smallest box enclosing both the true and predicted box, normalised by the GT box area, **minimum 1** (lower is better). Recent editions report AP. **State which you used.**
- **Split:** the official test annotations are **withheld**. Every paper reports on a self-chosen validation split. The de-facto comparable split is Laroca's: 70 train / 7 val videos = `dji_mavick_mountain`, `2019_10_16_C0003_3633_inspire`, `parrot_disco_distant_cross_3`, `GOPR5843_002`, `swarm_dji_phantom4_2`, `dji_phantom_4_hillside_cross`, `gopro_002`.
- **Reproducing the SOTA config exactly** (Laroca, IJCNN 2025): YOLO11m; SGD lr 0.01, momentum 0.9, batch 16, patience 15; **NMS IoU 0.1, conf 0.375**; tiles = **4 crops at 55 % of W and H** (e.g. 1056×594 from 1920×1080) plus the whole frame, predictions aggregated; temporal post-pass over **±6 frames**, matching on same class and **IoU ≥ 0.1**, interpolated boxes get **confidence / 2**. Their bird class is trained but **all bird predictions are discarded at inference**.
- **Annotation format:** one line per frame — `framenum num_objs obj_x obj_y obj_w obj_h obj_class`.
- **Known data defects:** frame/label desynchronisation (documented by Laroca, who published corrected annotations at https://raysonlaroca.github.io/supp/drone-vs-bird/). **Birds are present but unlabelled** — a bird FP scores as an ordinary FP, never as a confusion.

### 4.2 ARD-MAV (MGMD / GLAD)

- **Matching:** **IoU threshold 0.25** (not 0.5) — chosen precisely because 6×3 px boxes make 0.5 meaningless. Reports **P / R / F1 / AP** plus FPS.
- **Split:** official test = 15 videos, `phantom{05,08,09,10,19,30,41,43,46,47,58,63,65,70,86}`; the remaining 45 are train/val.
- **Difficulty strata (publish per-stratum, it is free):** Ordinary {09,10,30,47,70}; Complex {05,08,58,65,86}; Small Objects {19,41,43,46,63}.
- **SOTA to beat:** MGMD **P 0.84 / R 0.59 / F1 0.69 / AP 0.55 @ 28 FPS**. Baselines on the same split: YOLOv5s 0.86/0.27/0.41/0.26 @82; TPH-YOLOv5l 0.99/0.32/0.48/0.36 @11; Dogfight 0.77/0.24/0.36/0.24 @1; MEGA 0.26/0.37/0.31/0.23 @3.
- Annotations are **Pascal VOC XML**, one file per frame, 1-based index (`dronedet` already parses this: `tools/make_dataset_external.py:parse_ardmav`).

### 4.3 ARD100 / NPS-Drones (YOLOMG)

- **Matching:** plain **AP @ IoU 0.5**. Input resolutions reported separately at **640** and **1280**.
- **Split:** ARD100 = **65 train / 35 test videos**; the test set deliberately holds out unseen scenes plus low-light and strong-light videos.
- **SOTA:** ARD100 YOLOMG-1280 **0.85 @35 FPS**, YOLOMG-640 **0.78 @133 FPS**; next best YOLOv9 0.64, CFINet 0.63. NPS-Drones YOLOMG-1280 **0.95 @35 FPS**, TransVisDrone 0.95 @5 FPS, YOLOv5/v8 0.93.
- **The ablation to reproduce:** on ARD100 @640 — 2-frame diff k=1 → AP 0.73; 2-frame k=2 → 0.77; 3-frame k=1 → 0.75; **3-frame k=2 → 0.78**; **RGB+RGB (motion channel replaced by RGB) → 0.33**; without the small-object layer → 0.76.

### 4.4 SMOT4SB (MVA 2025) — **the citation that defends centre-distance scoring**

- **Metric:** **SO-HOTA** = HOTA with IoU replaced by **Dot Distance**: `DotD(A,B) = exp(−D / S)`, where `D` = Euclidean distance between box centres and `S` = mean object size. Averaged over **19 thresholds, 0.05 → 0.95**. Reported as SO-HOTA / SO-DetA / SO-AssA, with plain HOTA alongside.
- **Split:** 128 videos / 66,602 frames / 1,256 IDs train; 38 / 16,489 / 509 public test; 45 / 25,101 / 475 private test.
- **Reference scores:** private-test winner **50.59**, then 46.22 / 43.87 / 43.71 / 40.49; **baseline (OC-SORT) 9.90**.
- **Why it matters to SpeckLock:** this is a peer-reviewed international challenge that independently concluded IoU is unusable at this scale and replaced it with normalised centre distance. Cite it — with RGBT-Tiny's SAFit — whenever τ=12 px is called idiosyncratic.

### 4.5 RGBT-Tiny — **SAFit**, the principled metric upgrade

- `SAFit = σ(√A / C − 1) · IoU + (1 − σ(√A / C − 1)) · NWD(C)`, with **C = 32**. In words: **NWD for tiny boxes, IoU for large ones**, blended smoothly by box area.
- Usable as **both metric and loss**. SpeckLock already uses NWD in the fusion variant (`dronedet/nwd.py`), so adopting SAFit is close to a one-line change that makes the scoring citable to TPAMI 2025 rather than idiosyncratic.
- **Successor to know about:** **Gaussian Combined Distance** (GCD, IEEE GRSL 2025, https://github.com/MArKkwanGuan/mmdet-GCD) claims NWD is not scale-invariant and fixes it; mmdet config `retinanet_r50_aitodv2_gcd_1x.py` with `cocoapi-aitod==12.0.3`. A NWD→GCD swap is one training run and an honest measured delta.

### 4.6 Anti-UAV family (410 / 600 / 4th challenge)

- **Tracks 1 & 2 — "State Accuracy" (SA/AOA):** mean over frames of `IoU_t` **when the target is visible**, plus a `p_t` visibility-flag term that **penalises false alarms on frames where the target is absent**. This is unusually well-suited to SpeckLock's FP-suppression story: an empty-frame false alarm is directly punished.
- **Track 3 — MOTA:** `1 − (FP + FN + IDS) / GT`.
- **Reference bars:** 4th ed. Track 1 winner **AOA 73.23** (2nd 73.08, 3rd 71.45); Track 2 winner 66.76. Historic: CVPR 2023 Track 1 top-3 0.700 / 0.688 / 0.680, Track 2 0.611 / 0.591 / 0.570; ICCV 2021 0.6444 / 0.6388 / 0.6380; CVPR 2020 0.7381 / 0.7346 / 0.7338. **Note how low these are** — that is what a genuinely hard tiny-target problem looks like, and it is the honest counter to "amateurish".

### 4.7 LTS Multi-UAV (Codabench 16223)

- **Metric:** **HOTA / MOTA / IDF1** — i.e. **IoU-based association**. Submission = zip of MOT-format `.txt`. No external data permitted for badge eligibility.
- **The opening:** on 3–14 px targets, IoU-based association is exactly the brittleness SpeckLock's centre-distance argument targets. A strong SpeckLock row **plus a DotD/SO-HOTA re-scoring of the same leaderboard** would be a publishable critique, not just an entry.

### 4.8 DAUB / infrared small-target convention

- **DAUB (TRX-TCRNet, Sensors 2026):** a detection within **5 pixels** of ground truth is a TP. NMS keeps up to **20 peaks** with edge masking. Reports mAP50 / F1 / AUC. **This is a second independent adoption of centre-distance matching** — cite it.
- **NUDT-MIRSDT / IRDST / ITSDT-15K convention:** segmentation-style **IoU / nIoU / Pd (detection probability) / Fa (false-alarm rate, ×10⁻⁴)**. `Fa` units are per-pixel and **do not** translate to FP/frame — see §5.7.
- **IRSTD-UAV (TDCNet, AAAI 2026):** P / R / F1 / AP50, and **the data must be pre-aligned with GIM registration** before training or testing. TDCNet therefore depends on an external ego-motion compensation stage it never characterises — a gap SpeckLock is uniquely placed to fill with a stabilisation-error-vs-accuracy curve.

### 4.9 AI-TOD-v2 / tiny-object stills

- COCO AP with tiny-specific bands: **APvt (very tiny, 2–8 px)**, APt (8–16), APs (16–32), APm. Harness: `cocoapi-aitod`, mmdet configs at https://github.com/Chasel-Tsui/mmdet-aitod.
- Split: 11,214 train / 2,804 val / 14,018 test; mean object **12.7 px**.
- Reference: NWD-RKA + DetectoRS **24.7 AP / 57.2 AP50 / 9.7 APvt**; RFLA 25.7 / 58.9 / **9.2 APvt**; DQ-DETR 30.2 / 68.6 / 15.3; D³R-DETR 31.3 / 65.1 / 16.6; ScaleBridge-Det (3.0B params) 35.7 / 72.1 / **16.2 APvt**.
- **Read the APvt column, not the AP column.** Four years and a 3-billion-parameter model have moved very-tiny AP from 9.2 to ~16.6. That is the field conceding that scale does not solve sub-8-px.

### 4.10 SpeckLock's own protocol (for the record, and what must change)

- Current: centre-distance matching with radius `max(τ, 0.5·√area)`, **τ = 12 px** (`dronedet/evaluate.py`). Training labels inflated to a uniform **24 px** square (`tools/make_datasets_v3.py:50 LABEL = 24.0`).
- **Consequence, measured today:** the detector always predicts ~24 px boxes (measured width constant at 24.0), the 10_06 GT box width is a synthetic constant **8.0 px**, so det-vs-GT IoU has a hard ceiling of (8·8)/(24·24) = **0.111** and **COCO AP is structurally 0.000**. The one method that predicts variable-extent boxes (`moe3-stacked`) scores AP50 0.373.
- **Therefore: no SpeckLock detection number is currently comparable to any published number, on any benchmark, at any IoU threshold.** Fixing this is a precondition for §2 and §3, not an optional polish. The route is to keep centre-distance as the primary metric (now defensible via SO-HOTA, SAFit, DAUB, NWD) *and* emit real-extent boxes so an IoU/COCO/SAFit column can be reported alongside.

---

## 5. Honest warnings — which numbers are not comparable, and why

### 5.1 ⚠ **SpeckLock's own ARD-MAV number is on the wrong split, and the model was trained on the official test videos**

This is the most urgent finding in the briefing.

- `tools/make_dataset_external.py:41` **defines** the official split: `ARD_TEST_IDS = [phantom05, 08, 09, 10, 19, 30, 41, 43, 46, 47, 58, 63, 65, 70, 86]` with the comment *"ARD-MAV official split (Guo et al.): 15 test videos"*.
- But `combined_splits()` at line 209 — the function that actually built the round-5 combined training set — **ignores it** and re-splits by index:
  ```python
  def split(ids):
      tr, va, te = [], [], []
      for i, v in enumerate(sorted(ids)):
          (te if i % 10 == 0 else va if i % 10 in (1, 2) else tr).append(v)
      return {"train": tr, "val": va, "test": te}
  ```
  That yields **6 test videos out of 60**, chosen by position, not the official 15.
- Since the training pool is 42 index-chosen videos and the official test list is a fixed 15, **the great majority of the official ARD-MAV test videos are in SpeckLock's training set.** The reported **0.836** is therefore (a) on a home-made 6-video split, (b) under **centre-distance τ=12 px**, not MGMD's IoU 0.25, and (c) from a model that saw most of the official test data.
- **It cannot be placed in the same table as MGMD's AP 0.55.** Doing so is the exact charge that "amateurish" is reaching for. The fix is mechanical — re-acquire ARD-MAV, retrain on the official 45, score the official 15 at IoU 0.25 — but it must be done before any external claim.
- Compounding: the README headline "ARD-MAV AP 0.994 · NPS AP 0.801" is **one clip per dataset** (`phantom16`, `Clip_19`), not the split. The repo's own audit shows ~0.156 of the 0.84→0.994 jump is clip choice and ~0.002 is architecture.

### 5.2 Drone-vs-Bird numbers do not compare across papers

Test annotations are withheld, so **every** published DvB number is on a self-chosen validation split. Concretely:

- Laroca **0.7390 mean mAP50** = 7 named validation videos, model trained on DvB + USC-GRAD-STDdb + a 4,516-frame set + **10,000 DUT Anti-UAV images**.
- YOLOMG **AP 0.41** on "Drone-vs-Bird" = **zero-shot cross-dataset transfer**, no DvB training at all.
- 2020-edition numbers (Gradiant 80.0 %, Alexis 79.8 %, Eagledrone 66.8 %) are on a **different, older test set** with a **different metric**.

These three are frequently quoted side by side. They are three different experiments.

### 5.3 The "drone-vs-bird" benchmark does not measure drone-vs-bird

Birds appear in DDS but are **not annotated**. A bird false alarm is counted as an ordinary FP. **No bird-specific false-alarm rate has ever been published for this challenge in eight editions.** Any claim of "bird discrimination" measured on DDS is a claim about generic false positives. (This is also the opening described in §2, Priority 5.)

### 5.4 ~~Laroca's own citation trap~~ — **corrected, this claim was wrong**

The briefing originally asserted that arXiv v1 of 2504.19347 claims first place while v2
(2025-11-15) revises to an undisclosed top-3. **Checked directly against v2 on 2026-08-12: it still
says "The proposed approach attained first place in the 8th WOSDETC Drone-vs-Bird Detection Grand
Challenge, held at the 2025 International Joint Conference on Neural Networks (IJCNN)."** The
first-place claim stands in the current version; cite it as first place. Retained here rather than
deleted, because the rest of this section is about not repeating unchecked claims.

### 5.5 Cross-dataset AP is not a difficulty ranking — the benchmarks differ in kind

| Reported | Value | What it actually measures |
|---|---|---|
| DUT Anti-UAV (survey) | mAP50 **0.971** | Large, well-resolved RGB targets, static camera |
| Anti-UAV410 / IR multi-frame | mAP50 **93–98**, Pd 97–99 | 1–10 px but **high-contrast blobs against cold sky, static camera** |
| ARD100 (YOLOMG) | AP **0.85** | 1080p, air-to-air, moving camera, <12×12 px |
| **ARD-MAV (MGMD)** | **AP 0.55, recall 0.59** | **The real state of the art on RGB moving-camera few-pixel drones** |
| AI-TOD-v2 APvt | **9.2 → 16.6** over four years | Very-tiny static aerial stills |

A 0.83 on a hard set is not "worse" than a 0.97 on an easy one. **Always print the regime next to the number.**

### 5.6 Third-party re-implementations are not the original papers

The widely-quoted collapse **TransVisDrone 0.95 (NPS) → 0.15 (ARD100)** comes from **YOLOMG's** re-evaluation, not from TransVisDrone's authors. TransVisDrone's published training config is **42 GB VRAM at batch 4, 1280 px, 5-frame Video-Swin** — physically impossible on 8 GB, so any local re-run is at a reduced configuration and is **not** the published method. Dogfight is worse: TensorFlow 1.12 / CUDA 9.0 / driver 384 cannot run on Blackwell (sm_120 needs CUDA 12.8+), and there are no released weights. **Treat Dogfight as an annotation source and a citation, never as something you ran.**

Same hazard in the other direction: this repo's own README bolds TransVisDrone 0.15 as a "collapse" while its own row is **ARD-MAV**, a different dataset — flagged in the local audit as issue #6.

### 5.7 Metric families that look similar and are not

- **`Fa` (false-alarm rate, ×10⁻⁴)** in the IR segmentation literature is **per-pixel**, not FP/frame. EV-SpSegNet's `Fa 1.63e-4` and SpeckLock's `0.002 FP/frame` are not on the same axis.
- **AP@0.5 vs mAP@0.5:0.95 vs AP(centre, τ=12)** — three different numbers routinely printed in the same column across papers.
- **HOTA/MOTA/IDF1 (IoU-based) vs SO-HOTA (DotD-based)** — the LTS board and the SMOT4SB board are *both* "tracking leaderboards" and are not comparable.
- **State Accuracy (Anti-UAV)** includes a visibility-flag penalty term; it is not IoU and not AP.
- **AI-TOD vs AI-TOD-v2** — SFDNet's "31.7 AP on AI-TOD" and D³R-DETR's "31.3 AP on AI-TOD-v2" are **different datasets**. v2 is the relabelled, noise-corrected version.
- **Re-splits masquerading as public benchmarks:** MI-DETR reports on `IRDST-H` and `DAUB-R`, its own re-splits; its **+26.35 mAP50** over prior work is not an apples-to-apples gain.
- **AP50 vs AP75 on few-px objects:** D³R-DETR wins AP75 (26.2 vs DQ-DETR's 22.3) but *loses* AP50 (65.1 vs 68.6). AP75 on a 4 px object measures sub-pixel box regression — the least meaningful axis in the problem. Under centre-distance matching the ranking of this entire family would likely reshuffle.

### 5.8 Numbers in the sweep that are snippet-derived or unread — **do not quote without opening the source**

| Item | Problem |
|---|---|
| YOLOBirDrone accuracy/precision deltas | from search-result text, not a fetched table |
| Xie et al. "Improved YOLOv7" (Wiley IJAE) | page returned **HTTP 402**; numbers are snippet-derived |
| YOLOv12-ADBC (drone 96.4 % / **bird 80.0 %**) | mdpi.com returned **HTTP 403**; second-hand |
| DSNet, DENet, TogetherNet, SABDR | all snippet-derived aggregations |
| CHAL (CVPR 2026) | CVF PDF **403**; README publishes results only as an image |
| Loddis, MIST, S2MVP, DQAligner, ADSUNet | README results are **images**, no numeric table extracted |
| Nano-VID-weather (WACV 2025 W) per-subset table | CVF PDF **403**; only the aggregate "+0.21 mAP" is quotable |
| UFDT-YOLO | ScienceDirect paywalled |
| FR-DETR, HiProto, SPIRIT, BIRD, TEP-SAM, FeedbackSTS-Det, Na-IRSTD, SR-TOD, Deep-NFA, NSQR, LAF-YOLOv10, Anti-Aliasing (ACML 2023) | **no numbers extracted at all** |
| Drone-vs-Bird 8th-edition per-team scores | behind IEEE Xplore paywall (doc 11228314) |
| FL-Drones "P 0.89 / R 0.85 / F1 0.87 / AP 0.84" | **could not be attributed to any fetched paper** — sweep marked `verified:false` |
| RTTS arXiv ID | **inferred from the literature**, not fetched — sweep marked `verified:false` |
| USC-Drone | two contradictory frame counts in circulation, no live URL |

### 5.9 Papers that report only deltas, or only on private data

- **UAV-DETR:** "+6.61 % mAP50:95" on a **custom** dataset; the public DUT gain is only **+1.4 % P / +1.0 % F1**. No absolute public-leaderboard number → unplaceable.
- **SDD-YOLO:** 86.0 % mAP50 on **proprietary DroneSOD-30K**, +7.8 pp over a **YOLOv5n** baseline. No public benchmark, no code.
- **SpectraSentinel, EGD-YOLO (abstract), Zamani & Abedini (augmentation)** — claims without numbers or without named datasets.
- **Perception-to-Pursuit (P2P):** single author, no code, "**100 % drone classification accuracy**", "**597×** pursuit feasibility", target venue "ICCV 2027". The 100 % is almost certainly an artefact of Anti-UAV-RGBT containing **no bird distractors** — every mover *is* a drone. Cite for the **Intercept Success Rate** framing only; note that its ISR is computed **offline from predicted trajectories**, whereas SpeckLock's 33/33 oracle vs 28/33 real is **closed-loop through a renderer and an airframe**. Those are not the same evidence class, and the difference favours SpeckLock.

### 5.10 Dataset contamination hazards

- **AOD4** was assembled by scraping YouTube-8M, **Anti-UAV**, and a third-party Roboflow set. Any AOD4 ↔ Anti-UAV cross-dataset evaluation is contaminated. **De-duplicate before use.**
- **BirDrone** contains **3,067 frames lifted from the DvB challenge** — so BirDrone ↔ DvB cross-evaluation is contaminated.
- **Laroca's model** trained on DUT Anti-UAV + USC-GRAD-STDdb + a fourth set, so its DvB result is not "DvB-trained".
- **Det-Fly numbers in the literature range 0.82 → 0.995** depending on the split; SPAE-YOLOv8 claims 0.922, LRDDv3's zero-shot YOLOv11m gets 0.485. **Fix and publish a protocol before quoting any Det-Fly number.**

### 5.11 Sampling-rate traps that silently break the temporal stack

- **LRDDv3** is released as stills **sampled at 5 FPS**. A 12-frame lag = **2.4 s** of motion. The temporal stack **cannot run** on the public release. Same class of problem for **VIP Cup** (frame contiguity unstated — verify on Kaggle before committing) and **DroneSwarms** (confirm frames are consecutive video *before* mailing the application form).
- **Det-Fly, AI-TOD-v2, SODA, Anti2, AOD4, BirDrone, TIB-Net, UETT4K** are stills. No temporal method can run. Any "we beat X on this set" claim there is a single-frame claim.
- Conversely, **SynDroneVision** is the one large synthetic drone set **recorded sequentially** — it is the only synthetic option where the stack applies.

### 5.12 The 0.06 → 0.83 headline will be probed, and here is what a reviewer will find

- The closest published analogue is **Temporal-YOLOv8 (TNO, Sensors 2024)**: *"instead of using a single video frame as input, multiple frames are stacked from different time steps"* — the identical mechanism, published **two years earlier**, with a near-identical jump (**0.465 → 0.839 mAP**). It also ablates variants SpeckLock has not: 9-channel Color-T-YOLO **0.743** and 11-channel Manyframe-YOLO **0.781**, both **worse** than the plain 3-grayscale stack at 0.839. **This must be cited.** SpeckLock's defensible novelty is what TNO did *not* do: ego-motion stabilisation before stacking, the specific t−12/t−6/t lags, the 4th ego-motion channel, centre-distance scoring, and the track-classifier announce rule.
- A more typical published temporal delta is **XS-VID / YOLOFT: +2.8 AP overall, +3.4 AP on the extremely-small split**. Against that, a 0.77-point jump invites the question "how weak was your single-frame baseline?" — and the honest answer per the local audit is that the 0.06→0.83 figures are Ultralytics internal-val on the **07_05 val segment**, not an independent test set.
- **The moving-camera framing is not supported by the two flagship numbers.** Measured locally with the repo's own `Stabilizer('translation')`: `10_06` drifts **0.76 px in x, 1.07 px in y across the entire clip** (per-frame max 0.30 px); `07_05` drifts 3.3 px / 0.64 px over 571 frames. That is a **static rig to sub-pixel precision**. The moving-camera evidence is the ARD-MAV/NPS row — which is the row with the split problem in §5.1. Say this plainly before someone else does.
- **Sample size behind every headline detection number:** 2 videos, 1 drone each, 885 scored drone instances, 934 bird instances **all from one flock in one video**, 2 independent flights. A 30-frame block bootstrap gives ~12 independent looks on 10_06; the CI [1.000, 1.000] means "perfectly consistent within one flight" and says nothing about a second flight.
- **Test-set hygiene:** the weights are clean (no dataset builder reads `10_06.mp4` — all five builders checked), **but the pipeline was tuned against it** (`tools/make_datasets_v3.py:14` targets "the 10_06 foliage-crossing misses"; `dronedet/trackclass.py:5` says "Measured on 07_05 + 10_06" above six hand-set constants). The honest phrase is **development test set**, not "unseen".

### 5.13 Two independent 2025 papers found that naive early channel-fusion *loses*

**EGD-YOLO** reports 4-channel RGB+IR fusion at **mAP50-95 0.425 vs 0.50 for IR alone**, undiagnosed. **SpectraSentinel** deliberately avoids early fusion for the same reason. SpeckLock's 4-channel `[R,G,B,ego-motion]` variant should *not* suffer this — its 4th channel is derived from the **same sensor**, not a heterogeneous modality — but that is a hypothesis, and the right response is a measured two-head late-merge ablation, not an assertion. The literature is genuinely split, which makes a measured answer publishable.

### 5.14 Where the field's own evidence says scaling will not save anyone

**DEIMv2** (DINOv3 backbones, up to 50.3M params, 57.8 COCO AP) states outright that its gains *"primarily arise from improvements on medium and large objects, while performance on small objects remains largely unchanged"*. Independently corroborated: **DINOv3-7b-sat scores 9.2 APvt on AI-TOD-v2 — worse than 2022's RFLA**. Billion-scale self-supervised pretraining does not help sub-16-px. That is the single most defensible framing available: **the scaling lane is closed for few-pixel targets; the prior lane — motion, temporal, geometry — is open.**

---

## 6. Convergent evidence worth citing whenever "amateurish" comes up

Not a comparison table — a defence file. Each row is an independent group reaching one of SpeckLock's contested design choices.

| SpeckLock choice | Independent published corroboration |
|---|---|
| Stack temporal moments as network input channels | **Temporal-YOLOv8** (TNO, Sensors 2024) 0.465→0.839; **"A Simple Detector with Frame Dynamics is a Strong Tracker"** — literally `cat(x_t, x_t−x_{t−1}, x_t−x_{t−2})`, **1st place, 4th Anti-UAV Challenge Track 1, CVPR 2025 (AOA 73.23)** |
| Homography stabilise **then** difference | **Dual-Interval Motion Cues** (arXiv 2605.22605, May 2026, PKU): SIFT/ORB GMC + dual-lag differencing, 23.3→27.4 mAP50 on VisDrone-VID, 38.2 FPS on Jetson Orin Nano |
| **Multiple** lags, not two frames | Same paper's ablation: short-only+MGA **24.4**, long-only+MGA **22.4** (*worse than the 23.3 baseline*), both together **27.4**. Also YOLOMG: 3-frame k=2 **0.78** vs 2-frame k=1 **0.73** |
| Centre distance, not IoU, at few px | **SO-HOTA/DotD** (MVA 2025 challenge); **SAFit** (TPAMI 2025); **5-px centre matching** (DAUB, Sensors 2026); **NWD/NWD-RKA** (ISPRS 2022); **DotD-YOLOv9-C** |
| Track-level verification for FP suppression | **OBSS track-based confidence boosting** — 1st place DvB 2021; **Sequence Models for Drone vs Bird** (arXiv 2207.10409) — bird classification **+73 %**, F1 **+35 %** from track context |
| Motion proposes, appearance disposes, with a candidate pool | **ASUMOT** (2026) motion-consistency clustering; **EventRadar** SAGE bearing-indexed persistence memory; **MGMD** trajectory filter (age ≥3 + displacement stats) |
| Native-scale crop, not downscaled full frame | **Na-IRSTD** (2026) native-resolution features + token selection; **LiM-YOLO** Nyquist "feature dilution" argument; **LRDDv3**: 640×640 mAP50 **0.543** vs 1920×1920 **0.822** on identical data |
| P2 stride-4 head | **SDD-YOLO** (2026), **SPAE-YOLOv8** (2026), **LAF-YOLOv10** (2026), **LiM-YOLO** (P2–P4) |
| Bearing is exact, monocular range is not | **ODD-SEC** (2026) reports **angular error <2°** as its headline, not IoU or range |
| Cheap engineered temporal beats heavy learned temporal | **LVNet**: 1.77M params / 17.88 GFLOPs beats **LMAFormer** 390M / 380 GFLOPs by 18.4 % nIoU; **TRX+TCRNet**: mAP50 97.40 at **0.17 GFLOPs**; **TransVisDrone** 0.95→**0.15** moving to tinier targets |
| Super-resolution on crops buys nothing | **CoLR-Det** (2026): SR and detection want different things — use restoration as a training-only latent regulariser, never a preprocessing stage. Also **HazyDet**: best dehazer 24.2 vs 23.5 baseline; **"From Fog to Failure"**: dehazing *degrades* clear-image detection |

**The one thing SpeckLock has that none of them do:** the sweep contains **zero** papers that combine ego-motion stabilisation **and** a multi-lag continuous-intensity input stack **and** a track-level verifier **and** a closed-loop pursuit half. Dual-Interval has (1)+(2) but binarises the mask and has no tracker. YOLOMG has (2)+ a small-object head but no stabiliser and no tracker. Frame Dynamics has (2) only. Laroca has (3) only, as post-processing. **That intersection is the paper.**

---

## 7. Immediate next actions

1. **Today:** email WOSDETC for DDS access; email Drexel for LRDDv3 source clips. Zero cost, multi-day latency.
2. **Today:** fix §5.1 — either re-run ARD-MAV on the official 15-video split or strike every ARD-MAV number from the README until it is re-run. This is the difference between "rigorous" and "indefensible".
3. **This week:** download ARD-MAV (~60 GB) and Halmstad (CC0). Retrain on the official 45; score the official 15 at **IoU 0.25** *and* at centre distance τ=12 *and* at SAFit, printing all three columns.
4. **This week:** emit real-extent boxes so COCO AP stops being structurally 0.000 (§4.10). Nothing external is comparable until this lands.
5. **This month:** enter **Codabench 16223** (LTS Multi-UAV — 39 cloud-background sequences, open until 2030) and **Codabench 5101** (SMOT4SB post-comp). Two dated, third-party-scored public numbers, on an 8 GB laptop GPU, on data that matches the stated interests.
6. **Check manually this week:** the **ICPR 2026** competition list (Lyon, 2026-08-17→22) and the **ECCV 2026** workshop slate — both are running now and neither was enumerable from the sweep.
