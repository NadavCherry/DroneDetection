# Detecting Few-Pixel Drones in Camera Video: A Technical Survey of Best-Quality Methods

## TL;DR
- For maximum detection quality of 4–6 pixel drones, do **not** rely on a single off-the-shelf detector — the highest-performing approach is a **motion-guided + high-resolution-preserving pipeline**: exploit temporal motion (background/ego-motion compensation + frame differencing or motion fusion) to surface tiny targets, preserve native resolution via tiling (SAHI) or high input resolution + a P2/P4 detection head, train with a scale-robust label-assignment metric (NWD/RFLA instead of raw IoU), and confirm with a temporal tracker. Motion fusion alone lifted AP@0.5 from 0.53 → 0.85 on the hardest tiny-drone benchmark (ARD100).
- The biggest single levers, in order of evidence-backed impact: (1) **motion/temporal cues** (often +14–32 AP on few-pixel targets), (2) **input-resolution preservation** (raising 640→1280 gave a 25% mAP gain on VisDrone, far more than architectural tweaks at +6%), (3) **tiling (SAHI)** (+6.8% AP inference-only, up to +12.7–14.5% with slicing-aided fine-tuning), and (4) **tiny-object-aware loss/label assignment** (NWD raised AI-TOD AP from 11.1 → 16.1).
- Evaluate with **AI-TOD-style size-binned AP** (very-tiny 2–8 px, tiny 8–16 px) rather than mAP@0.5 alone, because IoU is unstable at few-pixel scales; validate motion pipelines by separately measuring recall on truly tiny targets and false-alarm rate against birds/clutter.

## Key Findings

1. **A few pixels means appearance is nearly gone — motion is your strongest signal.** Classic single-frame detectors trained on COCO collapse on 4–6 px targets because by the time the image is downsampled through the backbone, the object disappears (an 8×8 object at the P3 head of a 640-input YOLO becomes a 1×1 feature). The infrared-small-target (ISTD) and track-before-detect communities have specialized in few-pixel targets for decades; their temporal-accumulation methods transfer directly to RGB drone video.

2. **Resolution is destiny.** A systematic YOLOv8 study on VisDrone found that scaling input from 640→1280 yielded a 25% detection improvement, "substantially exceeding the gains obtained through architectural modifications, such as adding a P2 detection layer (+6%)." Tiling (SAHI) achieves the same resolution-preservation by slicing instead of downscaling.

3. **IoU is the wrong metric at this scale — for training and evaluation.** A 1-pixel shift on a 6-pixel box changes IoU drastically. Normalized Wasserstein Distance (NWD) and RFLA (Gaussian Receptive-Field Label Assignment) replace IoU in label assignment/NMS/loss and give large gains on tiny benchmarks.

4. **The current best tiny-drone results come from motion-fusion architectures**, e.g., YOLOMG, which fuses a pixel-level motion-difference map with RGB and beats the YOLOv5/YOLOv8 baseline by 32 absolute AP points on the hardest dataset.

## Details

### 1. Detection architectures for small/tiny objects

**Clarifying the input-resolution issue.** The "224×224" recollection is the classifier-backbone heritage; modern YOLO detectors default to **640×640**, with "P6"/large variants at **1280×1280**, and specialized work pushing to 1024–4K. This matters enormously: standard YOLO detection heads are P3 (stride 8, 80×80 for 640 input), P4 (stride 16, 40×40), P5 (stride 32, 20×20). A UAV under 32×32 px spans only ~4×4, 2×2, or 1×1 grid cells at these strides — so a 4–6 px drone has essentially no representation at P3 and none deeper.

**Small-object detection heads (P2).** Adding a **P2 head** at stride 4 (160×160 for a 640 input) extends coverage to objects as small as 4×4 px. For an 8-px target, the P2 feature response is 2×2 (preserving edge texture) whereas P3 collapses it to a single point. Many UAV detectors (SOD-YOLO, LEAF-YOLO, GAME-YOLO, LAF-YOLOv10, SDD-YOLO) add P2 and sometimes drop P5 to save compute. Empirically, P2 addition gives roughly +6% on VisDrone — useful but smaller than a resolution increase.

**YOLO family suitability for tiny objects.**
- **YOLOv5**: Still the workhorse base for the strongest tiny-drone papers (YOLOMG, Drone-vs-Bird winners) because of its mature ecosystem and easy P2 modification.
- **YOLOv8**: Best-documented for UAV; in the systematic VisDrone study YOLOv8l achieved the best accuracy (15.9% mAP50) while YOLOv8x degraded (7.32%) due to training instability under limited data — bigger is not always better with small datasets.
- **YOLOv9** (Programmable Gradient Information), **YOLOv10** (NMS-free), **YOLOv11**: incremental; numerous 2024–25 derivatives (MASF-YOLO, RFAG-YOLO, BGF-YOLOv10) add P2 heads, attention, and receptive-field modules, reporting +4–16% mAP50 over their baselines on VisDrone, but these are still general-small-object, not few-pixel.
- **YOLO-NAS / PP-YOLOE**: anchor-free, strong general detectors but no special few-pixel advantage out of the box.

**Anchor-based vs anchor-free for tiny targets.** RFLA shows that **both** the box prior (anchor) and the point prior (anchor-free) are sub-optimal for tiny objects because IoU/center-sampling produce many "outlier" tiny GT samples that get under-supervised. The fix is a receptive-field/Gaussian-based assignment rather than the anchor/anchor-free choice per se.

**One-stage vs two-stage.** Two-stage detectors (Faster R-CNN, Cascade R-CNN, DetectoRS) historically lead tiny-object benchmarks (AI-TOD) when combined with NWD/RFLA, because the RPN can be tuned for tiny proposals. One-stage YOLO wins on speed and, with P2 + tiling + motion, closes most of the gap. For pure best-quality offline analysis, a Cascade R-CNN / DetectoRS + NWD-RKA + RFLA is a very strong static-image baseline.

**Transformer detectors.** Vanilla DETR is poor on small objects (only highest-level features, slow convergence). **Deformable DETR** samples a few key points around each reference and substantially improves small-object detection. **RT-DETR** (Baidu, NMS-free, hybrid CNN-transformer encoder with AIFI + CCFF) is the real-time transformer baseline; many derivatives (Drone-DETR, DV-DETR, SO-DETR, PT-DETR, UAV-DETR) adapt it for UAV small objects with deformable attention, P2-level features, and IoU-aware query selection. **Co-DETR** (collaborative hybrid one-to-many + one-to-one assignment) is among the strongest high-accuracy detectors and a good choice when latency is irrelevant.

**Purpose-built tiny/drone architectures.** ISTD networks (below), YOLOMG (motion fusion), TransVisDrone (spatio-temporal transformer), SDD-YOLO (ground-to-air, P2 at 1024 input), and difference-map methods like SR-TOD ("Visible and Clear: Finding Tiny Objects in Difference Map") on the DroneSwarms dataset (avg object ~7.9 px).

### 2. Image tiling / patching (SAHI)

**How SAHI works.** Slicing Aided Hyper Inference (Akyon, Altinuc & Temizel, ICIP 2022, arXiv:2202.06934; open-source at github.com/obss/sahi) slices a large image into overlapping patches, runs the detector on each at full resolution (no downscaling), and merges. It works at **both inference and fine-tuning time**: slicing the training set effectively increases the number of training images and enlarges the apparent object size in each patch. Reported gains: SAHI "can increase object detection AP by 6.8%, 5.1% and 5.3% for FCOS, VFNet and TOOD detectors" (inference-only on VisDrone/xView), rising to a "cumulative increase of 12.7%, 13.4% and 14.5% AP" with slicing-aided fine-tuning.

**Slice size and overlap.** Ultralytics/SAHI defaults are 256×256 or 512×512 with **overlap_height_ratio / overlap_width_ratio ≈ 0.2**; common production configs use 640×640 slices at 0.2–0.25 overlap. A 25% overlap improved AP for small and medium objects with only a minor large-object decline. One instance-segmentation study found segmentation metrics peaked at ~6% overlap and plateaued by 9% while inference time rose sharply — so there are diminishing returns; for few-pixel drones, a moderately high overlap (0.2–0.3) is justified to ensure a drone sitting at a tile edge appears whole in at least one neighboring slice.

**Slice-size trade-off.** Smaller slices magnify tiny objects (good) but risk cutting large objects across boundaries (bad) and multiply compute. SAHI authors note smaller patches reduce large-object accuracy. An **optional full-image inference pass** can be merged in to catch large/close objects. For 4–6 px drones in a 1080p/4K stream, small slices (512–640) with high overlap maximize recall.

**Merging across tiles.** SAHI deduplicates with **NMS, Soft-NMS, Non-Maximum Merging (NMM), Greedy-NMM (default), or LSNMS**; the default GreedyNMM uses an STR-tree spatial index to avoid O(n²) comparisons and merges overlapping boxes with a weighted combination (match threshold ~0.5 IoU). Roboflow's `supervision` library exposes equivalent `with_nmm`.

**Compute cost.** Dense tiling means N inferences per frame plus merge overhead — the cost is linear in tile count, with "no way around" it. This is acceptable here since latency is explicitly deferred.

### 3. Super-resolution preprocessing

**Does SR help?** Evidence is **mixed but net-positive when SR is paired with detection, with real hallucination risk.** Positive results: SR preprocessing boosted mAP ~12% on 320×240 SoccerNet imagery (IoU 0.50:0.95); a StyleGAN+Swin remote-sensing hybrid hit 97.2% mAP@0.5 on VEDAI; an aerial SRGAN+YOLOv5 study reported +2.6% mAP from SR preprocessing plus +2.9% from architecture (+5.5% total) on VisDrone.

**The hallucination caveat.** SR is an ill-posed, generative process — it can "invent details that look plausible but are factually incorrect," and **inference-time SR cannot recover information absent from a 4–6 px input**; it can fabricate or smear a few-pixel blob. A training-time SR-guidance branch (used during training, discarded at inference) avoids inference hallucination but "cannot recover visual information absent in low-resolution test images."

**Best practice.** Prefer **joint/end-to-end SR+detection** or **detection-driven/task-driven SR** (Haris et al., "Task-driven super resolution"; JCS-Net joint classification+SR for small pedestrians; Joint-SRVDNet) where the SR objective is coupled to the detection loss, over naive Real-ESRGAN/SwinIR upscaling as a blind front-end. For a genuine 4–6 px drone, SR is a secondary enhancement — motion and tiling matter much more, and SR should be validated for false positives, not assumed beneficial.

### 4. Temporal / motion-based methods (most important for few-pixel targets)

**Why this dominates.** When a target is 4–6 px, inter-frame motion is often the only reliable discriminator from static background and sensor noise. This is the central insight of ISTD and track-before-detect.

**Background subtraction / modeling.** GMM-based background models + background subtraction work for static cameras. For moving cameras (PTZ, handheld, airborne), you must first do **camera-motion compensation**: estimate a global homography/affine transform (grid keypoints + optical flow, RANSAC), warp the background model/previous frame to the current frame, then subtract. Dense-optical-flow-based foreground/background segmentation handles illumination changes better but is costlier. Hybrid methods fuse classical background modeling with deep optical flow (FlowNet) for robustness.

**Frame differencing & optical flow.** Three-frame differencing after alignment is the core of YOLOMG's "motion difference map." Optical flow (sparse or dense) plus motion compensation lets you threshold compensated flow vectors (background clusters near zero; the drone is an outlier). Rozantsev et al. (CVPR 2015, "Flying objects detection from a single moving camera") showed regression-based motion compensation of spatio-temporal cubes gave ≥15% AP improvement over optical-flow alignment, and noted that off-the-shelf background subtraction and optical flow alone are insufficient in hard conditions.

**Camera motion compensation / stabilization / ego-motion.** Essential for any moving-camera deployment. Methods: 2D parametric (homography/affine) global motion compensation; spatial transformer network refinement; background orientation reconstruction for large-depth scenes. Limitation: fails under huge global motion (rapid pan/tilt) or large depth variation, and struggles with slowly-moving or temporarily-stationary objects.

**Multi-frame fusion / frame stacking.** Stacking N consecutive frames as input channels to a CNN (Thai et al.) improves accuracy at higher compute. Spatio-temporal ConvNets and 3D/2D fusion (STMENet for moving IR targets) aggregate temporal features.

**Track-before-detect (TBD).** Classic for dim, low-SNR targets: instead of thresholding each frame (detect-then-track), TBD makes "soft decisions," accumulating evidence across frames before declaring a detection. Methods: **Dynamic Programming Algorithm (DPA / Barniv 1985)**, Hough-transform TBD, particle-filter TBD, penalty-DP-TBD for multiple maneuvering targets, 3-D matched filtering (Reed-Gagliardi-Stotts). These are directly applicable to few-pixel drone trails and recover targets buried in noise that frame-wise thresholds miss.

**Infrared Small Target Detection (ISTD) — the most relevant specialized field.** ISTD targets are by definition a few pixels with low SNR. Key deep methods (segmentation-style, U-Net-derived):
- **ACM / ALCNet** (Dai et al.) — asymmetric contextual modulation + attentional local contrast, fusing model-driven local-contrast priors with CNNs.
- **DNANet** (Li et al., 2022) — densely nested attention network on UNet++, preserves tiny-target features through deep layers; a standard strong baseline.
- **ISNet** (Zhang et al.) — "shape matters," edge/contour-aware.
- **UIU-Net** — U-Net nested in U-Net for multi-scale fusion.
- **MSHNet** (scale & location sensitivity, 2024), **MTU-Net** (transformer), **EFLNet**, **SCTransNet**, **MSDA-Net** (direction-aware), **RPCANet** (deep-unfolded robust PCA).
Metrics in ISTD: **IoU, nIoU, Pd (probability of detection), Fa (false-alarm rate, ×10⁻⁶)**. These transfer to RGB: treat the drone as a low-contrast small target against sky/cloud/tree clutter. **Moving-ISTD** (multi-frame) substantially beats single-frame ISTD (see results below).

**Video object detection / feature aggregation.** MEGA, and temporal transformers (Video Swin) used in TransVisDrone, aggregate features across frames to stabilize weak per-frame detections.

### 5. Training methods & tricks for tiny objects

**Loss functions.**
- **Normalized Wasserstein Distance (NWD)** (Wang/Xu et al., 2021, arXiv:2110.13389): model each box as a 2-D Gaussian and measure similarity via Wasserstein distance. NWD is scale-invariant, smooth to location deviation, and works for non-overlapping boxes. Drop it into assignment, NMS, and loss. On AI-TOD, replacing IoU with NWD in label assignment raised AP from 11.1 → 16.1, AP50 from 26.5 → 43.8, and tiny-bin AP_t from 7.8 → 17.4 (and gave Faster R-CNN +6.7 AP overall). Best practice: keep IoU loss for larger objects, add NWD for tiny (as in YOLO-FaceV2).
- **NWD-RKA** (Xu et al., ISPRS J. Photogramm. 2022): NWD + Ranking-based assignment; on DetectoRS it boosts performance by 4.3 AP over contemporary competitors on AI-TOD-v2.
- **Wise-IoU, Shape-IoU, Focaler-IoU, Focal loss**: handle imbalance and hard samples; used in recent UAV-YOLO variants.

**Label assignment.** **RFLA** (ECCV 2022, arXiv:2208.08738): Gaussian receptive-field prior + Receptive Field Distance (RFD) + Hierarchical Label Assignment; +4.0 AP on AI-TOD over SOTA. Pair RFLA + NWD for tiny targets.

**Data augmentation.**
- **Copy-paste / oversampling of small objects** (Kisantal et al., 2019, arXiv:1902.07296): oversample images containing small objects and copy-paste small instances multiple times — directly addresses the "too few small-object samples" problem; "Simple Copy-Paste" (Ghiasi, CVPR 2021) gives large gains with large-scale jittering. For drones, paste real drone crops onto diverse sky/cloud/treeline backgrounds, with scale/brightness jitter and realistic placement.
- **Mosaic + scale augmentation + multi-scale training** standard in YOLO.
- Caveat: random copy-paste can raise false positives by pasting over background and reducing negative diversity; context-restricted paste mitigates this.

**Anchor design.** Recompute anchors via k-means on your dataset (drones cluster to tiny, near-square boxes); custom dynamic anchors notably boosted small-drone detection in the Drone-vs-Bird challenge (Gradiant).

**Input resolution / multi-scale.** Train at the highest resolution you can afford (1280+), use multi-scale training, and/or train on SAHI slices.

### 6. Datasets

**Drone/UAV detection:**
- **Anti-UAV** (CVPR 2020 / ICCV 2021 / CVPR 2023 challenges): 410 RGB+IR video sequences; Anti-UAV410 (thermal, 410 seqs, tiny-target heavy). CST Anti-UAV (220 thermal seqs, 78k tiny instances) for tiny UAV tracking in complex scenes.
- **Drone-vs-Bird** (WOSDETC, 6 editions through ICASSP 2023): video, static and moving cameras, 720×576 to 4K, 8 drone types, with birds as distractors — the canonical drone-vs-bird discrimination benchmark.
- **NPS-Drones, FL-Drones** (drone-to-drone), **AOT** (Amazon Airborne Object Tracking), **USC Drone, Det-Fly, MIDGARD**, DUT Anti-UAV, TIB (extremely small targets), **ARD100** (YOLOMG; 100 videos, 202,467 frames, 1920×1080 @30 fps, DJI Mavic2/M300, objects usually <12×12 px, ~0.01% of frame), **DroneSwarms** (avg ~7.9 px, 99.6% tiny).
- **UETT4K Anti-UAV** (2025): 4K real-world drone footage.

**Small/tiny-object benchmarks:**
- **AI-TOD / AI-TOD-v2**: 28,036 aerial images, ~700k–752k instances, **mean object size ~12.8 px** (12.7 px for v2), ~86% of objects <16 px — the definitive tiny benchmark. It defines the size bins very-tiny (2–8 px), tiny (8–16 px), small (16–32 px), medium (32–64 px) for AP. AI-TOD-v2 is a careful relabeling that fixes missing-annotation/location-error problems.
- **TinyPerson** (sub-20 px persons), **SODA-D/SODA-A** (driving/aerial, avg ~20 px), **VisDrone** (10k images, 540k boxes, dense small objects), **DOTA** (aerial oriented), **AI-TOD-R** (oriented tiny, mean 10.6²).

**Synthetic data.** Rendering 3-D drone models over 2-D/3-D backgrounds with domain randomization (position, rotation, lighting, post-effects) works well: the OBSS Drone-vs-Bird entry fine-tuned YOLOv5 on real + synthetic data with a Kalman tracker and found an optimal real/synthetic mix improved performance. Pure-synthetic training has been demonstrated (Faster R-CNN). Copy-paste is effectively cheap synthetic augmentation.

**Annotation for few-pixel objects.** Point/centroid annotation plus a fixed small box, or single-point supervision (used in ISTD), is more practical than precise boxes at 4–6 px; relabeling efforts (AI-TOD-v2) show that annotation noise dominates at this scale, so careful/relabeled GT and tolerance to localization error in metrics matter.

### 7. Complete end-to-end pipeline designs

**Pipeline A — Motion-first cascade (recommended for a mostly-static or stabilizable camera; best recall on few-pixel targets).**
1. **Stabilize / ego-motion compensate** (homography via grid keypoints + optical flow + RANSAC).
2. **Motion candidate generation**: aligned 3-frame differencing and/or background subtraction → morphological filtering → connected components → candidate ROIs. (Optionally a TBD/DP accumulator to pull dim targets out of noise.)
3. **Tile + detect**: run SAHI-style slices (or just crop ROIs) at full resolution through a P2-headed detector trained with NWD/RFLA, to confirm/classify each candidate.
4. **Classifier head** drone-vs-bird/clutter on each ROI (appearance + short motion-track features).
5. **Temporal track confirmation**: Kalman/SORT-style tracker; require persistence over K frames and consistent motion to raise an alarm, suppressing single-frame false positives. (Drone-vs-Bird winners used exactly this track-boosting.)

**Pipeline B — Motion-fusion end-to-end (current SOTA for tiny drones; YOLOMG-style).** Compute a pixel-level motion-difference map (after RANSAC alignment), fuse it with the RGB frame via a bimodal attention/fusion module, and feed the fused tensor to an enhanced YOLOv5/v8 with a small-object (P2) head. End-to-end, no separate proposal stage. This produced the largest measured gain on the hardest tiny benchmark (below).

**Pipeline C — Spatio-temporal deep network (TransVisDrone / IRST-DETR style).** CSPDarkNet (spatial) + Video-Swin (spatio-temporal) over a short clip → detection head; or a temporal-transformer DETR. End-to-end, learns motion implicitly, edge-deployable.

**Pipeline D — ISTD-style segmentation + temporal (best when target is a low-contrast blob against sky).** Treat as moving-infrared-small-target detection: DNANet/MSHNet-style segmentation backbone + multi-frame bidirectional temporal propagation → centroid extraction → tracker. Strong when targets are dim and nearly textureless.

**Pipeline E — High-accuracy offline (no latency limit).** Co-DETR or DetectoRS + NWD-RKA + RFLA, run on SAHI slices at high overlap, plus a full-image pass, merged with GreedyNMM, then temporal association across frames. Add training-time SR guidance. This is the "throw everything at quality" configuration.

**Hybrid classical+deep rationale.** Classical motion methods are cheap, unsupervised, and excellent at flagging *where* something moved; deep detectors are excellent at *what* it is. The cascade uses motion for high-recall region proposal (so the detector only sees small crops at full resolution) and the deep net + tracker for precision — this directly attacks the few-pixel problem while controlling false alarms from birds, clouds, and insects.

### 8. Evaluation

**Metrics.** Use **AI-TOD size-binned AP**: AP_vt (2–8 px), AP_t (8–16 px), AP_s (16–32 px), plus AP, AP50, AP75. mAP@0.5 alone is misleading at few-pixel scale because a 1-px localization error tanks IoU; report AP at lower IoU and the NWD-based matching, and track **Pd / Fa** (ISTD convention) for the operational alarm view. For video, report per-track detection rate and false-alarm rate per minute.

**Validating motion+detector pipelines.** (a) Ablate motion vs no-motion on the *same* detector (YOLOMG ablation: RGB+RGB no-motion AP 0.33 vs motion-fused 0.78). (b) Measure recall specifically on the very-tiny bin. (c) Stress-test camera-motion compensation on moving-camera sequences and against bird-heavy clips (false-alarm rate). (d) Cross-dataset generalization (e.g., train AI-TOD → test VisDrone) to detect overfitting.

### State-of-the-art quantitative results (2023–2025)

**YOLOMG (arXiv 2503.07115, Guo/Lin/Zhao, 2025), AP@0.5:**
- **ARD100 (avg object ~0.01% of frame, <12×12 px): YOLOMG-1280 = 0.85 vs YOLOv5/YOLOv8 baseline 0.53 (+0.32 absolute / 32 points)**; YOLOMG-640 = 0.78. Best non-YOLOMG method (YOLOv9) = 0.64, so +0.21 absolute over the best prior method.
- **NPS-Drones: YOLOMG-1280 = 0.95**, tying the previous best (TransVisDrone 0.95).
- Ablation: removing the motion map (RGB+RGB) drops ARD100 AP to 0.33; removing the small-object layer costs ~2 points. Speed: 35 FPS at 1280, 133 FPS at 640.
- Note: there is a minor internal inconsistency in the paper (intro says "22%", experiments say "21% absolute"); the 32-point gap is specifically vs the YOLOv5/v8 baselines.

**TransVisDrone (arXiv 2210.08423, ICRA 2023), AP@0.5IOU:** NPS-Drones **0.95**, FL-Drones **0.75**, AOT **0.80**. (On the much harder ARD100, YOLOMG's authors report TransVisDrone collapses to 0.15, illustrating ARD100's extreme difficulty — but that figure is from a competing paper, not the TransVisDrone authors.)

**Moving-ISTD, BIRD (arXiv 2508.15415, 2025), mAP50:** On IRDST, BIRD (multi-frame) = **92.60** vs best single-frame method (MSHNet) 78.50 — a **+14.1-point** temporal gain; vs best prior multi-frame (SSTNet) 83.25, +9.35. On DAUB, BIRD = **97.85** vs DNANet 89.24. Ablation isolating temporal modules: IRDST baseline (no temporal) 75.10 → full bidirectional 92.60 (**+17.5 points** from temporal modeling alone). BIRD: 9.39M params, 55 FPS. This is the cleanest evidence that multi-frame/temporal modeling dominates single-frame for few-pixel targets.

**Tiny-object label-assignment gains:** NWD on AI-TOD: AP 11.1→16.1 (assignment); RFLA: +4.0 AP on AI-TOD; NWD-RKA: +4.3 AP on AI-TOD-v2. **SAHI:** +6.8% (FCOS) inference-only, +12.7–14.5% with slicing-aided fine-tuning. **Resolution (VisDrone):** 640→1280 ≈ +25%, vs P2 ≈ +6%.

## Recommendations

**Stage 0 — Characterize your data first.** Measure the actual pixel-size distribution of drones in your footage and whether the camera is static, PTZ, or moving. This single fact dictates everything: a static camera makes background subtraction trivially powerful; a moving camera mandates ego-motion compensation. Annotate a validation set with AI-TOD-style size bins.

**Stage 1 — Build the motion backbone (highest ROI).** Implement ego-motion compensation (homography + RANSAC) → aligned 3-frame differencing / background subtraction → morphological cleanup → candidate ROIs. Validate that you can surface 4–6 px moving blobs at high recall, accepting high false positives at this stage. If the camera is static, this alone may give near-complete recall.

**Stage 2 — Preserve resolution in the detector.** Train a P2-headed YOLOv8/v11 (or YOLOv5 for ecosystem parity with YOLOMG) at 1280+ input, **and** run SAHI inference (512–640 slices, 0.2–0.3 overlap, GreedyNMM merge, plus one full-image pass). Recompute anchors via k-means. Resolution preservation is worth more than any single architecture tweak (640→1280 ≈ +25% vs P2 ≈ +6% on VisDrone).

**Stage 3 — Fix the loss/assignment for tiny boxes.** Add NWD (keep IoU for larger objects) and adopt RFLA-style assignment. This is a cheap, high-impact change (NWD raised AI-TOD AP 11.1→16.1).

**Stage 4 — Go motion-fusion or spatio-temporal for best quality.** Reproduce/adapt **YOLOMG** (motion-difference-map + RGB bimodal fusion) — it is the strongest published tiny-drone result (0.85 vs 0.53 baseline AP on ARD100) and directly fits the "few pixels, video available" scenario. As an alternative/ensemble, add a **TransVisDrone**-style spatio-temporal clip model, and if targets are dim sky blobs, an **ISTD multi-frame** branch (DNANet/MSHNet + bidirectional temporal propagation, cf. BIRD's +17.5-point temporal gain).

**Stage 5 — Confirm temporally and discriminate birds.** Add a Kalman/SORT tracker; require K-frame persistence and motion consistency before alarming. Train an explicit drone-vs-bird classifier on track-level features (birds flap/have different kinematics).

**Stage 6 — Data.** Pretrain/benchmark on Anti-UAV, Drone-vs-Bird, NPS-Drones, ARD100, and AI-TOD; augment with copy-paste of real drone crops onto your specific backgrounds plus rendered-3D synthetic drones with domain randomization. Use single-point/small-fixed-box annotation for the tiniest targets.

**Stage 7 — Treat SR as an optional, validated add-on.** Try task-driven/joint SR (not blind Real-ESRGAN). Keep it only if it improves AP_vt without inflating false positives — at 4–6 px it can hallucinate.

**Thresholds that change the plan:** If static camera and AP_vt recall from motion alone is high, you can skip the heavy spatio-temporal nets. If false-alarm rate against birds is the bottleneck (not recall), invest in the track-level classifier and temporal persistence rather than the detector. If 640→1280 gives <5% gain, your targets are large enough that standard P2-YOLO suffices and you can drop tiling.

## Caveats
- **Hallucination risk**: inference-time SR can fabricate detail on few-pixel inputs; validate empirically, prefer joint/training-time SR.
- **Metric instability**: mAP@0.5 is unreliable at 4–6 px; some reported gains in the literature are sensitive to annotation noise (hence AI-TOD-v2 relabeling). Cross-paper number comparisons are hazardous — datasets, IoU conventions, and "AP" definitions differ (AP@0.5 vs COCO AP).
- **Single-source numbers**: the TransVisDrone 0.15 on ARD100 and several baseline numbers come from competing authors' reimplementations, not original papers; treat cautiously.
- **Moving-camera failure modes**: global motion compensation breaks under rapid pan/tilt, large depth variation, and for slow/stationary targets — budget for these explicitly.
- **Compute**: dense tiling, spatio-temporal nets, and multi-branch ensembles are expensive; this report optimizes for quality per the brief, deferring latency. The strongest configurations (Co-DETR + SAHI + temporal) are offline-grade.
- **ISTD transfer is not free**: ISTD methods assume IR-like low-texture targets; on RGB with cluttered backgrounds (foliage, urban), false alarms rise and retraining/domain adaptation is required.
- Several cited 2025/2026 arXiv items are recent preprints; treat their headline numbers as not-yet-peer-reviewed.