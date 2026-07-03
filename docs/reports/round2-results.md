# Round 2 — Re-evaluation on Manual Ground Truth + Multi-Frame Methods

Round 1 (`docs/reports/round1-pipeline.md`) built the pipeline and compared methods against a semi-automatic GT.
The user then hand-labeled the full video (`work/gt_labeled_all.json`: class `drone` = the far/moving target, plus 8 `bird` tracks), which **corrected the ground truth itself** — everything here is scored against `work/gt_user.json` derived from those labels.

## 1. What the manual labels changed

- **Frames 0–290**: the auto-GT and the manual labels agree at **1.2 px median** — mutual validation of the descent + low-skim segment.
- **Frames 336–570**: they disagree by **330 px median**. The auto-GT's "return cruise across the sky" was a **bird** (an unlabeled tenth flying object); the real drone **stays low over the bushes, drifting slowly left** ((765,512)@274 → (655,477)@411 → (514,471)@547) at ~0.5 px/frame, nearly invisible in any single frame. The user could only label it by flipping consecutive frames — that observation drives every improvement below.
- The manual labels also fill the auto-GT's 45-frame gap (291–335) and add 8 bird tracks (a flock crossing the treeline leftward, frames 2–304).
- The near (landed) drone is unlabeled by choice; it is an **ignore region** in scoring (finding it is fine, missing it costs nothing).
- Frames 548–570 are unlabeled → excluded. Val split stays 342+ (train < 342), and under the corrected GT the val drone is the **hardest** target in the video: a 4–10 px low-contrast smudge against field/bush texture.

## 2. Old methods against the real target (val = frames 342–547)

Re-scoring every round-1 detector against the corrected GT collapsed almost everything:

| method (round 1) | AP(far) | R(far) | FP/frame | note |
|---|---|---|---|---|
| moe-hybrid (round-1 winner) | 0.004 | 0.06 | 1.5 | was detecting the bird cruise |
| motion-median | 0.005 | 0.05 | 0.6 | slow drifter absorbed into its own background model + flicker-suppressed |
| **motion-mog2** | 0.020 | **0.864** | 38.9 | per-pixel variance modeling still sees it — recall exists, precision hopeless |
| yolo-* (pretrained), yolo-ft, ft5-sahi | ≈ 0 | ≈ 0 | – | single-frame appearance has almost nothing to work with |

Two lessons: (a) the *proposal* stage must handle **slow movers** — my median detector's 150-frame window put the drone inside its own background; (b) single-frame appearance is near-blind here, so the verification stage needs **time**.

## 3. Round-2 components

**`motion-slow`** — the median/MAD detector with a **lagged background** (samples only from frames ≥90 old, window 240): a 0.5 px/frame drifter has moved 45+ px out of its own background model, while static scenery is unaffected. Val: R(far) 0.049 → **0.573** at 3.5 FP/frame (vs MOG2's 0.864 @ 38.9). A parameter sweep confirmed looser thresholds only flood the ranking — the fix is the lag, not the threshold.

**`ft6`** — 2-class (drone/bird) single-frame tiny specialist trained on the user's labels (real instances + copy-paste of both classes, near drone erased, 24 px labels). Internal val on *real* instances peaked at **mAP50 ≈ 0.06** — single-frame appearance confirmed near-blind on the bush-phase target.

**`ft7`** — the same 2-class specialist but **multi-frame**: input channels are *stabilized grays at t-12 / t-6 / t*, so static scenery cancels to gray while anything moving leaves a color-fringed trail — the user's manual labeling technique, formalized (and the frame-stacking idea from the research surveys). Copy-paste simulates temporal signatures: drone patches pasted rigidly with velocity 0–2 px/frame (including hover), bird patches faster with per-channel size jitter (wing-flap). On the same real-instance val: **mAP50 0.827, P 0.939, R 0.777** (single-frame ft6: 0.06) — still improving at epoch 60, so a longer run should gain more.

**`moe2-hybrid` / `moe3-stacked`** — the full pipelines: proposals = motion-slow ∪ MOG2 (union recall with precision left to verification), verified by ft6 (single-frame) or ft7 (temporal) respectively, bird-classified candidates suppressed to low scores, unioned with the ft1 near/big-expert full-frame pass.

**`sr-hybrid`** — ablation: the round-1 hybrid with learned super-resolution (FSRCNN ×4) on verification crops instead of bicubic.

**`tracked-moe3`** — track-before-detect flavored temporal integration: the tracker's output (coast-filled, persistence-confirmed) converted back into per-frame detections and scored as a detection method.

## 4. Results (val = frames 342–547, the hardest segment)

| method | AP(far) | best-F1 | R(far) | P | FP/frame | fps |
|---|---|---|---|---|---|---|
| **tracked-moe3** (temporal verifier + tracker integration) | **0.960** | **0.938** | **1.000** | 0.884 | 0.13 | 4.0 |
| **moe3-stacked** (temporal verifier, per-frame) | 0.656 | 0.760 | 0.767 | 0.752 | 0.25 | 4.0 |
| moe2-hybrid (single-frame verifier, same proposals) | 0.222 | 0.316 | 0.558 | 0.220 | 1.98 | 3.5 |
| motion-slow (proposals alone) | 0.090 | 0.226 | 0.573 | 0.141 | 3.50 | 19.2 |
| motion-mog2 | 0.020 | 0.042 | 0.864 | 0.022 | 38.9 | 25.8 |
| sr-hybrid (FSRCNN ×4 crops) | 0.014 | 0.059 | 0.049 | 0.075 | 0.60 | 9.7 |
| hybrid (bicubic ×4 crops) | 0.005 | 0.055 | 0.049 | 0.063 | 0.72 | 10.3 |
| best round-1 method (yolo-ft-hybrid) | 0.023 | 0.060 | 0.049 | 0.079 | 0.57 | 10.7 |
| all pretrained yolo variants, yolo-ft, ft5-sahi | ≈ 0.000 | ≤ 0.001 | ≤ 0.005 | – | – | – |

moe3 vs moe2 is a clean ablation — identical proposals, identical training data and recipe, the only change is 3 stacked stabilized-gray frames instead of 1 RGB frame: **AP ×3, F1 0.316 → 0.760**. Adding the tracker's temporal integration (persistence + coasting converted back to detections) reaches **R 1.000 at F1 0.938** — every labeled frame of the hardest segment is covered.

Full video (all 548 labeled frames): tracked-moe3 **AP 0.906, F1 0.881, P 0.991, 0.01 FP/frame**; moe3-stacked 0.754/0.831; moe2 0.559/0.687. Full table: `work/eval_user_full.md`.

## 5. Tracking on the corrected target

On `moe3-stacked` detections, the tracker covers **97.1% of all 548 labeled far-drone frames as ONE track — zero ID switches, 1.0 px median error, longest unbroken streak 302 frames** — through descent, bush-skim, the fast dash, and the slow drift. In drone-confirmed mode (score ≥ 0.55): same far coverage, 4 tracks total, of which 2 are birds that occasionally fool the verifier (at 4 px a bird and a drone genuinely converge; the track-level flap/kinematics classifier remains the next step). The single-frame moe2 confirmed tracker reaches 86.5% coverage, also as a single ID.

| tracks (vs user GT) | far coverage | IDs (switches) | streak | med err | false tracks |
|---|---|---|---|---|---|
| moe3-stacked, all-objects | **0.971** | 1 (0) | 302 | 1.00 px | 7 (birds/unconfirmed movers) |
| moe3-stacked, drone-confirmed | 0.971 | 1 (0) | 302 | 1.02 px | 2 |
| moe2-hybrid, drone-confirmed | 0.865 | 1 (0) | 302 | 1.09 px | 0 |

## 6. Super-resolution verdict

Learned SR (FSRCNN ×4) vs bicubic ×4 on identical verification crops: AP 0.014 vs 0.005 on val, 0.155 vs 0.151 full-video — a marginal nudge, both noise-level where it matters. **SR is not the lever for few-pixel targets; temporal information is.** (Consistent with the research surveys' hallucination/marginal-gain warnings; task-coupled SR could still be tried in the heavy fine-tune, but with low priority.)

## 7. Takeaways for the heavy fine-tune

1. **Temporal input is not an optimization — it is the difference between blind and seeing** for sub-10 px low-contrast targets: single-frame mAP50 0.06 vs multi-frame 0.83 on identical data and architecture. Prioritize motion-difference/stacked-frame input channels (YOLOMG-style) in the production model.
2. **Slow movers need a lagged/long-memory background model** — classic background subtraction with a short window silently absorbs them. Keep both a fast channel (MOG2-like) and a lagged channel.
3. Ground truth for few-pixel targets is itself a modeling problem: the semi-automatic GT was pixel-accurate where verification was possible and confidently wrong where it wasn't (it followed a bird for 234 frames). Label with motion (flip frames), verify with contact sheets, and treat annotation as iterative.
4. The 2-class (drone/bird) formulation lets the verifier suppress the main false-alarm source instead of delegating everything to track-level logic.
5. Copy-paste remains the workhorse; for temporal models, paste *trajectories* (per-channel offsets + flap jitter), not just pixels.

## Artifacts (round 2 — nothing from round 1 was overwritten)

- `work/gt_user.json` — canonical GT from the manual labels; `tools/build_gt_user.py` regenerates + prints the agreement analysis
- `work/eval_user_val.md`, `work/eval_user_full.md` — comparison tables
- `work/det2/`, `work/tracks2/`, `work/vis2/` — round-2 detections, tracks, videos (videos painted with the user's labels in green)
- `work/models/yolo-ft6-best.pt` (2-class single-frame), `work/models/yolo-ft7-best.pt` (2-class temporal), `work/models/FSRCNN_x4.pb`
- `tools/make_dataset_ft6.py`, `tools/make_dataset_ft7.py`, `tools/final_round2.py`, `tools/tracks_to_dets.py`
