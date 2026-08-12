# SpeckLock — adversarial audit

Reviewer posture: CVPR Reviewer 2 + staff engineer doing pre-award due diligence for a defence
programme. Every claim below was verified against the tree at commit `998ed2d` (single commit,
`main`). Findings that failed an independent refutation pass have been dropped; a short list of what
was *checked and cleared* appears at the end, because the owner needs to know which attacks fail.

---

## 1. Verdict

**"Amateurish" is wrong about the engineering and right about the presentation layer, and the gap
between the two is itself the most damaging finding.** The round reports are a genuinely disciplined
research record: negative results are recorded with their measurements, the centre-distance matching
rule is correctly motivated rather than asserted, `pursuit/` carries 540 well-built tests with a
deliberately torch-free CI contract, and the honest version of almost every problem below is already
written down *somewhere in this repository by the author*. That is not amateur work. What is amateur
is the top 50 lines of `README.md` and the whole of `docs/index.html`'s hero, where those honest
numbers are systematically stripped of their qualifiers on the way up: the flagship "24/24
intercepted, 0 buildings hit" is a run in which the simulator handed the guidance law its own
bounding box (`work/pursuit/city/results.json` → `"detector": "oracle"`), while the same mission with
the shipped seeker is 0/3 with all three buildings destroyed (`work/pursuit/city_pipe/results.json`);
the public gallery prints "detection rate 0.97" under ten videos that contained no detector; "AP/F1 =
1.000" is one video, one drone, one track, three distinct confidence values, with no *n* anywhere
near it; "ARD-MAV AP 0.994" is a single clip drawn from a six-clip split the repo itself defined and
scored at 0.836; and the entire detection half — the code producing five of the nine substantive
headline rows, including `dronedet/evaluate.py`, the scorer behind every published AP — has zero
committed tests while "540 unit tests" sits four rows below in the same table. A reviewer needs about
twenty minutes and two `results.json` files to take the front page apart, and once they have, they
stop trusting the 2,000 lines below it that are honest. The precise sense in which the criticism is
correct is therefore not *incompetent* but **unblinded and unpropagated**: nothing in the evaluation
protocol is blind, every rounding and omission on the way to the headline runs in the flattering
direction, and there is no mechanism — no test, no CI check, no propagation step — that would catch
it.

---

## 2. Findings, ranked

Severity: **CRITICAL** = would end the review / would fail a technical due-diligence gate.
**MAJOR** = would not survive scrutiny as written. **MINOR** = real defect, low blast radius.

| # | Sev | Dimension | Finding | Evidence | Fix | Effort |
|---|-----|-----------|---------|----------|-----|--------|
| 1 | CRITICAL | Claim integrity / Presentation | The flagship "24/24 intercepted, 0 buildings hit" is a perfect-sensor run. The same mission with the shipped seeker is **0/3, all three buildings struck**. The sensor is named on none of the six surfaces a reader meets first. | `work/pursuit/city/results.json` → `detector:"oracle"`, all degradations 0, `latency_frames:0`; `pursuit/perception.py:191` reads `gt["bbox"]` straight from the sim; `work/pursuit/city/METRICS.md:39` "there is no detector in the loop". Uncaveated at `README.md:42`, `docs/index.html:246`, `:7` (meta description), `:9` (og:description), `tools/make_social_card.py:33` → the og:image PNG, `tools/make_arch_figure_system.py:254` → the SVG linked as "the method in one diagram", whose lane 4 is headed "HOW EVERY NUMBER HERE WAS MEASURED". Counter-run: `work/pursuit/city_pipe/results.json` → `detector:"yolo"`, n=3, hits=0, all three `outcome:"target_struck"`, `detect_rate` 0.022–0.044. Config blocks are identical apart from `calibrate_frames`, so it is the same mission. Disclosed only at `README.md:314`, line 314 of 347. | Put the sensor in the claim wherever the number appears; add the 0/3 row beside it, as `pursuit/README.md:14-17` already does. Re-caption the hero GIF ("yellow is the tracker's estimate, fed from the simulator's own bounding box"). Regenerate both images. | hours |
| 2 | CRITICAL | Presentation | The public gallery advertises a **"detection rate"** for ten clips in which no detector ran, on the same page where that label denotes a real detector's hit rate. | `docs/gallery.html:96-105` — ten city cards ending "· detection rate 0.97 ·". `grep -c oracle docs/gallery.html` = **0**. `docs/media/showcase.json` sources them from `work/pursuit/city_clips/`, whose `results.json` is `detector:"oracle"`, `weights:null`, `max_bearing_err_deg:0.0`, `motion_ms:0.0`. `tools/make_gallery.py:179` prints the field verbatim. The number is not even perception-shaped: `pursuit/episode.py:693` divides by `n_visible` (`:602`, geometrically in-frame), so the shortfall is the renderer's annotator giving up past ~120 m (`episode.py:596-601`). Same page, `docs/gallery.html:110-115`, uses the identical label for chase clips that *did* run a detector (`work/pursuit/final/*/results.json`, `detector:"fusion"`). | Emit `detector` from `results.json` into `showcase.json`; label the city cards "sensor: simulator ground truth · in-frame rate 0.88" and the chase cards "detection rate 0.92 (fusion detector)". | hours |
| 3 | CRITICAL | Testing & CI / Reproducibility | The detection half has **zero committed tests**, including `dronedet/evaluate.py`, the scorer behind every published AP. | Committed `pytest.ini` → `testpaths = pursuit/tests`; `git ls-files '*test_*.py'` → 7 files, all `pursuit/tests/`. Untested tracked lines: `dronedet/` 2,440 + `realtime/` 1,585 + `final/` 151 + `tools/` 6,962 = **11,138**; two `assert` statements exist in the whole half (`dronedet/mc_data.py:37`, `tools/fuse_dets.py:71`), neither a test. No pursuit test imports `dronedet`. `.github/workflows/tests.yml:4` concedes it in a comment. Five of the nine substantive rows of `README.md:37-46` come from this code, two of them through the untested τ=12 px matcher at `dronedet/evaluate.py:37` (`tools/eval_improvements.py:47` → `tools/eval_external.py:23`). `DT = 6` — the temporal-stack spacing behind the 0.06→0.83 thesis — is duplicated across `tools/make_dataset_ft7.py:31`, `dronedet/methods/hybrid2.py:32`, `realtime/tools/make_datasets_rt.py:30`, `realtime/pipelines.py:59`, bound only by a prose comment. | Commit `dronedet/tests/` (it exists untracked — see #12), add unit tests for `_match_frame`/`_ap` on hand-built PR curves, and a golden-file test (#4). | days |
| 4 | CRITICAL | Claim integrity / Reproducibility | "ARD-MAV AP 0.994 · NPS AP 0.801" is **one clip per dataset**, presented as a dataset-level result, with no committed evaluation artifact. | `README.md:40` reads "public data, moving cameras" with no link. Source is `docs/reports/round7-fusion.md:50-51`, a three-row per-*clip* table; `tools/eval_improvements.py:32-35` hardcodes `phantom16`, `Clip_19`, `10_06`. The repo's own protocol (`round5-moving-camera-multidataset.md:20-21`, `tools/make_dataset_external.py:276-291`) defines the held-out split as **6** ARD-MAV / **5** NPS videos, scored 0.836/0.811 at `round5:53`. The cherry-pick is provable from the report's own baseline column: the round-5 checkpoint scores **0.992 on phantom16 vs 0.836 pooled**, so ~0.156 of the 0.84→0.994 jump is clip choice and ~0.002 is the new architecture. NPS is not inflated (single clip 0.801 < pooled 0.811). `README.md:123` quotes 0.84/0.81 for the same claim 83 lines later with no reconciliation. `work/improve` and `work/ext_datasets` do not exist and are untracked, so no artifact backs either number. | Headline the full-split numbers, or label the cells "one held-out clip (phantom16 / Clip_19)". Commit `work/improve/RESULTS.json` and the `gt_test` JSONs. Reconcile :40 with :123. | days |
| 5 | MAJOR | Evaluation validity | **AP = 1.000 is a degenerate PR curve**: 1 video, 1 drone, 1 track, 337 GT frames, 3 distinct scores. | `data/videos/10_06.mp4` decodes 361 frames; `realtime/work/gt_1006_v2.json` has exactly one non-ignore object (`far`, 337 frames, 23..360) plus 24 excluded frames. `work/det3/1006/tracked-pcmax.json` = 338 detections, **all labelled `track9`**, scores `{0.994:272, 0.7952:17, 0.5467:49}` because `tools/tracks_to_dets.py:75` multiplies one per-track constant by `STATUS_FACTOR` (`:21`). `dronedet/evaluate.py:95-112` is a correct continuous AP; with FP=0 it collapses to precision at recall 1. One missed frame moves AP by 1/337; one surviving 100-frame false track takes P to ~0.77. `README.md:37` carries no *n*; the Results table (`:33-46`) has no sample size anywhere. The repo concedes scene-specificity at `round3-deliverables.md:217` and `docs/guides/methods.md:120`, and prints per-frame 0.846/0.910 at `methods.md:93` — none of which reaches the headline. | State "1 video / 1 drone / 337 labelled frames" wherever the number appears. Report the tracked result as coverage + false-track count; reserve AP for the per-frame detector. | hours (label) / weeks (more flights) |
| 6 | MAJOR | Claim integrity | The contribution-level superlative — "specialist-class on every set with one set of weights, which no published method does" — rests on a cross-dataset, cross-metric table whose key number the repo's own survey says not to trust. | `README.md:41`, `:127-132`, table at `:116-123`. (a) The "collapse" cells are **ARD100** (Dogfight 0.50, TransVisDrone **0.15**) while our row is **ARD-MAV** 0.84 — different datasets (`few-pixel-drone-methods-survey.md:107` vs `round4-external-datasets.md:18`; `round4:156` calls ARD100 "the deferred, even-tinier set"). (b) The generalist was never trained or scored on ARD100 (`round5:18-22`), so "every set" excludes the one where the collapse was measured. (c) The bolded 0.15 is exactly what `few-pixel-drone-methods-survey.md:180` warns is "competing authors' reimplementations, not original papers; treat cautiously" — reproduced in bold, hyperlinked to a paper that does not contain it, under a disclaimer (`README:125`) that mis-describes its provenance. (d) `arXiv 2504.11967`, the citation for "lists no unified multi-dataset model", appears nowhere in `docs/references/`. (e) The "and does it in real time" clause borrows 74 fps from EDGE-RT (07_05-only, `final/README.md:5`) and 107–122 fps from the nano Edge v1 (`round6:121`), which scores 0.795/0.770, not the 0.836/0.811 quoted. No single configuration demonstrates both halves of the sentence. Same table is live at `docs/index.html:435-470`. | Delete the superlative or restate as a hypothesis. Split ARD100/ARD-MAV columns, footnote the reimplementation caveat, quote fps only for the checkpoint the accuracy row belongs to. | hours |
| 7 | MAJOR | Presentation | "EDGE-RT · 74 fps" is a discrete **laptop** GPU number; the only edge estimate that exists is 5–7× lower and appears on no top-level surface. | `README.md:39`, `:86`, `:130`; `docs/index.html:467` quotes 74 fps with **no hardware at all**. `realtime/README.md:76-77` projects RT-C @1280 at **10–15 fps** and RT-D @640 at 25–35 fps; `final/README.md:58` repeats it. `grep -n 'Jetson\|Orin' README.md docs/index.html` → one hit, inside a directory listing (`README.md:291`). The projection is explicitly *not a measurement* (`realtime/README.md:73` "stated assumptions, not measurements… to be validated on device") — i.e. EDGE-RT has never run on an edge device. Hardware is also elided one-directionally: every deep doc says "RTX 5070 **Laptop**" (`final/README.md:35,51`; `realtime/README.md:9,20,33,66`; `methods.md:93`), every top-level surface drops "Laptop" (`README.md:39,130`; `docs/index.html:383`; `docs/gallery.html:121`). `final/README.md:41` heads the section "EDGE-RT — real-time, **edge hardware**" over a laptop-dGPU-only measurement. | "74 fps on an RTX 5070 Laptop (TensorRT FP16); projected 10–15 fps on Jetson Orin Nano @1280 — not yet measured on target." | hours |
| 8 | MAJOR | Presentation | The same measured claim is published with two different values, and the **live site quotes a figure the repo has documented as fabricated**. | Latency compensation, six locations, two answers: `README.md:319-320`, `pursuit/README.md:702-703`, `:927-928`, `:975` all say **32/42 vs 18/42**; `pursuit/guidance.py:794-795` and `docs/index.html:756` say **36/42 vs 17/42**. Same 150 ms, same `full` suite. I re-ran `python -m pursuit.sandbox --suite full --latency-frames 3 [--compensate-latency]` → 32/42 and 18/42, so the README is right and the site and the docstring are wrong — while `pursuit/README.md:703` asserts "deterministic, so both numbers reproduce exactly". Worse: `pursuit/README.md:382-383` records that "53.9 ms, 18.5 FPS" "was a hardcoded literal in `pursuit/tools/city_report.py` with no run behind it"; the literal is gone from the tool and the corrected 231.5 ms / 4.4 FPS is printed at `docs/index.html:655` — yet `docs/index.html:749-750` still asserts 53.9 ms, and uses it to conclude "the detector is 100% of the reason" and "a pure inference problem", when 208.0 of the 231.5 ms is the CPU motion stage (`README.md:317-318` already says so). `docs/.nojekyll` confirms `docs/` is the live Pages source. `tools/check_docs.py` validates only link targets, so nothing could catch either drift. | Pick the reproducible run, propagate it to all six sites, delete 53.9 ms from the site. Extend `check_docs.py` to flag headline figures with conflicting values across documents. | hours |
| 9 | MAJOR | Reproducibility | The **shipped** models' headline number has no committed artifact and no written command that produces it. | `README.md:83-86` credits AP 1.000 to PC-MAX/EDGE-RT, i.e. `final/`. `git ls-files final/` = README + `run_final.py` + 6 weights; `final/out/` gitignored (`.gitignore:71`). `final/run_final.py` contains **no eval call and no GT path** (`grep -n 'eval\|gt_'` → empty). The shipped-generation figures (0.846/0.876, `round3-deliverables.md:190-191`) and run names `final-ftS`/`final-ftC` exist as prose only — `git grep` finds no JSON behind them. The weights are provably a different generation from every committed detection set: `md5 final/pc_max/fullS.pt` `efdfbe65` vs `work/models/yolo-ftSv3-best.pt` `978ad48c`; `final/edge_rt/edge_n1280.pt` `4d638502` vs `realtime/work/models/v3_full_temporal_n1280.pt` `060481e6`. The only committed 10_06 table is the split-trained generation (per-frame 0.910). | Commit `out_pc/tracked_dets.json` for 10_06 and one line in `final/README.md`: `python -m dronedet eval --gt realtime/work/gt_1006_v2.json --dets out_pc/tracked_dets.json`. | hours |
| 10 | MAJOR | Presentation | The one-line pitch names an operating condition — **moving camera** — that neither flagship detection number was measured under, and nothing says so. | `README.md:5`, `docs/index.html:209-211`, `:7`, and the `CITATION.cff` abstract all promise "from a moving camera". `README.md:37` (AP 1.000) is 10_06; `README.md:38` (0.06→0.83) is the 07_05 val split (`round2-results.md:31,33,75`). The repo calls that rig near-static six times (`round5:4,22,67,93`; `round6:36`; `round7:75`). I measured it with the repo's own `Stabilizer('translation')`: 10_06 drifts **0.76 px in x, 1.07 px in y across the entire clip** (per-frame max 0.30 px); 07_05, 3.3 px / 0.64 px over 571 frames (per-frame p95 0.14 px). That is static to sub-pixel — none of the parallax the phrase implies. `grep -c 'near-static'` = 0 in both `README.md` and `docs/index.html`, and neither Limits section mentions camera regime. Mitigation: `README.md:40-41` *does* carry labelled moving-camera rows. | One clause: say 07_05/10_06 are a fixed rig, and that the moving-camera evidence is the ARD-MAV/NPS row. Fix `CITATION.cff`. | hours |
| 11 | MAJOR | Testing & CI | The headline AP/F1 = 1.000 is reproducible **in 0.079 s from files already in git**, needs only numpy, and no test pins it. | I ran `evaluate_files('realtime/work/gt_1006_v2.json', ['work/det3/1006/tracked-pcmax.json'], tau=12.0)` → `1.000 1.000 1.000 1.000 0.000 1.000 0.00 0.9 4.4`, byte-identical to `work/eval_round3_1006test.md:5`. Re-ran with an import blocker on torch, ultralytics **and** cv2 — identical, because `dronedet/evaluate.py` needs only numpy and `detections.py`/`gt.py` are stdlib-only. It would run in the existing CI job (`tests.yml:29-30` already installs numpy) with zero new dependencies. 189 golden artifacts are tracked under `work/` and `realtime/work/`; there is not one golden-file test. Caveat: such a test pins the evaluator and the artifacts, not the detector or weights. | `dronedet/tests/test_golden_eval.py`: assert AP/F1/FP-per-frame to 3 dp for every (gt, dets, expected) triple already published in `work/eval_round3_*.md` and `realtime/work/eval3_*.md`. | hours |
| 12 | MINOR | Testing & CI | The `testpaths` pin is **already** silently swallowing 24 existing detection tests, and there is no packaging for CI to validate. | Committed `pytest.ini:10` = `testpaths = pursuit/tests`; `tests.yml:34` runs bare `pytest -q`; `README.md:252` documents the same. `dronedet/tests/test_metrics.py` exists untracked with 24 passing tests; `pytest` reports 540, `pytest dronedet/tests/test_metrics.py` reports 24 — dropped with no error and a green run. The working tree has already *edited* `pytest.ini` to `testpaths = pursuit/tests dronedet/tests` with a comment claiming "dronedet/tests holds the detection-half tests", while `git status` shows `?? dronedet/tests/` — so CI on GitHub is currently pointed at a directory that does not exist in the repository. The scope limitation is documented only at `CLAUDE.md:29`, which is gitignored (`.gitignore:74`). No `pyproject.toml`/`setup.py`/`setup.cfg` anywhere; CI never runs `pip install -e .`, so the import graph is validated only by pytest's basedir walk-up. | Commit `dronedet/tests/` and `dronedet/metrics.py`. Add a CI assertion on collected test count. Add a minimal `pyproject.toml` and `pip install -e .` in CI. | hours |
| 13 | MINOR | Testing & CI | Measured coverage is **41.3 %** of the tested half and **17.8 %** of production code; two thirds of `pursuit/` modules are at exactly 0.0 %. | Independent `sys.monitoring` statement coverage over the 540 tests (coverage.py is not installed): `pursuit/` non-test 2,311/5,601. At 0.0 %: `viz.py` 0/300, `run_pursuit.py` 0/194, and **all 18** statement-bearing modules under `pursuit/tools/` — 20 of 30 files. Covered core is healthy (dynamics 100 %, geometry 99.0 %, evader 94.8 %, guidance 91.1 %, perception 89.3 %, city 84.6 %, ring 67.3 %, sandbox 56.6 %, episode 48.9 %). Repo-wide 16,992 tracked statements, 2,418 covered. Sharpest instance: `wilson()` is defined at `pursuit/tools/analyze.py:49` and **duplicated verbatim** at `pursuit/tools/city_report.py:34`, both at 0.0 %, both returning (87.1, 76.6, 93.3) for 54/62 — the "CI [76.6, 93.3]" at `README.md:44`. Arithmetic is correct; a fix to one copy diverges silently from the other. These are pure two-integer functions — the cheapest possible tests. | Add `pytest-cov` to CI, print (don't gate) `--cov=pursuit --cov=dronedet`, then set and raise a floor. Test `wilson()` first and de-duplicate it. | hours |
| 14 | MINOR | Presentation | `README.md:126-127` states an absolute the repo violates by design, and "identical recipe" is not literally true. | `README.md:126-127`: "IoU swings wildly on a 6 px box, **which is why this repo never scores with it**" — but `dronedet/metrics.py:169-181` exposes `rule="iou"` documented as "comparable to papers", `metrics.py:248-262` ships `coco_ap()` (AP@[.5:.95]/AP50/AP75), and `round4-external-datasets.md:117-120` publishes a recomputed IoU@0.5 line. The headline at `:38`/`:76` (mAP50) is itself IoU-0.5. The *number* is fine — labels are inflated to a uniform 24 px (`make_dataset_ft6.py:31`, `make_dataset_ft7.py:31`), where IoU@0.5 ≈ 8 px centre tolerance, the same order as τ=12. Separately, "identical recipe" is false: `make_dataset_ft7.py:210-227` gives every pasted drone a per-channel velocity offset and every bird wing-flap jitter; `make_dataset_ft6.py:109-138` pastes statically. Everything else matches (same patch bank, scale/flip/brightness/blur/haze ranges, 3+3 pastes, 24 px labels, `SPLIT_AT = 342`), and neither val split contains pastes — so the conclusion is not confounded, only the word is wrong. `README.md:38` also cites no source and does not say the figures are Ultralytics internal-val on the 07_05 val segment. | Rewrite to "detection *scoring* is centre-distance; IoU appears only for like-for-like comparison with published numbers". Replace "identical recipe" with "same architecture, same real data, same hyper-parameters; synthetic positives differ only because a single frame cannot carry a motion signature". Add "(07_05 val split, round 2)" + link. | hours |

---

## 3. The eight fixes that move the needle, in order

Ordering rule: **kill-shot risk first** (things that end a review on sight), then **cheapest
credibility per hour**, then **structural** (things that stop the defect class from recurring).
Nothing in the top eight requires retraining a model.

**1. Put the sensor on the 24/24, everywhere, and publish the 0/3 beside it.** (Finding 1, hours.)
First because it is the only finding that a hostile reader will characterise as *misrepresentation
rather than sloppiness*. The claim is on the README results table, the site hero, the meta
description, the og:description, the generated social-card PNG, and the architecture SVG whose own
lane is titled "HOW EVERY NUMBER HERE WAS MEASURED". A defence reader parses "0 buildings hit" as a
survivability claim; measured survivability with the shipped seeker is 0/3. The repair is one clause
and two regenerated images, and `pursuit/README.md:14-17` already contains the correctly-worded
version — this is pure propagation.

**2. Fix the gallery's "detection rate".** (Finding 2, hours.) Second because it is the one place
the presentation *asserts* something the artifacts contradict rather than omitting context, and
because the gallery is the artifact most likely to be opened standalone and screenshotted. It is
also aggravated by being overloaded: the same label on the same page means a real detector's hit
rate for the chase clips. Two lines in `tools/make_gallery.py` plus a `detector` field in
`showcase.json`.

**3. Delete or downgrade the "no published method does both" superlative and rebuild the SOTA
table.** (Finding 6, hours.) Third because it is the only contribution-level claim in the repo and
the one with the weakest support — ARD100 numbers in the "collapse" column against an ARD-MAV number
in ours, a bolded figure the repo's own survey flags as an unreliable reimplementation, a citation
for the claim of absence that appears in no reference file, and speed evidence borrowed from two
models that are not the generalist. It is written as a superlative two clauses after a disclaimer
admitting the comparison is unsound. This is the single sentence Reviewer 2 will quote.

**4. Label the generalist numbers honestly and commit their artifact.** (Finding 4, days.) Fourth
because it is cherry-picking *inside the repo's own protocol* and therefore not defensible as an
oversight: the repo defines a 6-video ARD-MAV test split, the round-5 checkpoint scores 0.836 over
it and 0.992 on the single clip that got headlined. Reporting the full-split 0.836/0.811 costs the
headline 0.15 AP and buys back the credibility of every other number.

**5. Put *n* on every detection number, and stop calling the tracked result AP.** (Finding 5,
hours.) Fifth because it is the cheapest possible repair to the most-quoted figure. "AP/F1 = 1.000"
over three operating points on one track carries no ranking information; stated as "337 of 337 GT
frames covered, 0 false tracks, one 12-second clip", it is both *true* and still impressive, and it
inoculates against the obvious attack.

**6. Commit `dronedet/tests/`, fix collection, and add the golden-eval test.** (Findings 3, 11, 12,
hours→days.) Sixth because it converts the worst structural finding into a solved one for a few
hours of work: the test directory already exists untracked, the golden inputs are already in git,
the evaluation runs in 0.079 s with numpy alone inside the CI job that already exists, and CI is
*currently pointed at a directory that is not in the repository*. Add an assertion on collected test
count so the next silently-empty suite fails the build.

**7. Reconcile the conflicting numbers and delete the retracted 53.9 ms from the live site.**
(Finding 8, hours.) Seventh because a document that says "deterministic, so both numbers reproduce
exactly" while publishing two values has broken the only guarantee the reader was given — and
because the retracted figure is not merely stale, it *inverts the engineering conclusion* on the
page it sits on ("a pure inference problem" when 208 of 231.5 ms is the CPU motion stage). Then
extend `tools/check_docs.py` — which already enforces that every documented link resolves to a
tracked file — to flag headline figures with conflicting values across documents. That is the
structural fix for the entire defect class in this audit.

**8. Restore the hardware qualifier, add the Jetson projection, and commit a shipped-weights eval.**
(Findings 7, 9, hours.) Eighth because it is the finding most likely to matter to the actual
audience: anyone sizing this for a real platform takes 74 fps into a feasibility estimate and is
wrong by 5–7×, and anyone who runs the shipped `final/` models has no reference output to compare
against and no command to produce one. Both honest numbers already exist in `realtime/README.md` and
`final/README.md`; they simply never reach the front page.

Deliberately *not* in the top eight: coverage percentage (#13) and the IoU wording (#14) — real, but
nobody rejects a project over them; and the moving-camera pitch (#10), which is a one-clause fix but
partially mitigated by the labelled ARD-MAV/NPS row already in the same table.

---

## 4. What is genuinely good and must survive the rewrite

Be careful here. Several of the fixes above are "delete the claim", and it would be easy to gut the
parts that are the actual contribution. These must be protected:

- **The round reports (`docs/reports/round1..7`).** These are the best thing in the repository and
  the reason "amateurish" is wrong. `round3-deliverables.md:124-132` volunteers that a 1.000 depends
  on look-ahead and publishes the causal number instead. `round5:58-64` footnotes that the same
  checkpoint scores 0.149 or 0.811 depending on the split and explains why that is not a
  contradiction. `round7:69-84` documents where the new model *loses* to the old one. That is more
  self-critical than most accepted papers. Do not compress these into the README; fix the README to
  match them.

- **Recorded negative results, with their measurements.** Super-resolution on crops, INT8 on this
  GPU, DT=9 stacks, full-frame edge stabilisation, four failed discrimination filters
  (`docs/index.html:757-760`), and — the best one — "the proposal→verify architecture that wins on
  PC *loses* on edge". Most repos delete these. They are the single strongest evidence that the work
  is real.

- **The centre-distance matching rule and its justification** (`dronedet/evaluate.py:1-8`, τ = 12 px,
  `max(τ, 0.5·√area)`, 24 px inflated training labels). This is the correct call at 4 px, it is
  *argued* rather than asserted, and it is non-obvious. Keep it; just stop claiming the repo "never"
  uses IoU when `metrics.py` deliberately provides `coco_ap()` for paper comparison — that provision
  is itself good practice.

- **The `pursuit/` test suite.** 540 tests that are seeded (`test_dynamics.py:161,237,258,384,582`),
  that hard-code numbers specifically so a plausible refactor fails loudly (`test_geometry.py:105-113`
  pins an off-centre principal point so swapping `cx` for `width/2` breaks), that encode real field
  failures rather than API shapes (`test_ring.py:312-330`, "a persistent but motionless contact never
  steers"), and that assert the premise of their own scenarios (`test_ingress.py:74-78`). Coverage of
  the core is genuinely high — dynamics 100 %, geometry 99 %, guidance 91 %. This is better than most
  production code.

- **The torch-free CI contract** (`.github/workflows/tests.yml`, lazy imports in
  `pursuit/perception.py:243,770`). It is a deliberate, documented architectural constraint that
  makes the suite fast and portable, and it is what makes the golden-eval test in fix #6 free.

- **The oracle-vs-real ablation architecture itself.** `TargetEstimate` as the single boundary
  between `perception.py` and `guidance.py`, and `OracleDetector` as a deliberately degradable
  sensor, is exactly the right design — it lets accuracy be *attributed* (33/33 oracle vs 28/33 real)
  rather than argued. The finding above is not that the oracle run should not exist; it is that it
  must be labelled. Protect the ablation, fix the label.

- **The pursuit statistics.** Wilson intervals, Holm correction across six factors, Mann-Whitney on
  the perception-vs-guidance split (`work/pursuit/final/ANALYSIS.md §1-4`). This proves the author
  can do uncertainty quantification — which is why its total absence from the detection half reads as
  a choice. Extend it; do not remove it.

- **Provenance discipline that already exists**: the 🟢/🟡 real-vs-sim chips, the explicit Limits
  sections, `README.md:17-19` volunteering "There is no flight test here", `NOTICE.md`'s honesty
  about dataset redistribution, `docs/references/README.md` volunteering that its own sources "carry
  no byline, venue or date, and were never peer-reviewed", and `tools/check_docs.py` enforcing that
  every documented link resolves to a *tracked* file. Most repos never bother with the last one. It
  is also the natural place to hang the numeric-consistency check.

- **189 tracked golden artifacts** under `work/` and `realtime/work/` — the GT, the per-frame
  detections, the tracks, the eval tables. This is better artifact hygiene than most CVPR papers and
  it is what made this audit possible in an hour. Do not "clean up" `work/`.

- **The substantive technical findings** underneath all of it: the 3-moment temporal stack itself
  (0.06 → 0.83 is real, whatever metric it is quoted in); the lagged-background insight for sub-pixel
  drifters (R 0.049 → 0.573); the affine-transform persistence fix that took moving-camera coverage
  0.72 → 0.999; the NWD implementation; the `_CandidatePool` bearing-rate proof (3.8 % → 80.7 % time
  on the drone); and the ring's "motion proposes, appearance disposes" architecture. These are
  genuine engineering contributions and none of the findings above touch them.

---

## Appendix — attacks that were checked and *failed*

The owner should know these, because they are the obvious next accusations and they do not hold.
Each was tested against measurement, not argument:

1. **"The test GT is generated from the system's own output — circular."** Partly true of GT **v1**,
   and the headline uses **v2**. `harden_gt_1006.py:85-117` re-derives each position from raw pixels
   (non-causal windowed-median background, MAD-normalised SNR peak, uniqueness gate, sub-pixel
   centroid); 337/337 scored frames are `refined=True`. I re-scored with the two ignore objects
   deleted *and* all 361 frames included: AP 0.999, P 0.997. The circularity mechanism is worth
   ≤0.001. The repo also documents the defect and its repair at `round3-deliverables.md:60-80`.

2. **"The track classifier's thresholds were fitted on the test video."** Disabling the classifier
   entirely gives AP 0.999 / F1 0.996 instead of 1.000 / 1.000. Sweeping `N_CONF` 1..39 leaves the
   kept set unchanged over [4,16] for PC-MAX and is *completely inert* for EDGE-RT. It is worth
   0.001, not the headline.

3. **"Every P/R/F1 is read off a threshold swept on the test labels."** True of the intermediate
   `pc-max` diagnostic row; false for every headline, whose `best_thresh` is the minimum score in the
   file (the sweep degenerates to accept-all). Recomputing with no threshold gives identical figures.

4. **"Birds are `ignore`, so zero-FP cannot include them."** The 07_05 val window (frames 342–570)
   contains zero bird frames; the 10_06 test GT contains no bird objects at all, so a bird detection
   there scores as a full FP — and `round6:69-71` publishes exactly such a case (2 bird false tracks).

5. **"385 MB of binaries with no LFS, permanent in history."** History is a single squashed commit;
   a clone transfers ~426 MiB, and the largest checkpoints are force-added past `.gitignore`, not
   kept alive by negations.

6. **"No related work anywhere."** `docs/references/few-pixel-drone-methods-survey.md` is ~4,000
   words of structured prior-art prose with quantitative attribution, plus a second survey and a
   genuine comparison section at `round4:110-165`.

7. **"The moving-camera pipeline is unimplemented."** `mc_hybrid` implements grid-LK + RANSAC
   homography registration and is measured at 0.999 coverage on airborne ARD-MAV.

The pattern is instructive: **the technical attacks fail and the presentational ones land.** That is
the shape of the whole verdict.
