# Launch package — LinkedIn

Everything here is drafted to be defensible under a technical reader who clicks through.
Nothing in this file claims something the repository cannot show, and §6 lists the
sentences that must **not** be written, with what to say instead.

The through-line: **a representation change, measured under control, on a problem where a
single frame genuinely cannot work.** Not a leaderboard win — there isn't one, and a
reader who checks will find that out in about ninety seconds.

---

## 1 · Headline options

Ordered by how well each survives a sceptical click-through.

| # | headline | why it works | risk |
|---|---|---|---|
| **1** | **A drone 4 pixels wide is invisible in one frame. Stack three moments as colour channels and it isn't.** | States the idea and the problem in one line. Concrete, falsifiable, and the figure proves it. | none — this is the safest and the strongest |
| **2** | **I stacked three moments of video as R, G and B. Detection AP went from 0.159 to 0.895 — same network, same data, same resolution.** | Leads with the controlled number. "Same network, same data" pre-empts the first objection. | invites "on what dataset?" — answer is ready |
| 3 | **Six months on tiny-drone detection. The most useful thing I built was the thing that told me I was wrong.** | Differentiates on rigour rather than on score. Memorable. | softer hook; better as the *second* line than the first |
| 4 | **How do you detect something 3 pixels across from a moving camera? You stop looking at frames.** | Question hook, good reach. | slightly clickbait-adjacent; keep the answer immediate |
| 5 | **My detector loses to the state of the art. I published the comparison anyway — and what I found explaining the gap was more interesting.** | Highest-integrity framing; strong with senior researchers. | lowest reach; some readers stop at "loses" |

**Recommendation: #1 as the headline, with #2's number as the second line.** #1 states the
mechanism, #2 immediately backs it with a controlled measurement. #3 and #5 are the right
*closing* notes, not openers.

---

## 2 · The post

> **A drone 4 pixels wide is invisible in a single frame. Stack three moments as colour
> channels and it isn't.**
>
> That's the whole idea behind SpeckLock, and here's what it's worth: **AP 0.159 → 0.895.**
> Same network, same training data, same 1280 px input, same pipeline. The only thing that
> changed is whether the three input channels carry one frame's colour or three moments in
> time — t−12, t−6, and now, as R, G and B.
>
> Stabilise the video first and the static world cancels to grey. Anything that moved
> leaves a coloured trail. A detector that had nothing to look at suddenly does.
>
> **Why this problem is hard.** At 3–14 pixels there is no appearance to speak of. No
> texture, no shape, often no contrast. Every published leader on these benchmarks is a
> specialist — one dataset, one set of weights, scored at home — and off home turf they
> collapse. The camera is also moving, so "just subtract the background" doesn't work
> either: you have to compensate for the camera before the target's motion means anything.
>
> **What I built.** A YOLOv8s with a stride-4 head reading that three-moment stack, a
> Kalman tracker in stabilised coordinates, and a track-level classifier that decides
> whether a track is a drone rather than deciding frame by frame. On an RTX 4090 the edge
> configuration runs at **58.9 fps with AP 0.876** — accuracy and speed measured in the
> same pass, so the two numbers describe one execution and can't drift apart.
>
> **The result I'm most pleased with.** The test video carries eight hand-labelled bird
> tracks — 934 instances, median 6.0 px, the same size band as the 8.0 px drone. The
> detector fires on those birds constantly: 440 detections land on them. **Zero are ever
> raised as a target.** Three bird tracks form and the classifier rejects all three.
> Appearance can't separate a bird from a drone at six pixels. How the thing moved, over a
> whole track, can.
>
> **What I won't claim.** On the two public benchmarks, the competitor I retrained from its
> own code beats me overall — 0.834 vs 0.809 on ARD-MAV, 0.527 vs 0.487 on NPS. Below 10
> pixels the ordering reverses on every seed, but a paired significance test can't
> distinguish that from noise, so I'm calling it a consistent trend and not a result. It's
> in the README in those words.
>
> Along the way I checked why a published number on one of these benchmarks is 0.95 while
> I measure 0.527, and attributed about 78 % of the gap to three measurable causes. The
> largest, by a factor of six, wasn't a metric trick — it was simply **which videos get
> held out**. That finding changed how I report everything else.
>
> Code, every figure, and the raw per-detection evidence behind each number:
> **github.com/NadavCherry/SpeckLock**
>
> #ComputerVision #ObjectDetection #DeepLearning #Research #DroneDetection

**Length:** ~430 words — long for LinkedIn, but this audience reads. If it needs cutting,
drop the "Why this problem is hard" paragraph first; it's the most reconstructible from
context.

---

## 3 · Carousel

Five slides, each an existing figure from `docs/media/paper/`. No slide contains a number
that isn't in the repository.

| # | figure | slide title | the one line under it |
|---|---|---|---|
| **1** | `docs/media/temporal_input.jpg` | **Find the drone.** | Left: one frame — you can't, and neither can a detector. Right: the same instant as three stacked moments. Yellow = 12 frames ago, magenta = 6, cyan = now. |
| **2** | `fig5_qualitative_tiny_target.png` | **What the detector actually sees** | The same target at 6.5, 9.2, 14.5 and 20.6 px. Top row is one frame; bottom is three moments as R, G, B. The coloured edges on the buildings are parallax the stabiliser can't remove — that's the clutter the detector has to reject. |
| **3** | `fig3_single_vs_temporal.png` | **0.159 → 0.895** | Same network, same corpus, same resolution. The only variable is the input representation. |
| **4** | `fig1_accuracy_vs_size.png` | **Where it helps, and where it doesn't** | Accuracy against target size, three datasets, three seeds. Red brackets mark bins a paired test actually separated — and they're all on the competitor's side. |
| **5** | `fig4_dt_ablation.png` | **The ablation that changed my mind** | Left: tap spacing on validation — a clean peak at the value I ship. Right: the same models on held-out test. Different ranking, nothing separates. One seed would have given me either answer. |

**Order matters.** Slide 1 is the hook, 2 makes it concrete, 3 is the payoff, and 4–5 are
the credibility. Ending on the ablation that *undercuts* a design choice is deliberate: it
is the slide a serious reader remembers, and it is the one nobody else posts.

---

## 4 · Short technical version

For a comment reply, a DM, or a CV bullet.

> **SpeckLock** — detecting drones 3–14 px wide in 720p from a moving camera.
>
> Three ego-stabilised grayscale frames (t−12, t−6, t) become the R/G/B channels of one
> image, so a static world cancels and motion is left as colour. Read by a YOLOv8s with a
> stride-4 P2 head, tracked with a Kalman filter in stabilised coordinates, and classified
> at the track rather than the frame.
>
> Controlled ablation of the representation alone: **AP 0.159 → 0.895** (same network
> family, corpus, and 1280 px input). Bird rejection measured at track level: **0 raised
> over 934 labelled bird instances**, from 440 detections that land on them. Edge
> configuration: **58.9 fps at AP 0.876**, RTX 4090, TensorRT FP16, accuracy and speed from
> one pass.
>
> Compared against YOLOMG retrained from its own code on our splits and scored by one
> evaluator: it leads overall (0.834/0.809 ARD-MAV, 0.527/0.487 NPS). Below 10 px the
> ordering reverses on every seed but does not reach significance under a paired
> bootstrap + permutation test, and is reported as a trend.
>
> 950 tests. Every number points at the score-ordered detections that produced it.

---

## 5 · Answers ready for the obvious questions

Have these written before posting; the first three will be asked.

**"Did you beat the state of the art?"**
> No, not overall — YOLOMG leads on both public benchmarks and I've published the numbers.
> Below 10 px my curve is above theirs on every seed, but a paired test over 15 sequences
> can't separate it from zero, so I report it as a trend rather than a result. What I can
> defend is the representation ablation, which is controlled, and the track-level bird
> result.

**"Why not just use frame differencing / background subtraction?"**
> Because the camera is moving. You have to compensate for ego-motion before differencing
> means anything, and even then a translation-only stabiliser leaves parallax on 3D
> structure — slide 2 shows exactly that residue. The temporal stack keeps the raw
> evidence and lets the network decide, instead of thresholding it away early.

**"Isn't 3 stacked frames just optical flow / a video model?"**
> It's much cheaper than either and needs no architecture change: an off-the-shelf 2D
> detector consumes it unmodified. That's the point — the contribution is in the input, not
> the network.

**"What's the catch?"**
> Three, and they're in the README: clutter rejection is weak (11 sustained false tracks on
> the hard video), the tap-spacing choice isn't established on held-out test, and the
> interception results are simulation only — there's no flight test in this project.

**"Real-time?"**
> 58.9 fps at 1280 px on a 4090 with a TensorRT engine. Without the engine — which is what
> a fresh clone runs, since engines are architecture-specific — 35.2 fps. An older 100+ fps
> figure from an earlier round did **not** reproduce when I re-measured it, so I stopped
> quoting it.

---

## 6 · Sentences that must not appear

Each of these was true-sounding at some point in this project and is not supportable now.
The right version is beside it.

| do not write | write instead |
|---|---|
| "State of the art on tiny-drone detection" | "Competitive with a retrained SOTA baseline; it leads overall, I lead below 10 px but not significantly" |
| "Beats YOLOMG on small targets" | "Above it in every sub-10 px bin on every seed — a consistent trend that does not reach significance" |
| "100+ FPS" | "58.9 fps at 1280 px with a TensorRT engine, measured on an RTX 4090" |
| "24/24 interception success" | "24/24 with a *perfect sensor*, which measures the guidance law; with the real seeker the same mission is 0/3" |
| "Zero false positives" | "Zero *bird* false alarms at track level; 11 clutter tracks are raised on the same video" |
| "AP 1.000 on unseen video" | "AP 1.000 per frame on a *development* video — six classifier constants were tuned against it" |
| "dt = 6 is optimal" | "dt = 6 is supported by validation and not contradicted by held-out test; the sweep does not establish it" |
| "Works in all conditions" | Name the two videos and three datasets it was measured on |

---

## 7 · Before posting

- [ ] Repository is public and the README renders (it leads with the controlled result, not 24/24)
- [ ] The five figures are current — regenerate with `sbatch cluster/figures.sbatch`
- [ ] Links resolve: `python tools/check_docs.py`
- [ ] Social card is the corrected one (it claimed "1.000 on unseen video; 24/24" until this pass)
- [ ] §5 answers are to hand — the SOTA question will be the first comment
