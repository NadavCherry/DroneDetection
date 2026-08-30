#!/usr/bin/env python3
"""Accuracy AND speed for the edge model, measured in the same pass, on one GPU.

WHAT THIS SETTLES
-----------------
The project remembers an "ultra-fast, 100+ FPS" variant. It exists, but it is not what the
memory suggests, and three things have to be said before any number is quoted:

1. **The edge model is EDGE-RT / RT-C**: one YOLOv8n-P2 reading the same three-moment
   ego-stabilised stack, full-frame at 1280 px, no tiling, no proposal stage, no expert.
   Shipped as `final/edge_rt/edge_n1280.pt`.

2. **The 100+ FPS figure is the SAME checkpoint at half resolution.** `rt-d-full640` is not
   a separate, smaller network: hashing every tensor blob inside the two `.pt` archives
   gives the identical digest (75c9b80997c49cfeff0d over 441 tensors) for
   `full_temporal_n1280.pt` and `full_temporal_n640.pt`. One model, two export
   resolutions -- and the speed is bought with accuracy, which is the whole point of
   putting both on one axis.

3. **Every published edge FPS assumed a TensorRT engine the repository does not ship**
   (`.gitignore` excludes `*.engine`, because engines are architecture-specific). A fresh
   clone silently runs the `.pt` and gets roughly two thirds of the rate. Both are measured
   here, labelled, on the same GPU.

The published rates (74 / 84.8 / 104 fps) were taken on an **RTX 5070 Laptop**. Nothing
measured here is comparable to them: this runs on whatever GPU the job lands on, and the
engine is rebuilt for that card. The point is not to confirm the old number, it is to
produce one that can be reproduced.

HOW SPEED IS MEASURED
---------------------
`realtime.runner.run_pipeline` times one pass over the whole video and divides -- the
repository's own "end-to-end" definition, and the one behind the published figures. It
includes video decode, stabilisation, the forward with ultralytics pre/post-processing and
NMS, and the inline tracker step. It has no warm-up, so the first frames -- CUDA context
creation, cuDNN autotuning, the temporal buffer filling before any inference happens at
all -- are charged to the average.

This reports both:

  end_to_end_fps   the repo's definition, so the published numbers have a like-for-like
                   successor
  steady_state     per-frame timings after `--warmup` frames are discarded, as p50/p95/p99
                   -- because a real-time claim lives or dies on the tail, and a mean over
                   a run that includes its own warm-up flatters it

Accuracy comes from the detections that same pass produced, scored by the unified
evaluator. Speed and accuracy therefore cannot drift apart: they describe one execution.

    PYTHONPATH=. python tools/bench_edge.py \
        --weights final/edge_rt/edge_n1280.pt \
        --video data/videos/10_06.mp4 --gt realtime/work/gt_1006_v2.json \
        --imgsz 1280 640 --out work/reports/edge
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dronedet import metrics as M  # noqa: E402
from dronedet.console import use_utf8_stdio  # noqa: E402
from dronedet.detections import Detection, DetectionSet  # noqa: E402
from dronedet.gt import GroundTruth  # noqa: E402
from dronedet.track import Tracker  # noqa: E402
from dronedet.video import frames  # noqa: E402


def gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def timed_pass(video: str, pipe, warmup: int, min_track_score: float = 0.2):
    """One pass over the video, mirroring realtime.runner.run_pipeline.

    Kept as its own loop rather than calling run_pipeline because that function reports a
    single aggregate rate and this needs per-frame samples to discard warm-up and report a
    tail. The stage accounting and the tracker placement are identical to it, so the
    `end_to_end_fps` below is the same quantity the published numbers used.
    """
    ds = DetectionSet(video=video, method=pipe.name)
    tracker = Tracker(min_score=min_track_score)
    per_frame: list[float] = []
    n = 0
    t0_all = time.perf_counter()
    for idx, frame in frames(video):
        t_frame = time.perf_counter()
        dets = pipe.process(idx, frame)
        m = pipe.stab._acc if hasattr(pipe.stab, "_acc") else (0.0, 0.0)
        dx = m[0] / pipe.stab.scale if hasattr(pipe.stab, "scale") else 0.0
        dy = m[1] / pipe.stab.scale if hasattr(pipe.stab, "scale") else 0.0
        tracker.step(idx, [Detection(d.x1 + dx, d.y1 + dy, d.x2 + dx, d.y2 + dy,
                                     d.score, d.label) for d in dets])
        per_frame.append((time.perf_counter() - t_frame) * 1000.0)
        ds.add(idx, dets)
        n += 1
    elapsed = time.perf_counter() - t0_all

    tail = per_frame[warmup:] if len(per_frame) > warmup else per_frame
    s = sorted(tail)

    def q(p):
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))] if s else float("nan")

    return ds, {
        "n_frames": n,
        "end_to_end_fps": round(n / elapsed, 2),
        "warmup_frames_discarded": min(warmup, len(per_frame)),
        "steady_state": {
            "n": len(tail),
            "p50_ms": round(q(0.50), 3), "p95_ms": round(q(0.95), 3),
            "p99_ms": round(q(0.99), 3),
            "mean_ms": round(st.fmean(tail), 3) if tail else None,
            "fps_at_p50": round(1000.0 / q(0.50), 1) if s and q(0.50) > 0 else None,
        },
        "stage_ms": {k: round(v, 3) for k, v in pipe.stage_report(n).items()},
    }


def score(ds: DetectionSet, gt_path: Path, rule: str, tau: float, iou: float) -> dict:
    gt = GroundTruth.load(gt_path)
    ev = M.evaluate(gt, ds, rule=rule, tau=tau, iou_thr=iou)
    thr = M.pick_threshold(ev)
    s = M.summarise(ev, thr)
    return {"ap": round(M.average_precision(ev.records, ev.n_gt), 4),
            "recall": round(s.recall, 4), "precision": round(s.precision, 4),
            "n_gt": ev.n_gt, "n_frames_scored": ev.n_frames}


def main() -> int:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help=".pt or .engine")
    ap.add_argument("--engine-dir", default=None,
                    help="directory holding edge_n<imgsz>.engine. A TensorRT engine is "
                         "built for ONE input shape, so each resolution needs its own "
                         "file; passing a single engine path and reusing it across "
                         "--imgsz would silently run the 1280 engine at 640.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[1280, 640])
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=20,
                    help="frames discarded before the steady-state percentiles; the "
                         "temporal buffer alone needs 2*dt frames before it infers at all")
    ap.add_argument("--rule", default="centre", choices=("centre", "iou"))
    ap.add_argument("--tau", type=float, default=12.0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    from realtime.pipelines import FullFramePipeline
    from realtime.rt_models import Detector

    # (backend, imgsz, weights). The .pt is shape-agnostic and is run at every
    # resolution; each engine is bound to the one shape it was built for.
    jobs: list[tuple[str, int, str]] = [("pt", sz, a.weights) for sz in a.imgsz]
    if a.engine_dir:
        for sz in a.imgsz:
            e = Path(a.engine_dir) / f"edge_n{sz}.engine"
            if e.exists():
                jobs.append(("engine", sz, str(e)))
            else:
                print(f"  no engine for imgsz {sz} at {e} -- .pt only at that size")

    rows = []
    for backend, imgsz, wpath in jobs:
        if not Path(wpath).exists():
            print(f"  skip {backend}@{imgsz}: {wpath} absent")
            continue
        if True:
            name = f"edge-{backend}-{imgsz}"
            print(f"\n===== {name} =====", flush=True)
            pipe = FullFramePipeline(name, Detector(wpath, imgsz, conf=a.conf),
                                     temporal=True)
            ds, timing = timed_pass(a.video, pipe, a.warmup)
            acc = score(ds, a.gt, a.rule, a.tau, a.iou)
            row = {"arm": name, "backend": backend, "imgsz": imgsz,
                   "weights": str(wpath), "gpu": gpu_name(), **timing, **acc}
            rows.append(row)
            print(json.dumps({k: row[k] for k in
                              ("arm", "end_to_end_fps", "ap", "recall", "precision")},
                             indent=1), flush=True)
            if a.out:
                a.out.mkdir(parents=True, exist_ok=True)
                ds.save(a.out / f"{name}-dets.json")

    if not rows:
        raise SystemExit("no arms ran")

    L = ["# Edge model: accuracy against speed", "",
         f"GPU: **{rows[0]['gpu']}**. Video: `{Path(a.video).name}`, scored against "
         f"`{a.gt.name}` (rule={a.rule}, tau={a.tau}). Speed and accuracy come from the "
         "SAME pass, so they describe one execution and cannot drift apart.", "",
         "The published 74 / 84.8 / 104 fps figures were taken on an RTX 5070 Laptop and "
         "are **not** comparable to these.", "",
         "| arm | backend | imgsz | AP | recall | precision | end-to-end fps | p50 ms | p95 ms | p99 ms |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ss = r["steady_state"]
        L.append(f"| {r['arm']} | {r['backend']} | {r['imgsz']} | **{r['ap']:.4f}** | "
                 f"{r['recall']:.3f} | {r['precision']:.3f} | **{r['end_to_end_fps']}** | "
                 f"{ss['p50_ms']} | {ss['p95_ms']} | {ss['p99_ms']} |")
    L += ["", "Per-stage means (ms/frame):", "",
          "| arm | " + " | ".join(sorted(rows[0]["stage_ms"])) + " |",
          "|---|" + "---|" * len(rows[0]["stage_ms"])]
    for r in rows:
        L.append(f"| {r['arm']} | "
                 + " | ".join(str(r["stage_ms"].get(k, "—"))
                              for k in sorted(rows[0]["stage_ms"])) + " |")
    L.append("")

    out = "\n".join(L) + "\n"
    print(out)
    if a.out:
        (a.out / "edge_bench.md").write_text(out, encoding="utf-8")
        (a.out / "edge_bench.json").write_text(json.dumps(rows, indent=2),
                                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
