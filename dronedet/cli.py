"""Command-line interface.

    python -m dronedet detect --video data/videos/07_05.mp4 --method motion-median --out work/det/motion-median.json
    python -m dronedet eval   --gt work/gt.json --dets work/det/*.json --out work/eval.md
    python -m dronedet track  --video data/videos/07_05.mp4 --dets work/det/hybrid.json --out work/tracks.json
    python -m dronedet render --video data/videos/07_05.mp4 --dets work/det/hybrid.json --out work/vis.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from .console import use_utf8_stdio


def cmd_detect(a: argparse.Namespace) -> None:
    from .methods import build_method
    from .methods.base import run_method

    kw = json.loads(a.method_kw) if a.method_kw else {}
    if a.weights:
        kw["weights"] = a.weights
    method = build_method(a.method, **kw)
    ds = run_method(a.video, method, stop=a.stop, stab_mode=a.stab)
    out = a.out or f"work/det/{a.method}.json"
    ds.save(out)
    print(f"saved {sum(len(v) for v in ds.frames.values())} detections "
          f"over {ds.meta['n_frames']} frames to {out} "
          f"({ds.meta['fps_end_to_end']} fps end-to-end)")


def cmd_eval(a: argparse.Namespace) -> None:
    from .evaluate import evaluate_files

    fr = None
    if a.frames:
        lo, hi = a.frames.split(":")
        fr = (int(lo), int(hi))
    report = evaluate_files(a.gt, a.dets, tau=a.tau, min_score=a.min_score,
                            frame_range=fr)
    print(report)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(report, encoding="utf-8")


def cmd_bench(a: argparse.Namespace) -> None:
    from .detections import DetectionSet
    from .gt import GroundTruth
    from . import metrics as M

    fr = None
    if a.frames:
        lo, hi = a.frames.split(":")
        fr = (int(lo), int(hi))
    targets = set(a.targets) if a.targets else None
    gt = GroundTruth.load(a.gt)

    confusers = tuple(a.confusers)
    distractor_names = [n for n, o in gt.objects.items()
                        if (o.ignore if targets is None else n not in targets)]
    confuser_names = [n for n in distractor_names if n.startswith(confusers)]
    n_conf_inst = sum(len(gt.objects[n].frames) for n in confuser_names)

    lines = [
        f"# Benchmark report (tau={a.tau} px)",
        "",
        f"Ground truth `{a.gt}` — targets: "
        f"{', '.join(sorted(targets)) if targets else 'all non-ignore objects'}.",
        f"**Confusers** ({len(confuser_names)} objects, {n_conf_inst} instances, prefix "
        f"{'/'.join(confusers)}): {', '.join(sorted(confuser_names)) or 'none'}. These are the "
        "things that must never be called a drone; a hit on one is the failure this pipeline "
        "exists to prevent, and it is reported here rather than silently discarded.",
        f"Other distractors (excluded from recall, hits not held against a method): "
        f"{', '.join(sorted(set(distractor_names) - set(confuser_names))) or 'none'}.",
        "",
        "| method | AP(centre) | AP very-tiny | AP tiny | COCO AP | AP50 | P | R | FP/frame | "
        "**confuser hits** | med err px |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for path in a.dets:
        ds = DetectionSet.load(path)
        ev = M.evaluate(gt, ds, rule="centre", tau=a.tau, targets=targets, frame_range=fr)
        thr = a.threshold if a.threshold is not None else M.pick_threshold(ev)
        s = M.summarise(ev, thr)
        coco = M.coco_ap(gt, ds, targets=targets, frame_range=fr)
        by = s.ap_by_size
        hits, inst = s.confuser_hits(confusers)
        lines.append(
            f"| {ds.method} | {s.ap:.3f} | {by.get('very-tiny', float('nan')):.3f} | "
            f"{by.get('tiny', float('nan')):.3f} | {coco['AP']:.3f} | {coco['AP50']:.3f} | "
            f"{s.precision:.3f} | {s.recall:.3f} | {s.fp_per_frame:.3f} | "
            f"**{hits}/{inst}** | {s.median_centre_error:.1f} |")
        if a.ci:
            lo, hi = M.bootstrap_ci(ev, block=a.block, n_resamples=a.resamples)
            lines.append(f"| ↳ *95% CI on AP(centre)* | *[{lo:.3f}, {hi:.3f}]* "
                         f"| | | | | | | | | |")

    lines += [
        "",
        f"Operating point: {'fixed --threshold ' + str(a.threshold) if a.threshold is not None else 'best-F1 chosen on THIS set (optimistic — pass --threshold from a val run for an honest number)'}.",
        "COCO AP is AP@[.50:.05:.95] on IoU, the metric published drone papers report. It is "
        "0.000 for any method whose boxes do not carry real extent, however well it localises "
        "the centre — see `dronedet/metrics.py`.",
    ]
    report = "\n".join(lines)
    print(report)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(report, encoding="utf-8")


def cmd_track(a: argparse.Namespace) -> None:
    from .track import run_tracker_file

    run_tracker_file(a.video, a.dets, a.out, video_out=a.video_out,
                     min_score=a.min_score)


def cmd_render(a: argparse.Namespace) -> None:
    from .render import render_detections

    render_detections(a.video, a.dets, a.out, min_score=a.min_score, zoom_best=True)


def main() -> None:
    use_utf8_stdio()
    p = argparse.ArgumentParser(prog="dronedet")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="run a detection method over a video")
    d.add_argument("--video", required=True)
    d.add_argument("--method", required=True)
    d.add_argument("--out")
    d.add_argument("--weights", help="override model weights path")
    d.add_argument("--stop", type=int)
    d.add_argument("--stab", default="translation",
                   choices=["translation", "affine", "off"])
    d.add_argument("--method-kw", help="extra method kwargs as JSON")
    d.set_defaults(fn=cmd_detect)

    e = sub.add_parser("eval", help="score detection JSONs against GT")
    e.add_argument("--gt", required=True)
    e.add_argument("--dets", nargs="+", required=True)
    e.add_argument("--tau", type=float, default=12.0, help="center-distance match radius (px)")
    e.add_argument("--min-score", type=float, default=None)
    e.add_argument("--frames", help="restrict scoring to a frame range, e.g. 342:571")
    e.add_argument("--out")
    e.set_defaults(fn=cmd_eval)

    b = sub.add_parser("bench", help="benchmark-grade scoring: centre-distance AND COCO AP, "
                                     "size bins, distractor (bird) hits, bootstrap CIs")
    b.add_argument("--gt", required=True)
    b.add_argument("--dets", nargs="+", required=True)
    b.add_argument("--tau", type=float, default=12.0)
    b.add_argument("--targets", nargs="+",
                   help="object names counting as positives; all others become distractors")
    b.add_argument("--threshold", type=float, default=None,
                   help="operating threshold, normally taken from a val run; "
                        "omitted means best-F1 on this set, which is optimistic")
    b.add_argument("--confusers", nargs="+", default=["bird"],
                   help="name prefixes of the distractors that must never be called a drone "
                        "(default: bird). Hits on these get their own column.")
    b.add_argument("--frames", help="restrict scoring to a frame range, e.g. 342:571")
    b.add_argument("--ci", action="store_true", help="add a block-bootstrap 95%% CI on AP")
    b.add_argument("--block", type=int, default=30, help="bootstrap block length in frames")
    b.add_argument("--resamples", type=int, default=2000)
    b.add_argument("--out")
    b.set_defaults(fn=cmd_bench)

    t = sub.add_parser("track", help="run tracker over a detection JSON")
    t.add_argument("--video", required=True)
    t.add_argument("--dets", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--video-out")
    t.add_argument("--min-score", type=float, default=0.25)
    t.set_defaults(fn=cmd_track)

    r = sub.add_parser("render", help="render detections onto the video")
    r.add_argument("--video", required=True)
    r.add_argument("--dets", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--min-score", type=float, default=0.25)
    r.set_defaults(fn=cmd_render)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
