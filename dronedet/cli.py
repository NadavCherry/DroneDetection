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
        Path(a.out).write_text(report)


def cmd_track(a: argparse.Namespace) -> None:
    from .track import run_tracker_file

    run_tracker_file(a.video, a.dets, a.out, video_out=a.video_out,
                     min_score=a.min_score)


def cmd_render(a: argparse.Namespace) -> None:
    from .render import render_detections

    render_detections(a.video, a.dets, a.out, min_score=a.min_score, zoom_best=True)


def main() -> None:
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
