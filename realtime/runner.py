"""Timed runner: video -> pipeline -> detections JSON + stage stats.

The tracker is stepped inline (so its cost is part of the timing) and the
track artifacts are produced afterwards by the standard offline tracker
for comparability with the PC results.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from dronedet.detections import Detection, DetectionSet
from dronedet.track import Tracker
from dronedet.video import frames

from .pipelines import RTBase


def run_pipeline(video: str, pipe: RTBase, out_dir: str,
                 min_track_score: float = 0.2) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = DetectionSet(video=video, method=pipe.name)
    tracker = Tracker(min_score=min_track_score)
    shifts = {}
    t_track = 0.0
    t0_all = time.perf_counter()
    n = 0
    for idx, frame in frames(video):
        dets = pipe.process(idx, frame)
        m = pipe.stab._acc if hasattr(pipe.stab, "_acc") else (0.0, 0.0)
        dx = m[0] / pipe.stab.scale if hasattr(pipe.stab, "scale") else 0.0
        dy = m[1] / pipe.stab.scale if hasattr(pipe.stab, "scale") else 0.0
        shifts[str(idx)] = [round(dx, 3), round(dy, 3)]
        tt = time.perf_counter()
        dets_stab = [Detection(d.x1 + dx, d.y1 + dy, d.x2 + dx, d.y2 + dy,
                               d.score, d.label) for d in dets]
        tracker.step(idx, dets_stab)
        t_track += time.perf_counter() - tt
        ds.add(idx, dets)
        n += 1
    elapsed = time.perf_counter() - t0_all
    pipe.times["tracker"] = t_track
    pipe.counts["tracker"] = n

    stage = pipe.stage_report(n)
    ds.meta = {
        "fps_end_to_end": round(n / elapsed, 2),
        "n_frames": n,
        "shifts": shifts,
        "stage_ms": {k: round(v, 2) for k, v in stage.items()},
    }
    ds.save(out / "dets.json")
    (out / "bench.json").write_text(json.dumps(ds.meta["stage_ms"] | {
        "fps_end_to_end": ds.meta["fps_end_to_end"]}, indent=2))
    print(f"[{pipe.name}] {n} frames, {ds.meta['fps_end_to_end']} fps end-to-end")
    for k, v in sorted(stage.items()):
        print(f"    {k:24s} {v:7.2f} ms/frame")
    return ds.meta
