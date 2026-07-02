"""Render detections / tracks / GT onto an annotated output video."""

from __future__ import annotations

import json
from pathlib import Path

from .detections import DetectionSet, Detection
from .gt import GroundTruth
from .video import frames, probe
from .viz import COLORS, VideoSink, draw_box, zoom_inset


def render_detections(video: str, dets_path: str, out: str,
                      min_score: float = 0.25, gt_path: str | None = None,
                      zoom_best: bool = True) -> None:
    ds = DetectionSet.load(dets_path)
    gt = GroundTruth.load(gt_path) if gt_path else None
    info = probe(video)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sink = VideoSink(out, info.fps, (info.width, info.height))
    for idx, frame in frames(video):
        if gt is not None:
            for name, obj in gt.objects.items():
                b = obj.box(idx)
                if b:
                    cx, cy, w, h = b
                    d = Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 1.0)
                    draw_box(frame, d, COLORS["gt"], label=f"gt:{name}", pad=4)
        best = None
        for d in ds.frames.get(idx, []):
            if d.score < min_score:
                continue
            draw_box(frame, d, COLORS["det"], label=f"{d.label} {d.score:.2f}")
            if best is None or d.score > best.score:
                best = d
        if zoom_best and best is not None:
            zoom_inset(frame, best.cx, best.cy)
        sink.write(frame)
    sink.close()
    print(f"wrote {out}")


def render_tracks(video: str, tracks_path: str, out: str) -> None:
    raw = json.loads(Path(tracks_path).read_text())
    per_frame: dict[int, list] = {}
    trails: dict[int, list] = {}
    for tr in raw["tracks"]:
        for f, (cx, cy, w, h, status) in tr["frames"].items():
            per_frame.setdefault(int(f), []).append(
                (tr["id"], cx, cy, w, h, status))
    info = probe(video)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sink = VideoSink(out, info.fps, (info.width, info.height))
    import cv2

    for idx, frame in frames(video):
        focus = None
        for (tid, cx, cy, w, h, status) in per_frame.get(idx, []):
            color = COLORS["track"] if status == "tracked" else COLORS["coast"]
            d = Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 1.0)
            draw_box(frame, d, color, label=f"id{tid}" + ("" if status == "tracked" else "?"))
            trail = trails.setdefault(tid, [])
            trail.append((int(cx), int(cy)))
            for a, b in zip(trail[-90:], trail[-89:]):
                cv2.line(frame, a, b, color, 1, cv2.LINE_AA)
            if focus is None or status == "tracked":
                focus = (cx, cy)
        if focus:
            zoom_inset(frame, focus[0], focus[1])
        sink.write(frame)
    sink.close()
    print(f"wrote {out}")
