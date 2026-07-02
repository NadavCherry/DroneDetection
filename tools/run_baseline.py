"""Run a plain detection model (no tracking, no motion, no fusion) on a
video and render it with the same visualization as the full pipeline --
for baseline presentations.

    python tools/run_baseline.py --weights baseline/yolo26n-...pt --video 10_06.mp4

The model runs full-frame at its own training imgsz (read from the
checkpoint, overridable with --imgsz). Detections are saved at a low
threshold (--conf) so the JSON keeps the full curve; the video is
rendered at --show-score (default 0.25, YOLO's usual display threshold).

If a reference track exists (work/infer/<stem>/tracks_confirmed.json from
tools/run_best.py), the summary reports the baseline's hit-rate along it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.methods import build_method
from dronedet.methods.base import run_method
from dronedet.render import render_detections
from dronedet.video import probe


def train_imgsz(weights: str) -> int | None:
    import torch

    try:
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        v = ck.get("train_args", {}).get("imgsz")
        return int(v) if v else None
    except Exception:
        return None


def hit_rate_along_reference(dets_path: str, ref_tracks: str,
                             thresh: float, tau: float = 12.0) -> str:
    det = json.loads(Path(dets_path).read_text())
    ref = json.loads(Path(ref_tracks).read_text())
    if not ref["tracks"]:
        return "no reference track"
    lines = []
    for tr in ref["tracks"]:
        hits = total = 0
        for f, (cx, cy, w, h, status) in tr["frames"].items():
            if status == "coast":
                continue
            total += 1
            for d in det["frames"].get(f, []):
                if d[4] < thresh:
                    continue
                dx = (d[0] + d[2]) / 2 - cx
                dy = (d[1] + d[3]) / 2 - cy
                if math.hypot(dx, dy) <= tau:
                    hits += 1
                    break
        lines.append(f"reference track id{tr['id']} ({total} measured frames): "
                     f"baseline hit {hits} ({hits / max(total, 1):.1%}) at conf>={thresh}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--show-score", type=float, default=0.25)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    imgsz = a.imgsz or train_imgsz(a.weights) or 640
    stem = Path(a.video).stem
    out = Path(a.out_dir or f"work/infer/{stem}_baseline")
    out.mkdir(parents=True, exist_ok=True)
    info = probe(a.video, count_frames=True)
    print(f"{a.video}: {info.width}x{info.height}, {info.frame_count} frames | "
          f"baseline imgsz {imgsz}", flush=True)

    method = build_method("yolo-ft", weights=a.weights, conf=a.conf)
    method.imgsz = imgsz
    method.name = f"baseline-{Path(a.weights).stem}"
    ds = run_method(a.video, method)
    dets_path = out / "dets.json"
    ds.save(dets_path)
    n_show = sum(1 for v in ds.frames.values() for d in v if d.score >= a.show_score)
    print(f"{sum(len(v) for v in ds.frames.values())} detections saved "
          f"({n_show} at conf>={a.show_score}) -> {dets_path} "
          f"({ds.meta['fps_end_to_end']} fps)")

    render_detections(a.video, str(dets_path), str(out / "dets.mp4"),
                      min_score=a.show_score)

    ref = Path(f"work/infer/{stem}/tracks_confirmed.json")
    summary = []
    if ref.exists():
        for th in (a.show_score, 0.1, a.conf):
            summary.append(hit_rate_along_reference(str(dets_path), str(ref), th))
        (out / "summary.txt").write_text("\n".join(summary) + "\n")
        print("\n=== baseline vs pipeline reference track ===")
        print("\n".join(summary))


if __name__ == "__main__":
    main()
