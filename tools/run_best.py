"""Run the best pipeline (moe3-stacked + tracker) on any video.

    python tools/run_best.py --video 10_06.mp4

Pipeline: stabilization -> motion proposals (lagged-background slow-mover
detector + MOG2) -> temporal 2-class verifier (ft7: stacked stabilized
grays t-12/t-6/t) -> full-frame expert (ft1) -> compensated Kalman tracker.

Outputs under work/infer/<video-stem>/:
    dets.json / tracks_all.json / tracks_confirmed.json
    dets.mp4               detections with score >= --show-score
    tracks_all.mp4         every confirmed flying-object track
    tracks_confirmed.mp4   drone-confirmed tracks only (the "alarm" view)
    summary.txt            per-track summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.methods import build_method
from dronedet.methods.base import run_method
from dronedet.render import render_detections, render_tracks
from dronedet.track import run_tracker_file
from dronedet.video import probe

FT7 = "work/models/yolo-ft7-best.pt"   # temporal 2-class tiny specialist
FT1 = "work/models/yolo-ft-best.pt"    # near/big-object expert


def summarize(tracks_path: str, dets_path: str) -> str:
    raw = json.loads(Path(tracks_path).read_text())
    det = json.loads(Path(dets_path).read_text())
    # per-frame drone-confirmed detections for label attribution
    confirmed = {}
    for f, ds in det["frames"].items():
        for d in ds:
            if d[5] == "drone+motion" and d[4] >= 0.55:
                confirmed.setdefault(int(f), []).append(((d[0] + d[2]) / 2,
                                                         (d[1] + d[3]) / 2))
    lines = []
    for tr in sorted(raw["tracks"], key=lambda t: -t["n"]):
        fs = {int(f): v for f, v in tr["frames"].items()}
        ks = sorted(fs)
        p0, p1 = fs[ks[0]], fs[ks[-1]]
        n_conf = 0
        for f in ks:
            for (cx, cy) in confirmed.get(f, []):
                if (cx - fs[f][0]) ** 2 + (cy - fs[f][1]) ** 2 <= 16 ** 2:
                    n_conf += 1
                    break
        verdict = ("DRONE" if n_conf >= max(5, 0.25 * len(ks))
                   else "flying object (unconfirmed)")
        lines.append(
            f"id{tr['id']:>3}: frames {ks[0]:>4}-{ks[-1]:>4} (n={len(ks):>4})  "
            f"({p0[0]:6.1f},{p0[1]:6.1f}) -> ({p1[0]:6.1f},{p1[1]:6.1f})  "
            f"score {tr['score']:.2f}  drone-confirmed {n_conf}/{len(ks)}  -> {verdict}")
    return "\n".join(lines) if lines else "(no confirmed tracks)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--weights", default=FT7)
    ap.add_argument("--full-weights", default=FT1)
    ap.add_argument("--show-score", type=float, default=0.5)
    a = ap.parse_args()

    out = Path(a.out_dir or f"work/infer/{Path(a.video).stem}")
    out.mkdir(parents=True, exist_ok=True)
    info = probe(a.video, count_frames=True)
    print(f"{a.video}: {info.width}x{info.height} @ {info.fps:.1f} fps, "
          f"{info.frame_count} frames", flush=True)

    method = build_method("moe3-stacked", weights=a.weights,
                          full_weights=a.full_weights)
    ds = run_method(a.video, method)
    dets_path = out / "dets.json"
    ds.save(dets_path)
    print(f"detections -> {dets_path} ({ds.meta['fps_end_to_end']} fps)", flush=True)

    for mode, score in (("all", 0.2), ("confirmed", 0.55)):
        run_tracker_file(a.video, str(dets_path), str(out / f"tracks_{mode}.json"),
                         min_score=score)

    render_detections(a.video, str(dets_path), str(out / "dets.mp4"),
                      min_score=a.show_score)
    render_tracks(a.video, str(out / "tracks_all.json"), str(out / "tracks_all.mp4"))
    render_tracks(a.video, str(out / "tracks_confirmed.json"),
                  str(out / "tracks_confirmed.mp4"))

    summary = summarize(str(out / "tracks_all.json"), str(dets_path))
    (out / "summary.txt").write_text(summary + "\n")
    print("\n=== track summary (all-objects tracker) ===")
    print(summary)


if __name__ == "__main__":
    main()
