"""Run one of the two final models on a video.

    # most powerful (desktop GPU):
    .venv/bin/python final/run_final.py --video some.mp4 --profile pc-max --out out_dir

    # real-time / edge profile (one nano net; ~85 fps on an RTX 5070,
    # projected 10-15 fps FP16 on a Jetson Orin Nano at 1280):
    .venv/bin/python final/run_final.py --video some.mp4 --profile edge-rt --out out_dir

Outputs in --out:
    dets.json            per-frame detections (original coords) + fps + stage timings
    tracks.json          all confirmed tracks (tracker output)
    tracks_drone.json    only classified drone/near tracks
    tracked_dets.json    track-integrated detections (the headline metric input)
    alarms.txt           per drone track: start frame, confirmation frame/latency, coverage
    annotated.mp4        video with the classified tracks painted

The edge profile prefers the TensorRT engine when present (build it on the
target device: `yolo export model=edge_rt/edge_n1280.pt format=engine half=True
imgsz=1280 batch=1`); it falls back to the .pt weights anywhere else.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

final = Path(__file__).resolve().parent


def run_edge(video: str, out: Path) -> None:
    from realtime.pipelines import FullFramePipeline
    from realtime.rt_models import Detector
    from realtime.runner import run_pipeline

    eng = final / "edge_rt/edge_n1280.engine"
    weights = str(eng if eng.exists() else final / "edge_rt/edge_n1280.pt")
    pipe = FullFramePipeline("edge-rt", Detector(weights, 1280), temporal=True)
    run_pipeline(video, pipe, str(out))


def run_pcmax(video: str, out: Path) -> None:
    import subprocess

    from realtime.pipelines import FullFramePipeline
    from realtime.rt_models import Detector
    from realtime.runner import run_pipeline

    py = sys.executable
    kw = json.dumps({"weights": str(final / "pc_max/verifier640.pt"),
                     "full_weights": str(final / "pc_max/expert1280.pt")})
    subprocess.run([py, "-m", "dronedet", "detect", "--video", video,
                    "--method", "moe3-stacked", "--method-kw", kw,
                    "--out", str(out / "moe3.json")], check=True, cwd=ROOT)
    pipe = FullFramePipeline("fullS-s1280",
                             Detector(str(final / "pc_max/fullS.pt"), 1280),
                             temporal=True)
    run_pipeline(video, pipe, str(out / "fullS"))
    subprocess.run([py, "tools/fuse_dets.py", "--dets",
                    str(out / "fullS/dets.json"), str(out / "moe3.json"),
                    "--out", str(out / "dets.json"), "--name", "pc-max"],
                   check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--profile", choices=["pc-max", "edge-rt"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. detections
    if a.profile == "edge-rt":
        run_edge(a.video, out)
    else:
        run_pcmax(a.video, out)

    # 2. tracker (camera-motion compensated, with local re-acquisition)
    from dronedet.track import run_tracker_file

    run_tracker_file(a.video, str(out / "dets.json"), str(out / "tracks.json"),
                     min_score=0.2)

    # 3. track-level classification
    from dronedet.trackclass import classify_files

    cls = classify_files(out / "tracks.json", out / "dets.json")
    raw = json.loads((out / "tracks.json").read_text())
    keep = [tr for tr in raw["tracks"] if cls[tr["id"]]["cls"] != "other"]
    (out / "tracks_drone.json").write_text(json.dumps(
        {**raw, "tracks": keep, "classification":
         {str(k): v for k, v in cls.items()}}))

    # 4. track-integrated detections (drone/near tracks only)
    import subprocess

    subprocess.run([sys.executable, "tools/tracks_to_dets.py",
                    "--tracks", str(out / "tracks.json"),
                    "--src", str(out / "dets.json"),
                    "--out", str(out / "tracked_dets.json"),
                    "--name", f"tracked-{a.profile}", "--classify",
                    "--smooth-coast"],
                   check=True, cwd=ROOT)

    # 5. alarm summary
    lines = []
    dets = json.loads((out / "dets.json").read_text())
    det_frames = {int(f): v for f, v in dets["frames"].items()}
    for tr in keep:
        info = cls[tr["id"]]
        fs = sorted(int(f) for f in tr["frames"])
        conf_times = []
        for f in fs:
            v = tr["frames"][str(f)]
            if v[4] != "tracked":
                continue
            for d in det_frames.get(f, []):
                cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
                if (math.hypot(cx - v[0], cy - v[1]) < 8
                        and d[5].startswith("drone") and d[4] >= 0.5):
                    conf_times.append(f)
                    break
        confirm = conf_times[7] if len(conf_times) >= 8 else None
        lines.append(
            f"track {tr['id']} [{info['cls']}]: frames {fs[0]}-{fs[-1]} "
            f"({len(fs)} covered), confirmed at frame {confirm} "
            f"(latency {confirm - fs[0]} frames)" if confirm is not None else
            f"track {tr['id']} [{info['cls']}]: frames {fs[0]}-{fs[-1]} "
            f"({len(fs)} covered), not confirmed")
    (out / "alarms.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines) or "no drone tracks")

    # 6. annotated video
    if not a.no_video:
        from dronedet.render import render_tracks

        render_tracks(a.video, str(out / "tracks_drone.json"),
                      str(out / "annotated.mp4"))


if __name__ == "__main__":
    main()
