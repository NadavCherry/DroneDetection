"""Convert tracker output back into a DetectionSet ("track-before-detect"
style temporal integration): the tracker's persistence turns flickery
per-frame detections into continuous ones, and coasting fills gaps.
Scored like any other detection method.

    python tools/tracks_to_dets.py --tracks work/tracks/moe2-all.json \
        --src work/det2/moe2-hybrid.json --out work/det2/tracked-moe2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.detections import Detection, DetectionSet

STATUS_FACTOR = {"tracked": 1.0, "reacq": 0.8, "coast": 0.55}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--src", required=True, help="detection JSON the tracker consumed "
                                                 "(for video/meta fields)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--classify", action="store_true",
                    help="drop tracks the track-level classifier calls 'other' "
                         "(foliage/birds/unknown movers)")
    ap.add_argument("--smooth-coast", action="store_true",
                    help="offline smoothing: linearly interpolate coast "
                         "positions across gaps bounded by real detections "
                         "on both sides (one-sided KF coasting drifts)")
    a = ap.parse_args()

    src = DetectionSet.load(a.src)
    raw = json.loads(Path(a.tracks).read_text())
    keep_cls = None
    if a.classify:
        from dronedet.trackclass import classify_tracks

        keep_cls = classify_tracks(raw, json.loads(Path(a.src).read_text()))
        for tid, info in sorted(keep_cls.items()):
            print(f"  track {tid}: {info['cls']} (conf_frac {info['conf_frac']}, "
                  f"n_conf {info['n_conf']}, n {info['n']})")
    ds = DetectionSet(video=src.video, method=a.name or f"tracked-{src.method}")
    ds.meta = {"fps_end_to_end": src.meta.get("fps_end_to_end"),
               "n_frames": src.meta.get("n_frames"),
               "shifts": src.meta.get("shifts", {}),
               "derived_from": a.tracks}
    frames: dict[int, list[Detection]] = {}
    for tr in raw["tracks"]:
        if keep_cls is not None and keep_cls[tr["id"]]["cls"] == "other":
            continue
        if a.smooth_coast:
            fs = {int(f): list(v) for f, v in tr["frames"].items()}
            ks = sorted(fs)
            anchors = [f for f in ks if fs[f][4] != "coast"]
            for f in ks:
                if fs[f][4] != "coast":
                    continue
                lo = max((k for k in anchors if k < f), default=None)
                hi = min((k for k in anchors if k > f), default=None)
                if lo is not None and hi is not None and hi - lo <= 60:
                    w = (f - lo) / (hi - lo)
                    fs[f][0] = fs[lo][0] * (1 - w) + fs[hi][0] * w
                    fs[f][1] = fs[lo][1] * (1 - w) + fs[hi][1] * w
            tr = {**tr, "frames": {str(f): fs[f] for f in ks}}
        tscore = float(tr.get("score", 0.5))
        for f, (cx, cy, w, h, status) in tr["frames"].items():
            s = min(1.0, max(0.05, tscore)) * STATUS_FACTOR.get(status, 0.5)
            frames.setdefault(int(f), []).append(
                Detection(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, s,
                          f"track{tr['id']}:{status}"))
    for f in range(int(src.meta.get("n_frames", 0) or (max(frames) + 1))):
        ds.frames[f] = frames.get(f, [])
    ds.save(a.out)
    print(f"{sum(len(v) for v in ds.frames.values())} track-detections -> {a.out}")


if __name__ == "__main__":
    main()
