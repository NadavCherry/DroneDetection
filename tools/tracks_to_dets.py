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
    a = ap.parse_args()

    src = DetectionSet.load(a.src)
    raw = json.loads(Path(a.tracks).read_text())
    ds = DetectionSet(video=src.video, method=a.name or f"tracked-{src.method}")
    ds.meta = {"fps_end_to_end": src.meta.get("fps_end_to_end"),
               "n_frames": src.meta.get("n_frames"),
               "shifts": src.meta.get("shifts", {}),
               "derived_from": a.tracks}
    frames: dict[int, list[Detection]] = {}
    for tr in raw["tracks"]:
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
