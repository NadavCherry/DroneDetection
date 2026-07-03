"""Track-level classification: which confirmed tracks are drones?

Per-frame appearance at 3-14 px is weak (a bird and a drone genuinely
converge at 4 px), but aggregated over a track's lifetime the 2-class
verifier's opinion separates cleanly. Measured on 07_05 (moe3) + 10_06:

    track                             conf_frac   n_conf
    far drone (07_05)                    0.99       432
    drone (10_06)                        0.94       203
    landed/near drone (07_05)            1.00       571   (big boxes)
    all foliage-clutter tracks        0.00-0.02      0-1
    brief unlabeled sky object           1.00         4   <- n_conf kills it

Rule: a track is a DRONE iff at least CONF_FRAC of its real detections
are verifier-confirmed drone detections (label 'drone*', score >= 0.5)
AND it has at least N_CONF such detections (sustained evidence -- a
4-detection flash is an anecdote, not a track). Tracks whose detections
are mostly large boxes are the landed/near drone ('near'); everything
else is 'other' (foliage, birds, unknown movers).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

CONF_FRAC = 0.50   # fraction of tracked dets that must be drone-confirmed
N_CONF = 8         # minimum confirmed detections (sustained evidence)
BIG_W = 60.0       # box width that says "near/landed drone", not the far target
MATCH_DIST = 8.0   # track-position -> detection association radius


def classify_tracks(tracks: dict, dets: dict) -> dict[int, dict]:
    """tracks/dets: parsed JSON payloads (tracker output + the detection
    set it consumed). Returns {track_id: {cls, conf_frac, n_conf, ...}}."""
    det_frames = {int(f): v for f, v in dets["frames"].items()}
    out = {}
    for tr in tracks["tracks"]:
        fs = {int(f): v for f, v in tr["frames"].items()}
        tracked = [(f, v) for f, v in fs.items() if v[4] == "tracked"]
        n_conf = 0
        n_big = 0
        n_matched = 0
        for f, v in tracked:
            best, bd = None, MATCH_DIST
            for d in det_frames.get(f, []):
                cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
                dist = math.hypot(cx - v[0], cy - v[1])
                if dist < bd:
                    bd, best = dist, d
            if best is None:
                continue
            n_matched += 1
            label, score, w = best[5], best[4], best[2] - best[0]
            if label.startswith("drone") and score >= 0.5:
                n_conf += 1
            if w >= BIG_W:
                n_big += 1
        conf_frac = n_conf / max(n_matched, 1)
        big_frac = n_big / max(n_matched, 1)
        if conf_frac >= CONF_FRAC and n_conf >= N_CONF:
            cls = "near" if big_frac >= 0.5 else "drone"
        else:
            cls = "other"
        out[tr["id"]] = {
            "cls": cls, "conf_frac": round(conf_frac, 3), "n_conf": n_conf,
            "big_frac": round(big_frac, 3), "n_tracked": len(tracked),
            "n": len(fs),
        }
    return out


def classify_files(tracks_path: str | Path, dets_path: str | Path) -> dict[int, dict]:
    return classify_tracks(json.loads(Path(tracks_path).read_text()),
                           json.loads(Path(dets_path).read_text()))
