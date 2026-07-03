"""Fuse detection JSONs from multiple detectors (weighted-box-fusion
adapted to center-distance land: boxes whose centers agree within a
radius merge into one detection at the score-weighted mean position with
a noisy-OR score -- agreement between independent detectors is evidence,
not a duplicate to discard, which is what plain NMS would do).

    python tools/fuse_dets.py --dets a.json b.json --weights 1.0 0.8 \
        --out fused.json --name pc-max
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.detections import Detection, DetectionSet

RADIUS = 10.0


def fuse_frame(groups: list[list[Detection]], weights: list[float]) -> list[Detection]:
    """groups[i] = detections of source i (already weighted by caller)."""
    pool = []
    for i, dets in enumerate(groups):
        for d in dets:
            pool.append((i, d))
    pool.sort(key=lambda t: -t[1].score * weights[t[0]])
    used = [False] * len(pool)
    out = []
    for a, (ia, da) in enumerate(pool):
        if used[a]:
            continue
        cluster = [(ia, da)]
        used[a] = True
        for b in range(a + 1, len(pool)):
            if used[b]:
                continue
            ib, db = pool[b]
            if (da.cx - db.cx) ** 2 + (da.cy - db.cy) ** 2 <= RADIUS ** 2:
                cluster.append((ib, db))
                used[b] = True
        ws = [weights[i] * d.score for i, d in cluster]
        cx = sum(w * d.cx for w, (_, d) in zip(ws, cluster)) / sum(ws)
        cy = sum(w * d.cy for w, (_, d) in zip(ws, cluster)) / sum(ws)
        # noisy-OR over clamped per-source scores: agreement raises score,
        # a single source alone keeps its own calibrated score
        p = 1.0
        for w, (_, d) in zip(ws, cluster):
            p *= 1.0 - min(0.99, w)
        score = 1.0 - p
        best = max(cluster, key=lambda t: t[1].score * weights[t[0]])[1]
        w0 = max(d.w for _, d in cluster)
        h0 = max(d.h for _, d in cluster)
        out.append(Detection(cx - w0 / 2, cy - h0 / 2, cx + w0 / 2, cy + h0 / 2,
                             score, best.label))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="fused")
    a = ap.parse_args()
    sets = [DetectionSet.load(p) for p in a.dets]
    weights = a.weights or [1.0] * len(sets)
    assert len(weights) == len(sets)

    ds = DetectionSet(video=sets[0].video, method=a.name)
    ds.meta = {"fps_end_to_end": min((s.meta.get("fps_end_to_end") or 1e9)
                                     for s in sets),
               "n_frames": max(s.meta.get("n_frames") or 0 for s in sets),
               "shifts": next((s.meta["shifts"] for s in sets
                               if s.meta.get("shifts")), {}),
               "sources": [s.method for s in sets]}
    all_frames = sorted(set().union(*[set(s.frames) for s in sets]))
    for f in all_frames:
        ds.frames[f] = fuse_frame([s.frames.get(f, []) for s in sets], weights)
    ds.save(a.out)
    print(f"fused {len(sets)} sources -> {a.out} "
          f"({sum(len(v) for v in ds.frames.values())} dets)")


if __name__ == "__main__":
    main()
