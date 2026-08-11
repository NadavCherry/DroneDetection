"""Score tracker output against GT: coverage, ID switches, false tracks."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.gt import GroundTruth

TAU = 16.0


def score(gt_path: str, tracks_path: str) -> dict:
    gt = GroundTruth.load(gt_path)
    raw = json.loads(Path(tracks_path).read_text())
    excl = set(gt.meta.get("exclude_frames", []))

    # per-frame track points
    per_frame: dict[int, list] = {}
    for tr in raw["tracks"]:
        for f, v in tr["frames"].items():
            per_frame.setdefault(int(f), []).append((tr["id"], v[0], v[1], v[4]))

    out = {"tracks_file": tracks_path, "dets": raw.get("dets"),
           "n_confirmed_tracks": len(raw["tracks"])}

    for name, obj in gt.objects.items():
        if obj.ignore:
            continue
        gtf = {f: b for f, b in obj.frames.items() if f not in excl}
        covered, ids, errs = [], [], []
        for f, (cx, cy, w, h) in sorted(gtf.items()):
            r = max(TAU, 0.5 * math.sqrt(w * h))
            best, bestd = None, 1e9
            for (tid, tx, ty, status) in per_frame.get(f, []):
                d = math.hypot(tx - cx, ty - cy)
                if d <= r and d < bestd:
                    best, bestd = tid, d
            covered.append(best is not None)
            if best is not None:
                ids.append(best)
                errs.append(bestd)
        switches = sum(1 for a, b in zip(ids, ids[1:]) if a != b)
        streak = best_streak = 0
        for c in covered:
            streak = streak + 1 if c else 0
            best_streak = max(best_streak, streak)
        out[name] = {
            "gt_frames": len(gtf),
            "coverage": round(sum(covered) / max(len(gtf), 1), 3),
            "id_switches": switches,
            "longest_streak": best_streak,
            "med_err_px": round(sorted(errs)[len(errs) // 2], 2) if errs else None,
            "n_ids": len(set(ids)),
        }

    # false tracks: mostly matching nothing
    gt_all = {f: [] for f in range(100000)}
    false_tracks = 0
    for tr in raw["tracks"]:
        unmatched = 0
        total = 0
        for f, v in tr["frames"].items():
            fi = int(f)
            if fi in excl:
                continue
            total += 1
            ok = False
            for name, obj in gt.objects.items():
                b = obj.box(fi)
                if b is None:
                    continue
                r = max(TAU, 0.5 * math.sqrt(b[2] * b[3]))
                if math.hypot(v[0] - b[0], v[1] - b[1]) <= r:
                    ok = True
                    break
            unmatched += not ok
        if total and unmatched / total > 0.5:
            false_tracks += 1
    out["false_tracks"] = false_tracks
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="work/gt.json")
    ap.add_argument("--tracks", nargs="+", required=True)
    a = ap.parse_args()
    for p in a.tracks:
        print(json.dumps(score(a.gt, p), indent=2))
