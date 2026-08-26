"""Score tracker output against GT: coverage, ID switches, false tracks.

A track is scored as one of three kinds, not two:

    true        the plurality of its frames sit on a TARGET object
    distractor  ...on an ``ignore`` object -- a labelled bird. The system followed a real
                flying animal and called it a target.
    clutter     ...on nothing at all

`false_tracks` counts distractor + clutter + ambiguous. It previously counted only
"matched no GT object of any kind", which forgave every track that rode a bird, because
the loop that decided it iterated `gt.objects` without checking `ignore` -- while the
coverage loop in the same function does check it. See the comment at the false-track
section for the measured cost of that inconsistency.
"""

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
    raw = json.loads(Path(tracks_path).read_text(encoding="utf-8"))
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

    # ------------------------------------------------------------------ false tracks
    # Every track is sorted into one of three kinds, by where the plurality of its
    # frames sits.
    #
    # THE BUG THIS REPLACES. The previous version looped `for name, obj in
    # gt.objects.items()` with no `ignore` check, so a track that rode a labelled BIRD
    # for its entire life "matched a GT object" and was therefore not a false track --
    # while the coverage loop twenty lines above does skip ignore objects. One file, two
    # conventions. On work/tracks3/0705/pc-max-all.json it reported 14 false tracks where
    # an ignore-aware count reports 19, and three of the five it forgave were riding
    # labelled birds.
    #
    # That is not a rounding difference in a side statistic. Bird rejection is the
    # hardest thing this pipeline does and the thing the whole project is sold on, so a
    # metric that silently forgives bird tracks was hiding exactly the failure mode that
    # matters most -- and hiding it in our favour.
    #
    # Distractor tracks are reported SEPARATELY from clutter rather than merged, because
    # they mean different things: clutter is the detector firing on nothing, a distractor
    # track is the system following a real flying animal and calling it a target.
    targets = {n for n, o in gt.objects.items() if not o.ignore}
    kinds = {"true": 0, "distractor": 0, "clutter": 0, "ambiguous": 0}
    detail = []
    for tr in raw["tracks"]:
        on_target = on_distractor = total = 0
        for f, v in tr["frames"].items():
            fi = int(f)
            if fi in excl:
                continue
            total += 1
            hit = None
            for name, obj in gt.objects.items():
                b = obj.box(fi)
                if b is None:
                    continue
                r = max(TAU, 0.5 * math.sqrt(b[2] * b[3]))
                if math.hypot(v[0] - b[0], v[1] - b[1]) <= r:
                    # A target outranks a distractor the track also overlaps, matching
                    # the convention in dronedet.metrics._match_frame.
                    if name in targets:
                        hit = "target"
                        break
                    hit = "distractor"
            on_target += hit == "target"
            on_distractor += hit == "distractor"
        if not total:
            continue
        ft, fd = on_target / total, on_distractor / total
        kind = ("true" if ft > 0.5 else
                "distractor" if fd > 0.5 else
                "clutter" if (1.0 - ft - fd) > 0.5 else "ambiguous")
        kinds[kind] += 1
        detail.append({"id": tr["id"], "kind": kind, "n_frames": total,
                       "frac_on_target": round(ft, 3),
                       "frac_on_distractor": round(fd, 3)})

    out["track_kinds"] = kinds
    out["tracks_detail"] = detail
    # Ignore-aware: a track spent on a bird is a false track. Ambiguous tracks -- no
    # single kind holding a majority -- are counted false too, because a track that
    # drifts between a bird and nothing is not a successful target track.
    out["false_tracks"] = kinds["distractor"] + kinds["clutter"] + kinds["ambiguous"]
    out["false_tracks_excluding_distractors"] = kinds["clutter"] + kinds["ambiguous"]
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="work/gt.json")
    ap.add_argument("--tracks", nargs="+", required=True)
    a = ap.parse_args()
    for p in a.tracks:
        print(json.dumps(score(a.gt, p), indent=2))
