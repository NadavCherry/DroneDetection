"""Canonicalize the user's hand labels into work/gt_user.json.

- ``far``: the user's drone track (548 continuous frames, 0..547) --
  authoritative; it corrects the auto-GT, whose second half had latched
  onto an unlabeled bird crossing the sky.
- ``near``: the landed drone (verified static box from the auto-GT),
  marked ``ignore`` per the user's intent ("no problem if the model finds
  it") -- detections there are neither TP nor FP.
- ``bird*``: the user's nine bird tracks, marked ``ignore`` for the
  drone-detection scores (a separate discrimination analysis uses them
  as negatives for 2-class models).
- frames 548..570 are excluded (unlabeled tail).
- camera shifts are copied from the auto-GT so trackers keep working.

Also prints an agreement analysis vs the auto-GT.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.gt import GroundTruth, GTObject

USER = "work/gt_labeled_all.json"
AUTO = "work/gt.json"
OUT = "work/gt_user.json"


def main() -> None:
    user = json.loads(Path(USER).read_text())
    auto = GroundTruth.load(AUTO)

    gt = GroundTruth(video=user["video"])
    gt.meta["shifts"] = auto.meta.get("shifts", {})
    gt.meta["source"] = "manual labels (gt_labeled_all.json) + near box from auto-GT"

    far = GTObject(name="far", ignore=False)
    far.frames = {int(k): tuple(v) for k, v in user["objects"]["far"]["frames"].items()}
    gt.objects["far"] = far

    last_far = max(far.frames)
    n_frames = 571
    gt.meta["exclude_frames"] = list(range(last_far + 1, n_frames))

    near = GTObject(name="near", ignore=True)
    near.frames = dict(auto.objects["near"].frames)
    gt.objects["near"] = near

    for name, o in user["objects"].items():
        if name == "far":
            continue
        obj = GTObject(name=name, ignore=True)
        obj.frames = {int(k): tuple(v) for k, v in o["frames"].items()}
        gt.objects[name] = obj

    gt.save(OUT)
    print(f"saved {OUT}: far {len(far.frames)} frames, "
          f"{sum(1 for o in gt.objects.values() if o.ignore and o.name != 'near')} bird tracks, "
          f"excluded {len(gt.meta['exclude_frames'])} tail frames")

    # agreement analysis vs auto-GT far
    afar = auto.objects["far"].frames
    excl_auto = set(auto.meta.get("exclude_frames", []))
    both = sorted(set(far.frames) & set(afar))
    seg = {"A(0..290)": [], "gap(291..335)": [], "B(336..)": []}
    for t in both:
        d = math.hypot(far.frames[t][0] - afar[t][0], far.frames[t][1] - afar[t][1])
        if t <= 290:
            seg["A(0..290)"].append(d)
        elif t <= 335:
            seg["gap(291..335)"].append(d)
        else:
            seg["B(336..)"].append(d)
    for name, ds in seg.items():
        if ds:
            ds.sort()
            print(f"auto-vs-user far distance {name}: n={len(ds)} "
                  f"median {ds[len(ds)//2]:.1f}px  p90 {ds[int(len(ds)*0.9)]:.1f}px")
    only_user = sorted(set(far.frames) - set(afar))
    print(f"frames labeled by user but absent/excluded in auto-GT: {len(only_user)} "
          f"(auto-gap coverage: {sum(1 for t in only_user if t in excl_auto or 291 <= t <= 335)})")


if __name__ == "__main__":
    main()
