#!/usr/bin/env python3
"""Why does a published AP differ from ours? Attribute the gap, axis by axis.

THE QUESTION THIS EXISTS TO ANSWER
----------------------------------
YOLOMG reports AP 0.95 on NPS-Drones. Scored by our evaluator, the model we trained from
their own code and recipe reaches 0.527. A 0.42 gap is far too large to wave at, and it
matters for a reason that has nothing to do with who wins: if OUR number and THEIR number
are not the same quantity, then placing our 0.487 beside their published 0.95 -- which
this project's README used to do -- is meaningless.

Rather than argue about which protocol is "right", this re-scores ONE fixed set of
detections under every combination of the protocol choices that differ, so each choice's
contribution is measured instead of asserted. The detections never change; only the rules
for counting them do.

THE AXES, AND WHY EACH IS PLAUSIBLY DIFFERENT
---------------------------------------------
frames      Ours scores EVERY frame of the test video. 26.2 % of NPS test frames contain
            no drone, and their pipeline is built from annotated frames only -- their own
            dataset scan reports "0 empty", i.e. no label file without a box. If they
            never evaluate background frames, they never pay for false positives there.

ap_style    Ours integrates the precision-recall curve over all points. YOLOv5 -- and so
            YOLOMG's val.py -- uses 101-point COCO interpolation
            (utils/metrics.py:compute_ap, method='interp'). GLAD uses 11-point. These are
            three different numbers from one curve.

matcher     Ours can match by IoU or by centre distance. Theirs is IoU only.

conf        Ours scores from 0.001 to keep the PR tail. A higher floor truncates it.

agg         Ours pools detections across sequences. Averaging per-video APs weights a
            300-frame clip like a 1,800-frame one.

    PYTHONPATH=. python tools/protocol_sweep.py \
        --gt work/ext_datasets/gt/nps --dets work/det/nps/yolomg_nps_seed0 \
        --label "YOLOMG seed 0"
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

#: numpy 2 renamed trapz -> trapezoid. YOLOMG's env pins numpy < 2 and ours does not, so
#: the reference implementation must be reachable from both without changing the result.
_trapz = getattr(np, "trapezoid", None) or np.trapz

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dronedet import metrics as M  # noqa: E402
from dronedet.console import use_utf8_stdio  # noqa: E402
from dronedet.detections import DetectionSet  # noqa: E402
from dronedet.gt import GroundTruth  # noqa: E402


def ap_from_records(scored, n_gt: int, style: str) -> float:
    """One PR curve, three published conventions for integrating it."""
    if not scored or n_gt == 0:
        return 0.0
    scored = sorted(scored, key=lambda r: -r[0])
    tp = np.cumsum([1.0 if o else 0.0 for _, o in scored])
    fp = np.cumsum([0.0 if o else 1.0 for _, o in scored])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-12)

    if style == "101pt":
        # YOLOv5 / COCO, exactly as utils/metrics.compute_ap does it: sentinels, envelope,
        # then trapezoid over 101 interpolation points.
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)
        return float(_trapz(np.interp(x, mrec, mpre), x))

    env = np.maximum.accumulate(precision[::-1])[::-1]
    if style == "11pt":
        return float(np.mean([env[recall >= t].max() if (recall >= t).any() else 0.0
                              for t in np.linspace(0, 1, 11)]))
    prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - prev) * env))


def score(gt_dir: Path, det_dir: Path, *, frames: str, matcher: str, conf: float,
          tau: float, iou: float):
    """-> {sequence: (scored_records, n_gt)} under one frame-selection rule."""
    out = {}
    for gp in sorted(gt_dir.glob("*.json")):
        dp = det_dir / gp.name
        if not dp.exists():
            continue
        gt = GroundTruth.load(gp)
        ds = DetectionSet.load(dp)

        if frames == "annotated":
            # Only frames the annotators marked with a target. This is the set a builder
            # that emits "annotated frames only" produces -- and the set whose label files
            # are never empty.
            keep = {int(f) for o in gt.objects.values() if not o.ignore for f in o.frames}
            sub = DetectionSet(video=ds.video, method=ds.method)
            for f, d in ds.frames.items():
                if int(f) in keep:
                    sub.frames[int(f)] = d
            ds = sub

        if conf > 0:
            filt = DetectionSet(video=ds.video, method=ds.method)
            for f, d in ds.frames.items():
                filt.frames[f] = [x for x in d if x.score >= conf]
            ds = filt

        ev = M.evaluate(gt, ds, rule=("iou" if matcher == "iou" else "centre"),
                        tau=tau, iou_thr=iou)
        out[gp.stem] = ([(float(r.score), r.outcome == "tp") for r in ev.records
                         if r.outcome in ("tp", "fp")], ev.n_gt)
    return out


def main() -> int:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--dets", required=True, type=Path)
    ap.add_argument("--label", default="model")
    ap.add_argument("--tau", type=float, default=12.0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    AXES = {"frames": ["all", "annotated"], "ap_style": ["all-point", "101pt", "11pt"],
            "matcher": ["iou", "centre"], "conf": [0.001, 0.25], "agg": ["pooled", "per-video"]}
    # Our published configuration, and the baseline every delta is measured against.
    OURS = {"frames": "all", "ap_style": "all-point", "matcher": "iou",
            "conf": 0.001, "agg": "pooled"}
    # Their published configuration, as read from YOLOMG's val.py and dataset build.
    THEIRS = {"frames": "annotated", "ap_style": "101pt", "matcher": "iou",
              "conf": 0.001, "agg": "pooled"}

    cache: dict = {}

    def ap_for(cfg) -> float:
        key = (cfg["frames"], cfg["matcher"], cfg["conf"])
        if key not in cache:
            cache[key] = score(a.gt, a.dets, frames=cfg["frames"], matcher=cfg["matcher"],
                               conf=cfg["conf"], tau=a.tau, iou=a.iou)
        per_seq = cache[key]
        if cfg["agg"] == "per-video":
            vals = [ap_from_records(rec, n, cfg["ap_style"]) for rec, n in per_seq.values()]
            return float(np.mean(vals)) if vals else 0.0
        recs = [r for rec, _ in per_seq.values() for r in rec]
        n_gt = sum(n for _, n in per_seq.values())
        return ap_from_records(recs, n_gt, cfg["ap_style"])

    L = [f"# Protocol sweep -- {a.label}", "",
         "One fixed set of detections, re-scored under each protocol choice. Nothing about "
         "the model changes between rows; only the counting rules do.", ""]

    base = ap_for(OURS)
    theirs = ap_for(THEIRS)
    L += ["| configuration | AP |", "|---|---|",
          f"| **our evaluator** ({', '.join(f'{k}={v}' for k, v in OURS.items())}) | **{base:.4f}** |",
          f"| **their published protocol** ({', '.join(f'{k}={v}' for k, v in THEIRS.items())}) | **{theirs:.4f}** |",
          "", f"Total gap attributable to protocol: **{theirs - base:+.4f}**", ""]

    # One axis at a time, from OUR configuration: the marginal effect of each choice.
    L += ["## One axis at a time, moving from our protocol toward theirs", "",
          "| axis | our value | their value | AP | delta |", "|---|---|---|---|---|"]
    for axis, their_val in THEIRS.items():
        if OURS[axis] == their_val:
            continue
        cfg = dict(OURS)
        cfg[axis] = their_val
        v = ap_for(cfg)
        L.append(f"| {axis} | {OURS[axis]} | {their_val} | {v:.4f} | {v - base:+.4f} |")
    L.append("")

    # Every combination, so interactions between axes are visible rather than assumed
    # additive -- the frame set and the AP style interact strongly, because dropping
    # background frames removes exactly the low-confidence tail the interpolation weights.
    L += ["## Full grid", "",
          "| frames | ap_style | matcher | conf | agg | AP |", "|---|---|---|---|---|---|"]
    keys = list(AXES)
    for combo in itertools.product(*(AXES[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        L.append("| " + " | ".join(str(cfg[k]) for k in keys)
                 + f" | {ap_for(cfg):.4f} |")

    out = "\n".join(L) + "\n"
    print(out)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
