"""Center-distance evaluation for tiny-target detection.

IoU is unstable for few-pixel boxes (a 1-2 px shift zeroes it), so a
detection matches a GT object when its center lies within a per-object
radius: ``max(tau, 0.5 * sqrt(gt_area))``. Frames listed in
``gt.meta.exclude_frames`` are skipped entirely (uncertain GT).
Detections matching ``ignore`` objects are neither TP nor FP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .detections import DetectionSet
from .gt import GroundTruth


@dataclass
class MethodReport:
    method: str
    ap_all: float
    ap_far: float
    best_f1: float
    best_thresh: float
    precision: float
    recall: float
    recall_per_object: dict[str, float]
    fp_per_frame: float
    med_err_far: float
    fps: float
    n_dets: int


def _match_frame(dets, gts, tau):
    """Greedy score-ordered matching. Returns per-det assignment:
    ('tp', obj_name, err) / ('fp',) / ('ign',), and set of matched objects."""
    assigned = {}
    used = set()
    for i, d in enumerate(sorted(range(len(dets)), key=lambda j: -dets[j].score)):
        det = dets[d]
        best, bestdist = None, 1e18
        for name, (cx, cy, w, h, ignore) in gts.items():
            if name in used:
                continue
            r = max(tau, 0.5 * math.sqrt(max(w * h, 1.0)))
            dist = math.hypot(det.cx - cx, det.cy - cy)
            if dist <= r and dist < bestdist:
                best, bestdist = name, dist
        if best is None:
            assigned[d] = ("fp",)
        elif gts[best][4]:
            assigned[d] = ("ign",)  # matched an ignore object; don't reuse it
        else:
            assigned[d] = ("tp", best, bestdist)
            used.add(best)
    return assigned


def evaluate(gt: GroundTruth, ds: DetectionSet, tau: float = 12.0,
             objects: set[str] | None = None,
             frame_range: tuple[int, int] | None = None) -> dict:
    """Score all detections; returns arrays for PR computation."""
    excl = set(gt.meta.get("exclude_frames", []))
    frames = sorted(set(ds.frames) - excl)
    if frame_range is not None:
        frames = [f for f in frames if frame_range[0] <= f < frame_range[1]]
    records = []  # (score, is_tp, obj, err)
    n_gt = 0
    gt_frames_per_obj: dict[str, int] = {}
    for f in frames:
        gts = {}
        for name, obj in gt.objects.items():
            box = obj.box(f)
            if box is None:
                continue
            treat_ignore = obj.ignore or (objects is not None and name not in objects)
            gts[name] = (*box, treat_ignore)
            if not treat_ignore:
                n_gt += 1
                gt_frames_per_obj[name] = gt_frames_per_obj.get(name, 0) + 1
        assigned = _match_frame(ds.frames[f], gts, tau)
        for d, a in assigned.items():
            det = ds.frames[f][d]
            if a[0] == "tp":
                records.append((det.score, 1, a[1], a[2], f))
            elif a[0] == "fp":
                records.append((det.score, 0, None, None, f))
    return {"records": records, "n_gt": n_gt, "n_frames": len(frames),
            "gt_frames_per_obj": gt_frames_per_obj}


def _ap(records, n_gt):
    if not records or n_gt == 0:
        return 0.0
    rs = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in rs])
    fp = np.cumsum([1 - r[1] for r in rs])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    # standard continuous AP with precision envelope
    ap, best_p = 0.0, 0.0
    prev_r = 0.0
    for i in range(len(rs) - 1, -1, -1):
        best_p = max(best_p, prec[i])
        prec[i] = best_p
    for i in range(len(rs)):
        ap += (rec[i] - prev_r) * prec[i]
        prev_r = rec[i]
    return float(ap)


def report(gt: GroundTruth, ds: DetectionSet, tau: float = 12.0,
           frame_range: tuple[int, int] | None = None) -> MethodReport:
    ev = evaluate(gt, ds, tau, frame_range=frame_range)
    ev_far = evaluate(gt, ds, tau, objects={"far"}, frame_range=frame_range)
    recs, n_gt = ev["records"], ev["n_gt"]

    # best-F1 operating point: single cumulative pass over score-sorted records
    rs = sorted(recs, key=lambda rr: -rr[0])
    best = (0.0, 0.0, 0.0, 0.0)  # f1, thresh, P, R
    tp_c = fp_c = 0
    for i, rr in enumerate(rs):
        tp_c += rr[1]
        fp_c += 1 - rr[1]
        if i + 1 < len(rs) and rs[i + 1][0] == rr[0]:
            continue  # apply thresholds only at score-group boundaries
        pp = tp_c / max(tp_c + fp_c, 1)
        rr_ = tp_c / max(n_gt, 1)
        ff = 2 * pp * rr_ / max(pp + rr_, 1e-9)
        if ff > best[0]:
            best = (ff, rr[0], pp, rr_)
    f1, th, p, r = best

    per_obj = {}
    for name, cnt in ev["gt_frames_per_obj"].items():
        got = sum(1 for rr in recs if rr[1] == 1 and rr[2] == name and rr[0] >= th)
        per_obj[name] = got / max(cnt, 1)
    fp_pf = sum(1 for rr in recs if rr[1] == 0 and rr[0] >= th) / max(ev["n_frames"], 1)
    errs = [rr[3] for rr in recs if rr[1] == 1 and rr[2] == "far" and rr[0] >= th]
    med_err = float(np.median(errs)) if errs else float("nan")

    return MethodReport(
        method=ds.method,
        ap_all=_ap(recs, n_gt),
        ap_far=_ap(ev_far["records"], ev_far["n_gt"]),
        best_f1=f1, best_thresh=th, precision=p, recall=r,
        recall_per_object=per_obj, fp_per_frame=fp_pf, med_err_far=med_err,
        fps=float(ds.meta.get("fps_end_to_end", float("nan"))),
        n_dets=sum(len(v) for v in ds.frames.values()),
    )


def evaluate_files(gt_path: str, det_paths: list[str], tau: float = 12.0,
                   min_score: float | None = None,
                   frame_range: tuple[int, int] | None = None) -> str:
    gt = GroundTruth.load(gt_path)
    rows = []
    for p in det_paths:
        ds = DetectionSet.load(p)
        if min_score is not None:
            ds.frames = {f: [d for d in v if d.score >= min_score]
                         for f, v in ds.frames.items()}
        rows.append(report(gt, ds, tau, frame_range=frame_range))
    rows.sort(key=lambda r: -r.ap_far)

    def fmt(r: MethodReport) -> str:
        ro = r.recall_per_object
        return (f"| {r.method} | {r.ap_far:.3f} | {r.ap_all:.3f} | {r.best_f1:.3f} | "
                f"{ro.get('far', 0):.3f} | {ro.get('near', 0):.3f} | "
                f"{r.precision:.3f} | {r.fp_per_frame:.2f} | "
                f"{r.med_err_far:.1f} | {r.fps:.1f} |")

    scope = (f", frames {frame_range[0]}..{frame_range[1] - 1}"
             if frame_range else "")
    lines = [
        f"# Detection method comparison (tau={tau}px, center-distance matching{scope})",
        "",
        "| method | AP(far) | AP(all) | best-F1 | R(far) | R(near) | P | FP/frame | med-err(far) px | fps |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *[fmt(r) for r in rows],
        "",
        "AP(far): average precision on the tiny moving drone only (near drone as ignore).",
        "R/P/FP at the best-F1 threshold of each method. fps is end-to-end single-thread.",
    ]
    return "\n".join(lines)
