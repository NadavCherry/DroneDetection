#!/usr/bin/env python3
"""Aggregate center-distance detection metrics over a corpus of test videos.

Given a directory of GT jsons (dronedet schema) and a directory of matching
detection jsons (same basename), pool per-frame match records ACROSS videos and
report one corpus-level AP / best-F1 / precision / recall / FP-per-frame, plus a
per-video breakdown. External datasets have no far/near split, so we score
AP(all) over every non-ignore drone object.

Each video is scored only over its labeled frame range [0 .. last GT frame] so a
detector is never penalised for firing on frames the dataset left unlabeled.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dronedet.detections import DetectionSet
from dronedet.evaluate import _ap, evaluate
from dronedet.gt import GroundTruth


def labeled_range(gt: GroundTruth):
    mx = -1
    for o in gt.objects.values():
        if o.frames:
            mx = max(mx, max(o.frames))
    return (0, mx + 1) if mx >= 0 else None


def best_f1(records, n_gt):
    rs = sorted(records, key=lambda r: -r[0])
    tp = fp = 0
    best = (0.0, 0.0, 0.0, 0.0)  # f1, thresh, P, R
    for i, rr in enumerate(rs):
        tp += rr[1]
        fp += 1 - rr[1]
        if i + 1 < len(rs) and rs[i + 1][0] == rr[0]:
            continue
        p = tp / max(tp + fp, 1)
        r = tp / max(n_gt, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        if f > best[0]:
            best = (f, rr[0], p, r)
    return best


def median_err(records, thr):
    e = [r[3] for r in records if r[1] == 1 and r[0] >= thr]
    return float(np.median(e)) if e else float("nan")


def score_dir(gt_dir, det_dir, tau, min_score):
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.json")))
    pooled, total_gt, total_frames = [], 0, 0
    per_video = []
    for gp in gt_files:
        stem = Path(gp).stem
        dp = os.path.join(det_dir, stem + ".json")
        if not os.path.exists(dp):
            per_video.append((stem, None))
            continue
        gt = GroundTruth.load(gp)
        ds = DetectionSet.load(dp)
        if min_score is not None:
            ds.frames = {f: [d for d in v if d.score >= min_score]
                         for f, v in ds.frames.items()}
        ev = evaluate(gt, ds, tau, frame_range=labeled_range(gt))
        recs, n_gt = ev["records"], ev["n_gt"]
        f1, th, p, r = best_f1(recs, n_gt)
        per_video.append((stem, {
            "ap": _ap(recs, n_gt), "f1": f1, "P": p, "R": r,
            "n_gt": n_gt, "n_frames": ev["n_frames"],
            "fp_pf": sum(1 for rr in recs if rr[1] == 0 and rr[0] >= th) / max(ev["n_frames"], 1),
            "med_err": median_err(recs, th),
        }))
        pooled.extend(recs)
        total_gt += n_gt
        total_frames += ev["n_frames"]
    f1, th, p, r = best_f1(pooled, total_gt)
    agg = {
        "ap": _ap(pooled, total_gt), "f1": f1, "thr": th, "P": p, "R": r,
        "n_gt": total_gt, "n_frames": total_frames,
        "fp_pf": sum(1 for rr in pooled if rr[1] == 0 and rr[0] >= th) / max(total_frames, 1),
        "med_err": median_err(pooled, th),
    }
    return agg, per_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--det-dir", required=True)
    ap.add_argument("--tau", type=float, default=12.0)
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    agg, per_video = score_dir(a.gt_dir, a.det_dir, a.tau, a.min_score)
    tag = a.label or Path(a.det_dir).name
    print(f"\n### {tag}  (tau={a.tau}px)")
    print("| video | AP | F1 | P | R | GT | FP/frame | med-err px |")
    print("|---|---|---|---|---|---|---|---|")
    for stem, m in per_video:
        if m is None:
            print(f"| {stem} | — MISSING DETS — |")
            continue
        print(f"| {stem} | {m['ap']:.3f} | {m['f1']:.3f} | {m['P']:.3f} | {m['R']:.3f} | "
              f"{m['n_gt']} | {m['fp_pf']:.2f} | {m['med_err']:.1f} |")
    print(f"| **CORPUS** | **{agg['ap']:.3f}** | **{agg['f1']:.3f}** | {agg['P']:.3f} | "
          f"{agg['R']:.3f} | {agg['n_gt']} | {agg['fp_pf']:.2f} | {agg['med_err']:.1f} |")
    print(f"\n(best-F1 threshold {agg['thr']:.3f}; {agg['n_frames']} scored frames)")


if __name__ == "__main__":
    main()
