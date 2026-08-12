#!/usr/bin/env python3
"""Score a detector the way the closed loop consumes it: recall against pixel span.

A single mAP is the wrong summary for this system. What the pursuit needs to
know is the answer to one question -- *how small can the target get before the
detector stops finding it* -- because that number, and nothing else, sets the
range at which an engagement can begin:

    acquisition range  =  fx * 0.47 m / (smallest reliably detected span)

At this camera's ``fx`` of 922, a floor of 5 px means 87 m and a floor of 12 px
means 36 m. The sandbox sweeps the same quantity as ``min_span_px``, so the
number this tool prints plugs straight into the guidance envelope and the two
halves of the project meet on a shared axis instead of on a vibe.

The other number it prints is the **false-positive rate on background frames**,
which matters more here than precision usually does. In an offline benchmark a
false positive is a scored error; in a closed loop it is an aircraft flying at a
cloud, and the tracker will happily lock onto it.

    .venv/bin/python -m pursuit.tools.eval_detector \\
        --weights work/runs/sim-s-p2/weights/best.pt --data work/simdata --conf 0.15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Span buckets in pixels. The edges are the ranges they correspond to at this
# camera: 4 px = 108 m, 6 px = 72 m, 9 px = 48 m, 14 px = 31 m, 25 px = 17 m,
# 60 px = 7 m, beyond that the terminal phase.
BUCKETS = [(0, 4), (4, 6), (6, 9), (9, 14), (14, 25), (25, 60), (60, 150),
           (150, 10 ** 6)]


def load_labels(path: Path, w: int, h: int):
    """YOLO label file -> list of ``(cx, cy, span_px)`` in pixels."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _c, cx, cy, bw, bh = (float(v) for v in parts[:5])
        out.append((cx * w, cy * h, max(bw * w, bh * h)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="work/simdata")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=1440)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--tau-px", type=float, default=None,
                    help="centre-distance tolerance; default max(12, span)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import cv2
    from ultralytics import YOLO

    root = Path(a.data)
    images = sorted((root / "images" / a.split).glob("*.jpg"))
    if a.limit:
        images = images[:a.limit]
    if not images:
        raise SystemExit(f"no images under {root / 'images' / a.split}")
    print(f"{len(images)} {a.split} images from {root}")

    model = YOLO(a.weights, task="detect")
    stats = defaultdict(lambda: {"gt": 0, "hit": 0, "err": []})
    fp_total = 0
    bg_frames = 0
    bg_frames_with_fp = 0
    t0 = time.perf_counter()

    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gts = load_labels(root / "labels" / a.split / f"{img_path.stem}.txt", w, h)
        r = model(img, imgsz=a.imgsz, conf=a.conf, half=a.half, verbose=False,
                  max_det=16)[0]
        dets = []
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            dets.append(((x1 + x2) / 2, (y1 + y2) / 2, float(b.conf[0])))

        used = set()
        for (gx, gy, span) in gts:
            tau = a.tau_px if a.tau_px else max(12.0, span)
            key = next(f"{lo}-{hi if hi < 10 ** 5 else '+'}"
                       for lo, hi in BUCKETS if lo <= span < hi)
            stats[key]["gt"] += 1
            best, best_d = None, float("inf")
            for j, (dx, dy, _s) in enumerate(dets):
                if j in used:
                    continue
                d = np.hypot(dx - gx, dy - gy)
                if d <= tau and d < best_d:
                    best, best_d = j, d
            if best is not None:
                used.add(best)
                stats[key]["hit"] += 1
                stats[key]["err"].append(best_d)
        fp = len(dets) - len(used)
        fp_total += fp
        if not gts:
            bg_frames += 1
            bg_frames_with_fp += int(fp > 0)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(images)} "
                  f"({(i + 1) / (time.perf_counter() - t0):.1f} img/s)", flush=True)

    fps = len(images) / max(1e-6, time.perf_counter() - t0)
    print(f"\n{'span px':<12}{'n':>7}{'recall':>9}{'median err px':>16}"
          f"   ~range m")
    print("-" * 60)
    rows = {}
    for lo, hi in BUCKETS:
        key = f"{lo}-{hi if hi < 10 ** 5 else '+'}"
        s = stats.get(key)
        if not s or not s["gt"]:
            continue
        rec = s["hit"] / s["gt"]
        med = float(np.median(s["err"])) if s["err"] else float("nan")
        rng_lo = 921.8 * 0.47 / max(hi, 1e-6) if hi < 10 ** 5 else 0
        rng_hi = 921.8 * 0.47 / max(lo, 1e-6) if lo else float("inf")
        rows[key] = {"n": s["gt"], "recall": round(rec, 4),
                     "median_err_px": round(med, 2) if s["err"] else None}
        print(f"{key:<12}{s['gt']:>7}{rec:>9.3f}{med:>16.2f}"
              f"   {rng_lo:5.0f}-{rng_hi if rng_hi < 999 else 999:.0f}")

    print("-" * 60)
    print(f"false positives      {fp_total} over {len(images)} images "
          f"({fp_total / len(images):.3f} per frame)")
    if bg_frames:
        print(f"background frames    {bg_frames}, "
              f"{bg_frames_with_fp} with a false positive "
              f"({100 * bg_frames_with_fp / bg_frames:.1f}%)")
    print(f"throughput           {fps:.1f} img/s at imgsz {a.imgsz}")

    # The headline: the smallest bucket that still clears 0.9 recall is what the
    # guidance envelope can be built on.
    # A bucket needs enough samples to mean anything: the 0-4 px bin routinely
    # holds a handful of boxes, and two lucky hits out of three would otherwise
    # be reported as a detection floor of zero and an infinite engagement range.
    floor = None
    for lo, hi in BUCKETS:
        key = f"{lo}-{hi if hi < 10 ** 5 else '+'}"
        r = rows.get(key)
        if r and r["n"] >= 25 and r["recall"] >= 0.9:
            floor = lo
            break
    if floor is not None:
        print(f"\nusable detection floor ~{floor} px "
              f"-> acquisition range ~{921.8 * 0.47 / max(floor, 1e-6):.0f} m")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"weights": a.weights, "conf": a.conf, "imgsz": a.imgsz,
             "buckets": rows, "fp_total": fp_total, "images": len(images),
             "bg_frames": bg_frames, "bg_frames_with_fp": bg_frames_with_fp,
             "img_per_s": round(fps, 2), "floor_px": floor}, indent=1), encoding="utf-8")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
