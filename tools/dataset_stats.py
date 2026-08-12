#!/usr/bin/env python3
"""Characterise a dataset's target-size distribution before training on it.

The repo's own survey opens with "Stage 0 — characterize your data first", and then no
tool ever did it. This is that tool. It answers the three questions that decide whether a
number computed on a dataset can be compared with a published one:

1. **How big are the targets, really?** Reported as AI-TOD size bins on sqrt(w*h)
   (very-tiny 2-8 px, tiny 8-16, small 16-32, medium 32+), because an unbinned AP hides
   where a method fails.
2. **How much does label inflation distort them?** Both `make_dataset_external.py`
   (``--min-side``, default 12) and `make_datasets_v3.py` (``LABEL = 24.0``) enlarge small
   boxes so that IoU-based label assignment stays stable. That is defensible for training
   and fatal for evaluation: a detector trained on inflated labels predicts inflated
   boxes, and an inflated box cannot reach the IoU threshold of the true annotation. This
   prints, per candidate `min_side`, the fraction of boxes affected and the **best IoU
   still achievable** against the true annotation.
3. **What IoU threshold is even reachable?** If the answer is below the benchmark's
   threshold (ARD-MAV/MGMD use 0.25, most papers use 0.5), the comparison is impossible
   *before any model is trained*, and the inflation has to go.

    python tools/dataset_stats.py --dataset ardmav
    python tools/dataset_stats.py --gt work/gt_user.json --name 07_05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dronedet.metrics import SIZE_BINS, size_bin  # noqa: E402


def _iou_after_inflation(w: float, h: float, min_side: float) -> float:
    """Best IoU an inflated, perfectly-centred prediction can reach against the true box.

    Inflation only grows a side, and a concentric grown box fully contains the original,
    so intersection = true area and union = inflated area.
    """
    W, H = max(w, min_side), max(h, min_side)
    return (w * h) / (W * H) if W * H > 0 else 0.0


def summarise(sides: np.ndarray, wh: np.ndarray, name: str,
              thresholds=(8.0, 12.0, 16.0, 24.0)) -> None:
    n = len(sides)
    print(f"\n{'=' * 78}\n{name}: {n:,} boxes\n{'=' * 78}")
    print(f"  sqrt(area): min {sides.min():.1f}  p25 {np.percentile(sides, 25):.1f}  "
          f"median {np.median(sides):.1f}  p75 {np.percentile(sides, 75):.1f}  "
          f"max {sides.max():.1f} px")
    print(f"  width:  median {np.median(wh[:, 0]):.1f}  min {wh[:, 0].min():.1f}")
    print(f"  height: median {np.median(wh[:, 1]):.1f}  min {wh[:, 1].min():.1f}")

    print("\n  AI-TOD size bins:")
    for bname, _, _ in SIZE_BINS:
        c = sum(1 for s in sides if size_bin(s, s) == bname)
        if c:
            print(f"    {bname:11s} {c:8,}  {100 * c / n:5.1f} %")

    print(f"\n  Label inflation — what it costs at evaluation time:")
    print(f"    {'min_side':>9s} {'boxes grown':>12s} {'median best IoU':>16s} "
          f"{'% that can reach IoU 0.5':>25s} {'IoU 0.25':>10s}")
    for ms in thresholds:
        ious = np.array([_iou_after_inflation(w, h, ms) for w, h in wh])
        grown = float(np.mean((wh[:, 0] < ms) | (wh[:, 1] < ms)))
        print(f"    {ms:9.0f} {100 * grown:11.1f}% {np.median(ious):16.3f} "
              f"{100 * np.mean(ious >= 0.5):24.1f}% {100 * np.mean(ious >= 0.25):9.1f}%")
    print("\n    'best IoU' is the ceiling for a perfectly-centred prediction; a real one is worse.")


def load_ardmav(root: Path):
    import xml.etree.ElementTree as ET
    ann = root / "Annotations"
    if not ann.is_dir():
        sys.exit(f"no Annotations/ under {root}")
    wh, per_video = [], {}
    for vid_dir in sorted(p for p in ann.iterdir() if p.is_dir()):
        vid_boxes = []
        for xf in sorted(vid_dir.glob("*.xml")):
            try:
                root_el = ET.parse(xf).getroot()
            except ET.ParseError:
                continue
            for o in root_el.findall("object"):
                b = o.find("bndbox")
                if b is None:
                    continue
                x1, y1, x2, y2 = (float(b.find(t).text)
                                  for t in ("xmin", "ymin", "xmax", "ymax"))
                vid_boxes.append((abs(x2 - x1), abs(y2 - y1)))
        if vid_boxes:
            per_video[vid_dir.name] = vid_boxes
            wh.extend(vid_boxes)
    return np.asarray(wh, dtype=float), per_video


def load_gt_json(path: Path):
    d = json.loads(path.read_text())
    wh = []
    for name, o in d["objects"].items():
        if o.get("ignore"):
            continue
        for box in o["frames"].values():
            wh.append((box[2], box[3]))       # GT is (cx, cy, w, h)
    return np.asarray(wh, dtype=float), {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["ardmav"], help="a known external dataset")
    ap.add_argument("--root", type=Path,
                    default=REPO / "data/external/ard_mav/ARD-MAV")
    ap.add_argument("--gt", type=Path, help="a dronedet GT json instead")
    ap.add_argument("--name", default=None)
    ap.add_argument("--per-video", action="store_true",
                    help="also print the size distribution of each video")
    a = ap.parse_args()

    if a.gt:
        wh, per_video = load_gt_json(a.gt)
        name = a.name or a.gt.name
    elif a.dataset == "ardmav":
        wh, per_video = load_ardmav(a.root)
        name = a.name or "ARD-MAV"
    else:
        ap.error("pass --dataset or --gt")

    if not len(wh):
        sys.exit("no boxes found")
    sides = np.sqrt(wh[:, 0] * wh[:, 1])
    summarise(sides, wh, name)

    if per_video:
        print(f"\n  Per-video median sqrt(area), {len(per_video)} videos:")
        rows = sorted(((v, float(np.median([np.sqrt(w * h) for w, h in b])), len(b))
                       for v, b in per_video.items()), key=lambda r: r[1])
        for v, med, n in rows:
            print(f"    {v:14s} {med:6.1f} px  ({n:,} boxes)")


if __name__ == "__main__":
    main()
