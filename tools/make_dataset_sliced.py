"""Slicing-aided fine-tuning dataset (SF recipe from the research docs).

Instead of full 1280x720 frames, emit 640x640 native-resolution slices:
per frame one slice around the far drone (position jittered), one around
the near drone, and one random background slice (hard negatives: foliage,
birds, treeline). Any GT box landing inside a slice is labeled. Same time
split as the full-frame dataset; GT-gap frames dropped.

Rationale: on full frames the 6-14 px target gets almost no positive
label assignments (IoU-based assignment starves tiny boxes) -- the
full-frame fine-tune learned the near drone perfectly and the far drone
not at all, even on training frames. Halving the canvas doubles the
target's relative size and triples the number of training images.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.gt import GroundTruth
from dronedet.video import frames

VIDEO = "data/videos/07_05.mp4"
SPLIT_AT = 342
TILE = 640
MIN_BOX = 8.0
ROOT = Path("work/dataset_sliced")
rng = random.Random(20260702)


def slice_origin(cx: float, cy: float, w: int, h: int) -> tuple[int, int]:
    """Random slice origin such that (cx, cy) lands well inside the tile."""
    margin = 70
    x0 = int(cx - rng.uniform(margin, TILE - margin))
    y0 = int(cy - rng.uniform(margin, TILE - margin))
    return max(0, min(w - TILE, x0)), max(0, min(h - TILE, y0))


def labels_in(gt: GroundTruth, idx: int, x0: int, y0: int) -> list[str]:
    lines = []
    for obj in gt.objects.values():
        b = obj.box(idx)
        if b is None or obj.ignore:
            continue
        cx, cy, bw, bh = b
        if not (x0 <= cx < x0 + TILE and y0 <= cy < y0 + TILE):
            continue
        bw, bh = max(bw, MIN_BOX), max(bh, MIN_BOX)
        lines.append(f"0 {(cx - x0) / TILE:.6f} {(cy - y0) / TILE:.6f} "
                     f"{bw / TILE:.6f} {bh / TILE:.6f}")
    return lines


def main() -> None:
    gt = GroundTruth.load("work/gt.json")
    excl = set(gt.meta.get("exclude_frames", []))
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for idx, frame in frames(VIDEO):
        if idx in excl:
            continue
        h, w = frame.shape[:2]
        split = "train" if idx < SPLIT_AT else "val"
        slices = []
        far = gt.objects["far"].box(idx)
        near = gt.objects["near"].box(idx)
        if far is not None:
            slices.append(("far", *slice_origin(far[0], far[1], w, h)))
        if near is not None:
            slices.append(("near", *slice_origin(near[0], near[1], w, h)))
        slices.append(("bg", rng.randint(0, w - TILE), rng.randint(0, h - TILE)))
        for tag, x0, y0 in slices:
            name = f"f{idx:05d}_{tag}"
            crop = frame[y0:y0 + TILE, x0:x0 + TILE]
            cv2.imwrite(str(ROOT / f"images/{split}/{name}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (ROOT / f"labels/{split}/{name}.txt").write_text(
                "\n".join(labels_in(gt, idx, x0, y0)) + "\n")
            counts[split] += 1

    (ROOT / "data.yaml").write_text(
        f"path: {ROOT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: drone\n"
    )
    print(f"sliced dataset: {counts['train']} train / {counts['val']} val -> {ROOT}")


if __name__ == "__main__":
    main()
