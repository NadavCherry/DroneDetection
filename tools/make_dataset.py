"""Build a YOLO dataset from 07_05.mp4 + work/gt.json.

Time-based split (no leakage): train = frames <= 341, val = frames >= 342.
The val segment is the hardest part of the flight (3-5 px cruise).
GT-gap frames (uncertain far-drone position) are dropped entirely --
leaving the drone unlabeled would teach the model it is background.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.gt import GroundTruth
from dronedet.video import frames

VIDEO = "07_05.mp4"
SPLIT_AT = 342
MIN_BOX = 8.0  # pad tiny GT boxes up to this size for regression stability
ROOT = Path("work/dataset")


def main() -> None:
    gt = GroundTruth.load("work/gt.json")
    excl = set(gt.meta.get("exclude_frames", []))
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    n_tr = n_va = 0
    for idx, frame in frames(VIDEO):
        if idx in excl:
            continue
        h, w = frame.shape[:2]
        lines = []
        for obj in gt.objects.values():
            b = obj.box(idx)
            if b is None or obj.ignore:
                continue
            cx, cy, bw, bh = b
            bw, bh = max(bw, MIN_BOX), max(bh, MIN_BOX)
            lines.append(f"0 {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        split = "train" if idx < SPLIT_AT else "val"
        name = f"f{idx:05d}"
        cv2.imwrite(str(ROOT / f"images/{split}/{name}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        (ROOT / f"labels/{split}/{name}.txt").write_text("\n".join(lines) + "\n")
        if split == "train":
            n_tr += 1
        else:
            n_va += 1

    (ROOT / "data.yaml").write_text(
        f"path: {ROOT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: drone\n"
    )
    print(f"dataset: {n_tr} train / {n_va} val images -> {ROOT}")


if __name__ == "__main__":
    main()
