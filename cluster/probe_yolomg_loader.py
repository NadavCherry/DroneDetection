#!/usr/bin/env python3
"""Prove that labels and masks actually reach YOLOMG's training loop. Exits non-zero if not.

WHY THIS EXISTS
---------------
The 3-epoch smoke run prints `all  60  0  0  0  0` in its validation table -- 60 images,
0 labels -- which reads like a catastrophe. The dataset scan in the same log says the
opposite: `200 found, 0 missing, 0 empty, 0 corrupt` on both streams. One of those two
readings is cosmetic and the other means the competitor trains on nothing, and the
difference is six GPU-days and a false result.

Deciding by reasoning is not good enough here. This instantiates their own
`create_dataloader` and counts the targets that come out of it, which settles it.

It checks three things, each of which has a silent failure mode:
  1. targets > 0            -- labels reach the loss at all
  2. rgb.shape == mask.shape -- the two streams stay aligned through augmentation, which
                                is applied to both and could desynchronise them
  3. mask.std() > 1         -- the mask stream carries signal rather than being blank.
                               A file full of zeros is a perfectly valid JPEG, loads
                               without complaint, and silently removes the competitor's
                               entire contribution.

Run from inside third_party/YOLOMG (its imports are relative to the repo root).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="a YOLOMG data yaml (train + train2)")
    ap.add_argument("--yolomg", required=True, help="path to the YOLOMG checkout")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--batches", type=int, default=10)
    a = ap.parse_args()

    sys.path.insert(0, a.yolomg)
    import yaml
    from utils.datasets import create_dataloader

    d = yaml.safe_load(Path(a.data).read_text(encoding="utf-8"))
    hyp = yaml.safe_load(
        (Path(a.yolomg) / "data" / "hyps" / "hyp.scratch-low.yaml").read_text(
            encoding="utf-8"))

    loader, _ds = create_dataloader(
        d["train"], d["train2"], a.imgsz, a.batch, 32, hyp=hyp, augment=True,
        prefix="probe: ", prefix2="probe2: ")

    n_batch = n_tgt = 0
    mask_std = rgb_std = 0.0
    for batch in loader:
        imgs, imgs2, targets = batch[0], batch[1], batch[2]
        n_batch += 1
        n_tgt += int(targets.shape[0])
        if n_batch == 1:
            print(f"  rgb  batch {tuple(imgs.shape)}")
            print(f"  mask batch {tuple(imgs2.shape)}")
            if imgs.shape != imgs2.shape:
                print(f"FAIL: streams disagree on shape: {imgs.shape} vs {imgs2.shape}",
                      file=sys.stderr)
                return 1
            rgb_std = float(imgs.float().std())
            mask_std = float(imgs2.float().std())
            print(f"  rgb std {rgb_std:.2f}   mask std {mask_std:.2f}")
        if n_batch >= a.batches:
            break

    print(f"  {n_tgt} targets over {n_batch} batches")

    if n_tgt == 0:
        print("FAIL: no labels reach the training loop -- the competitor would train on "
              "nothing, and would still produce a model, a curve and a number",
              file=sys.stderr)
        return 1
    if mask_std < 1.0:
        print(f"FAIL: the mask stream is essentially constant (std {mask_std:.3f}). A "
              f"blank mask is a valid JPEG and loads silently, and it removes YOLOMG's "
              f"entire contribution -- we would be beating a plain YOLOv5s.",
              file=sys.stderr)
        return 1

    print("  labels reach the loss, streams aligned, mask carries signal -- OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
