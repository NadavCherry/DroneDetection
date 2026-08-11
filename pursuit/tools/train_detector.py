#!/usr/bin/env python3
"""Train the chaser's detector on the Isaac Sim dataset.

A thin wrapper around ultralytics, but the recipe is not the default one and
each departure is here for a reason the closed loop cares about:

**P2 head.** The target spans 5 pixels at 90 m and the whole engagement is
decided by how early it can be seen. A stock YOLO's finest stride is 8, so a
5-pixel object lands inside a single cell; the P2 (stride-4) head is what makes
it addressable at all. Same choice, same reason, as the rest of this repository.

**Native-ish resolution.** Downscaling is free accuracy loss on an object this
small -- at ``imgsz`` 1280 a 1440-wide frame shrinks by 0.89 and a 5-pixel
target becomes 4.4. The frames are 1440x840, so 1440 is native and nothing is
thrown away.

**Heavy scale jitter, light mosaic.** The target crosses two orders of magnitude
of apparent size within a single pursuit -- 5 px at acquisition, 500 px at
impact -- so scale augmentation is the axis that matters, and the usual mosaic
(which mostly teaches context and occlusion) mainly costs resolution here.

**NWD, optionally.** IoU is a hopeless training signal at 4 px: shifting a box by
one pixel can halve it, so the gradient is dominated by quantisation. Normalized
Wasserstein Distance measures box similarity in a way that stays smooth at tiny
scales, and this repository already has an implementation (``dronedet.nwd``)
that round 7 used for the same reason.

    .venv/bin/python -m pursuit.tools.train_detector --data work/simdata/data.yaml \\
        --model yolov8s-p2.yaml --imgsz 1440 --epochs 60 --name sim-s-p2

Stop the simulator first: Isaac Sim holds two to six gigabytes of an eight
gigabyte card, and the training run will not fit around it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="work/simdata/data.yaml")
    ap.add_argument("--model", default="yolov8s-p2.yaml")
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=1440)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--name", default="sim-s-p2")
    ap.add_argument("--project", default="work/runs")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--scale", type=float, default=0.6,
                    help="scale-jitter fraction; high on purpose (see module doc)")
    ap.add_argument("--mosaic", type=float, default=0.2)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--nwd", action="store_true", help="enable the NWD blend")
    ap.add_argument("--nwd-assign-ratio", type=float, default=0.5)
    ap.add_argument("--nwd-assign-c", type=float, default=16.0)
    ap.add_argument("--nwd-loss-ratio", type=float, default=0.5)
    ap.add_argument("--nwd-loss-c", type=float, default=2.0)
    a = ap.parse_args(argv)

    if a.nwd:
        from dronedet.nwd import enable_nwd
        enable_nwd(a.nwd_assign_ratio, a.nwd_assign_c,
                   a.nwd_loss_ratio, a.nwd_loss_c)

    from ultralytics import YOLO

    model = YOLO(a.model).load(a.weights)
    results = model.train(
        data=a.data,
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=a.batch,
        name=a.name,
        project=a.project,
        patience=a.patience,
        workers=a.workers,
        device=a.device,
        # One class, and it is never confused with anything -- the loss should
        # spend its capacity on localisation, not classification.
        cls=0.3,
        box=8.0,
        scale=a.scale,
        mosaic=a.mosaic,
        close_mosaic=a.close_mosaic,
        # The sky is the background and its colour is a genuine nuisance
        # variable (nine HDRIs are in the dataset), so hue/saturation jitter is
        # wanted here -- unlike the repo's temporal-stack runs, where the three
        # channels are moments in time and remixing them is meaningless.
        hsv_h=0.02, hsv_s=0.6, hsv_v=0.4,
        degrees=0.0,        # the horizon is level; the airframe never rolls
        translate=0.15,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.0,
        plots=True,
        val=True,
    )
    # Report where the weights ACTUALLY landed, not where --project said to put
    # them. Ultralytics resolves `project` relative to the `runs_dir` in its own
    # global settings, so a relative path here can end up under somebody's
    # unrelated workspace directory -- and it also suffixes the run name when the
    # directory exists. Printing the constructed path instead of the real one
    # sends the next step looking for a file that was never written there.
    save_dir = Path(getattr(model.trainer, "save_dir", Path(a.project) / a.name))
    best = save_dir / "weights" / "best.pt"
    print(f"weights: {best}")
    if not best.exists():
        print(f"WARNING: {best} does not exist -- check ultralytics settings runs_dir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
