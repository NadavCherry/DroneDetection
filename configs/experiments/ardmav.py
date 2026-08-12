"""ARD-MAV experiments: the headline pair, plus the published edge architecture.

ARD-MAV is the set with the most headroom of anything on disk (published bar: MGMD at
AP 0.55, IoU 0.25, on the official 15-video split) and the only one whose official split
this repo now honours. All three configs here train on the same 40 train / 5 val videos
and never touch the 15 test videos, so their numbers are placeable beside a published one.

A NOTE THAT WILL COST YOU A RUN IF YOU SKIP IT
----------------------------------------------
`tools/make_dataset_external.py --task ardmav-train-tiled` writes to a single hard-coded
directory, `work/ext_datasets/ardmav_yolo_tiled`, whatever `--min-side` it was given. The
baseline (min_side 12) and true-extent (min_side 0) datasets therefore *overwrite each
other*. Build one, train it, then build the other -- and rely on the fact that
`tools/train.py` samples the label files and aborts if the inflation on disk is not the
inflation the config claims. That check is the whole reason it exists.
"""

from __future__ import annotations

from .base import DEFAULT_AUG, ExperimentConfig

ARDMAV_TILED = "work/ext_datasets/ardmav_yolo_tiled/data.yaml"
_BUILD = ("PYTHONPATH=. .venv/bin/python tools/make_dataset_external.py "
          "--task ardmav-train-tiled --tile 640 --stride-train 4 --stride-val 10 "
          "--min-side {min_side}")


BASELINE_ARDMAV = ExperimentConfig(
    name="baseline_ardmav",
    datasets=("ardmav",),
    data=ARDMAV_TILED,
    model_cfg="yolov8s-p2.yaml",
    weights="yolov8s.pt",
    imgsz=640, batch=8, epochs=60, seed=0,
    tile_px=640, min_side=12.0,
    aug=DEFAULT_AUG,
    protocol_key="ardmav-official",
    build_command=_BUILD.format(min_side=12),
    tags=("ardmav", "baseline"),
    notes="The recipe rounds 5-7 actually used -- P2 head, 640 px native-scale tiles, "
          "labels inflated to a 12 px minimum side -- re-run against the OFFICIAL "
          "ARD-MAV split so that, for the first time, its number can be compared with "
          "MGMD's 0.55. This is the control for trueextent_ardmav, and the run whose "
          "COCO AP is expected to be near zero: a 12 px label on a 6x3 px drone caps the "
          "achievable IoU at ~0.13 before the detector does anything at all.")


TRUEEXTENT_ARDMAV = ExperimentConfig(
    name="trueextent_ardmav",
    datasets=("ardmav",),
    data=ARDMAV_TILED,
    model_cfg="yolov8s-p2.yaml",
    weights="yolov8s.pt",
    imgsz=640, batch=8, epochs=60, seed=0,
    tile_px=640, min_side=0.0,
    nwd=True, nwd_assign_ratio=0.5, nwd_assign_c=16.0,
    nwd_loss_ratio=0.5, nwd_loss_c=2.0,
    aug=DEFAULT_AUG,
    protocol_key="ardmav-official",
    build_command=_BUILD.format(min_side=0),
    tags=("ardmav", "headline"),
    notes="THE headline experiment. Identical to baseline_ardmav except the labels are "
          "true extents and NWD carries the assignment. The hypothesis is that the 12 px "
          "inflation was never about tiny objects being hard -- it was a prosthesis for "
          "IoU-based label assignment collapsing on few-pixel boxes, which is exactly "
          "what a Wasserstein assignment metric is for. If it holds, this run predicts "
          "true-sized boxes, and this repo reports a non-zero COCO AP for the first "
          "time. If it does not, the inflation is load-bearing and every IoU comparison "
          "in the project stays capped -- which is also worth knowing, and is why the "
          "control is run at the same three seeds.")


P2_NO_P5_ARDMAV = ExperimentConfig(
    name="p2_no_p5_ardmav",
    datasets=("ardmav",),
    data=ARDMAV_TILED,
    model_cfg="configs/yolov8s-p2-noP5.yaml",
    weights="yolov8s.pt",
    imgsz=640, batch=8, epochs=60, seed=0,
    tile_px=640, min_side=12.0,
    aug=DEFAULT_AUG,
    expected_strides=(4, 8, 16),
    protocol_key="ardmav-official",
    build_command=_BUILD.format(min_side=12),
    tags=("ardmav", "architecture"),
    notes="The Jul-2026 published edge recipe: add the stride-4 P2 head, delete the "
          "stride-32 P5 head. The P5 head costs the most parameters and, on a corpus "
          "whose median target is 11.8 px, can never fire -- its anchors are 32 px "
          "apart. Shares baseline_ardmav's dataset exactly, so the difference is the "
          "architecture and nothing else. The hand-written yaml was instantiated on "
          "2026-08-12 and measured: strides (4, 8, 16) and 7,409,459 parameters against "
          "yolov8s-p2's (4, 8, 16, 32) and 10,884,336, i.e. 31.9% fewer. It has still "
          "never been trained, which is why expected_strides stays set -- tools/train.py "
          "re-reads the head at launch and aborts before the first batch if a later edit "
          "moves it.")
