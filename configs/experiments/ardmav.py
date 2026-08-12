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

from .base import DEFAULT_AUG, NO_PHOTOMETRIC_AUG, ExperimentConfig

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


_LOCAL_BUILD = ("python tools/make_datasets_v3.py --split-at 548 --only crop640 "
                "--label-px {label_px} --suffix {suffix}")

BASELINE_LOCAL = ExperimentConfig(
    name="baseline_local",
    datasets=("local:07_05",),
    data="work/dsv3_crop640_lbl24/data.yaml",
    model_cfg="yolov8s-p2.yaml",
    weights="yolov8s.pt",
    # TWO classes, not one. make_datasets_v3.py pastes a bird bank as an explicit
    # class 1 (2,614 drone + 3,144 bird boxes in train), which is how the shipped
    # detector learns to reject birds rather than merely not fire on them. Left at
    # the ExperimentConfig default of ("drone",) these configs aborted in
    # tools/train.py's data check before the first batch -- which is the check
    # doing its job, since a 1-class head on 2-class labels trains silently wrong.
    classes=("drone", "bird"),
    imgsz=640, batch=8, epochs=60, seed=0,
    tile_px=640, min_side=24.0,
    nwd=False,
    aug=DEFAULT_AUG,
    protocol_key="specklock-centre",
    build_command=_LOCAL_BUILD.format(label_px=24, suffix="_lbl24"),
    tags=("local", "headline", "extent"),
    notes="The shipped round-3 recipe on this project's own 07_05 data, with its "
          "LABEL = 24.0 px fixed-square labels. This is the CONTROL for "
          "trueextent_local and the run that demonstrates the defect rather than "
          "fixing it: measured against 07_05's real annotations, a 24 px square caps "
          "the achievable IoU at a median of 0.110, so 0% of boxes can reach IoU 0.5 "
          "and this run's COCO AP is arithmetically bound to be 0.000 however good the "
          "detector is. Run it anyway -- a defect you can only assert is weaker than "
          "one you can show.")


TRUEEXTENT_LOCAL = ExperimentConfig(
    name="trueextent_local",
    datasets=("local:07_05",),
    data="work/dsv3_crop640_lbl0/data.yaml",
    model_cfg="yolov8s-p2.yaml",
    weights="yolov8s.pt",
    classes=("drone", "bird"),          # see baseline_local -- same 2-class labels
    imgsz=640, batch=8, epochs=60, seed=0,
    tile_px=640, min_side=0.0,
    nwd=True, nwd_assign_ratio=0.5, nwd_assign_c=16.0,
    nwd_loss_ratio=0.5, nwd_loss_c=2.0,
    aug=DEFAULT_AUG,
    protocol_key="specklock-centre",
    build_command=_LOCAL_BUILD.format(label_px=0, suffix="_lbl0"),
    tags=("local", "headline", "extent"),
    notes="THE experiment that fixes the fatal case. trueextent_ardmav addresses "
          "ARD-MAV, where inflation is tolerable (min_side 12 still leaves 74% of boxes "
          "able to reach IoU 0.5). It is THIS repo's OWN data at LABEL = 24 that is "
          "arithmetically hopeless at 0%, and only this config touches it. Report both "
          "centre-distance AP (the protocol that is honest at 4 px) and COCO AP (the "
          "one that makes the number comparable), against baseline_local at the same "
          "three seeds. Note the 10_06 ground truth is itself a constant 8.0 px, so a "
          "COCO number on the test video is capped by the LABELS as well as the model "
          "until that GT is re-derived with true extents -- score the val split first.")


# ---------------------------------------------------------------------------------
# The temporal A/B on ARD-MAV: does the founding claim survive on someone else's data?
#
# The three configs above train on SINGLE RGB FRAMES. That is not a criticism of them --
# they answer a label-geometry question and answer it well -- but it does mean that a
# number from `ardmav_headline` describes a plain single-frame YOLOv8s-p2, and this
# project's contribution is a stabilised temporal stack. Publishing the first against
# MGMD's 0.55 would compare the wrong method, and the sign of the result would not tell
# you which method was wrong.
#
# `temporal_stack_ablation` already asks this question, but on a POOLED corpus
# (ardmav + nps + local:07_05) at dt=3 with a self-chosen split, so its number is an
# internal A/B that cannot sit beside a published one. This pair asks it on ARD-MAV
# alone, at dt=6, on the OFFICIAL 15-video split, so the winner is directly placeable
# against MGMD.
#
# ONE VARIABLE. Both arms share labels, splits, tiles, stride, schedule, seed, NWD and
# augmentation; only the three input channels differ:
#
#     singleframe : B, G, R of frame t
#     temporal    : gray(t-12), gray(t-6), gray(t), ego-stabilised, dt=6
#
# Photometric augmentation is off in BOTH arms. It is forced off on the temporal arm --
# hue/saturation jitter would remix moments rather than colours -- and applied to the
# single-frame arm too so augmentation is not a second variable. That mildly handicaps
# the single-frame arm against how it would normally be trained; say so when reporting.
#
# The negative-supply confound that `temporal_stack_ablation` had to declare and could
# not fix does NOT apply here: `extract_yolo_tiled_temporal` emits the same drone-free
# negative tiles per frame as `extract_yolo_tiled`, so both arms see the same ratio of
# empty sky to target. That was the point of building it on the shared tiling core.
#
# HAZARD, same as the note at the top of this file: both arms build into their own
# directory, but `--min-side 0` here and `--min-side 12` for the baseline pair still
# overwrite each other within `ardmav_yolo_tiled`. Build, train, then rebuild.
# ---------------------------------------------------------------------------------

ARDMAV_TEMPORAL = "work/ext_datasets/ardmav_yolo_temporal/data.yaml"
_TEMPORAL_BUILD = ("PYTHONPATH=. .venv/bin/python tools/make_dataset_external.py "
                   "--task ardmav-temporal-tiled --tile 640 --stride-train 4 "
                   "--stride-val 10 --min-side 0 --dt 6")

_ARDMAV_AB = dict(
    datasets=("ardmav",),
    model_cfg="yolov8s-p2.yaml",
    weights="yolov8s.pt",
    imgsz=640, batch=8, epochs=60, seed=0,
    tile_px=640, min_side=0.0,
    nwd=True, nwd_assign_ratio=0.5, nwd_assign_c=16.0,
    nwd_loss_ratio=0.5, nwd_loss_c=2.0,
    aug=NO_PHOTOMETRIC_AUG,
    protocol_key="ardmav-official",
)

SINGLEFRAME_ARDMAV = ExperimentConfig(
    name="singleframe_ardmav",
    data=ARDMAV_TILED,
    temporal_stack=False,
    build_command=_BUILD.format(min_side=0),
    tags=("ardmav", "temporal-ab"),
    notes="Control arm. Identical to trueextent_ardmav except that photometric "
          "augmentation is off, so that the only difference from temporal_ardmav is "
          "what the three channels contain. Not redundant with trueextent_ardmav: that "
          "config answers 'do true extents unblock COCO AP', this one answers 'what "
          "does a single frame score when the temporal arm is held to the same "
          "augmentation'. Reporting the temporal gain against a DEFAULT_AUG control "
          "would credit motion for an augmentation difference.",
    **_ARDMAV_AB,
)

TEMPORAL_ARDMAV = ExperimentConfig(
    name="temporal_ardmav",
    data=ARDMAV_TEMPORAL,
    temporal_stack=True,
    stack_dt=6,
    build_command=_TEMPORAL_BUILD,
    tags=("ardmav", "temporal-ab", "headline"),
    notes="THE experiment that puts this project's actual method on a public benchmark "
          "with a published bar. Ego-stabilised grays at t-12/t-6/t, dt=6 to match the "
          "shipped PC detector and the ablation in work/ablation/REPORT.md -- dt=3 is "
          "what temporal_stack_ablation used and dt=9 was measured and lost, so dt is a "
          "knob with a history rather than a default.\n"
          "ARD-MAV is where the mechanism should be worth most and where it is least "
          "guaranteed: 60 videos, a MOVING camera, median target 11.8 px. Moving camera "
          "cuts both ways -- it is why stabilisation is necessary and why it can fail, "
          "so report the stabiliser's inlier/response rate beside the AP. If the gain "
          "does not survive here it is a real negative result about the founding claim, "
          "and worth more than another number on our own two clips.",
    **_ARDMAV_AB,
)
