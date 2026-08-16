"""The project's own two videos: train on 07_05, test on 10_06.

WHY THIS EXPERIMENT EXISTS ALONGSIDE TWO PUBLIC BENCHMARKS
----------------------------------------------------------
Neither benchmark is this task. Measured from the annotations:

    07_05  drone   3.7 - 15.3 px, median 8.0     548 annotated frames
    07_05  birds   2.9 - 11.7 px, median 6.0     8 objects, 934 frames
    10_06  drone   4.3 - 11.1 px, median 8.0     250 annotated frames
    ARD-MAV        median 11.8 px
    NPS            10 - 25 px

So the target here is smaller than ARD-MAV's and much smaller than NPS's, and -- the part
no public benchmark reproduces -- 07_05's EIGHT labelled birds sit at a median of 6.0 px
against a drone at 8.0. The distractors overlap the target in size. A method can score well
on ARD-MAV by being a good small-object detector; it cannot score well here without
discriminating, which is why these configs carry `classes=("drone", "bird")`.

WHAT A RESULT HERE IS AND IS NOT
--------------------------------
Two videos are the entire corpus, so this is a ONE-flight-to-one-flight generalisation
test: 548 training frames, a single held-out flight. That is a real test -- different
scene, different day, nothing shared -- and a very small one.

It cannot carry a cross-sequence bootstrap, because there is one test sequence. The
significance reported for it is a MOVING-BLOCK bootstrap over contiguous 30-frame blocks
of 10_06, which answers "is this difference stable across the flight's segments" and NOT
"does it generalise across flights". Those are different claims and the table says which
one it is showing.

THE 10_06 GROUND TRUTH, AND WHY v1
-----------------------------------
`realtime/work/gt_1006_v2.json` has more annotated frames (337 vs 250) and every box in it
is exactly 8.0 px -- a constant, i.e. fixed-size labels rather than measured extents.
Scoring IoU against constant labels measures the labels. `gt_1006.json` varies 4.3-11.1 px
and is what these configs score against. Primary metric is centre-distance
(`specklock-centre`), because at 8 px an IoU threshold is dominated by annotation noise;
IoU is reported as a secondary column, not suppressed.
"""

from __future__ import annotations

from .base import NO_PHOTOMETRIC_AUG, ExperimentConfig

LOCAL_TILED = "work/ext_datasets/local_yolo_tiled/data.yaml"
LOCAL_TEMPORAL = "work/ext_datasets/local_yolo_temporal/data.yaml"
_BUILD = ("PYTHONPATH=. python tools/make_dataset_external.py --task {task} "
          "--tile 640 --stride-train 1 --stride-val 4 --min-side 0 --dt 6")

#: Identical to the ARD-MAV and NPS A/B pairs in everything but the corpus, so a
#: difference between the three is attributable to the data. stride-train 1 because 548
#: frames is already a small training set and there is nothing to thin.
_LOCAL_AB = dict(
    datasets=("local:07_05",),
    model_cfg="yolov8s-p2.yaml",
    weights="yolov8s.pt",
    imgsz=640, batch=8, epochs=60, seed=0,
    tile_px=640, min_side=0.0,
    nwd=True, nwd_assign_ratio=0.5, nwd_assign_c=16.0,
    nwd_loss_ratio=0.5, nwd_loss_c=2.0,
    aug=NO_PHOTOMETRIC_AUG,
    protocol_key="specklock-centre",
)

SINGLEFRAME_LOCAL_AB = ExperimentConfig(
    name="singleframe_local_ab",
    data=LOCAL_TILED,
    temporal_stack=False,
    build_command=_BUILD.format(task="local-tiled"),
    tags=("local", "temporal-ab"),
    notes="Control arm on the project's own task: BGR of frame t. 60 epochs rather than "
          "ARD-MAV's 30 because the training set is ~40x smaller, so the same number of "
          "epochs would be a fraction of the gradient steps -- matching the epoch count "
          "across corpora of different size would be matching the wrong thing.",
    **_LOCAL_AB,
)

TEMPORAL_LOCAL_AB = ExperimentConfig(
    name="temporal_local_ab",
    data=LOCAL_TEMPORAL,
    temporal_stack=True,
    stack_dt=6,
    build_command=_BUILD.format(task="local-temporal"),
    tags=("local", "temporal-ab", "headline"),
    notes="Treatment arm: ego-stabilised grays at t-12/t-6/t. If the temporal stack is "
          "worth anything, this is where it should show most -- an 8 px drone against 6 px "
          "birds is the case where appearance carries least information and motion carries "
          "most. ARD-MAV gave +0.032 (not significant) and NPS gave -0.014; a corpus where "
          "the mechanism is supposed to matter is the honest place to look next.",
    **_LOCAL_AB,
)
