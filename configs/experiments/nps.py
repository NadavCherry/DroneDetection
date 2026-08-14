"""NPS-Drones experiments: the same one-variable A/B as ARD-MAV, on a second corpus.

Why a second corpus at all
--------------------------
One benchmark is an anecdote. The ARD-MAV result -- temporal clears GLAD on the small-MAV
condition and is below it overall -- is a claim about one dataset from one lab with no
public leaderboard and no third-party reproduction. NPS is air-to-air drone video from a
different group, a different continent and a different decade, and the mechanism either
survives that or it does not.

THE SPLIT IS NOT OFFICIAL, and that has to travel with every number from here.
NPS publishes no split at all. These configs use the de-facto one every comparable paper
uses (Dogfight, TransVisDrone, YOLOMG): clips 1-36 train, 37-40 val, 41-50 test. Call it
"the Dogfight split". A number on a self-chosen split is not placeable beside a published
one the way an ARD-MAV number is, and `Protocol.split` records exactly that.

THE ANNOTATIONS ARE DOGFIGHT'S, not Purdue's
--------------------------------------------
Three annotation sets ship for NPS and no two share a coordinate convention (see
`tools/make_dataset_external.parse_nps_dogfight`). The published numbers -- TransVisDrone
0.95, GLAD 0.89 -- are computed on Dogfight's re-annotations. Ours are too. Had we used
Purdue's originals the resulting AP would have been a number no one else has ever
computed, and the comparison would have been theatre.

The bar, for the record: TransVisDrone AP@0.5 = 0.95 and YOLOMG 0.95, GLAD 0.89. High
enough that a large win is implausible; the honest question is whether the temporal stack
helps here at all, and by how much, with the same one-variable design as ARD-MAV.
"""

from __future__ import annotations

from .base import NO_PHOTOMETRIC_AUG, ExperimentConfig

NPS_TILED = "work/ext_datasets/nps_yolo_tiled/data.yaml"
NPS_TEMPORAL = "work/ext_datasets/nps_yolo_temporal/data.yaml"
_BUILD = ("PYTHONPATH=. python tools/make_dataset_external.py --task {task} "
          "--tile 640 --stride-train 4 --stride-val 10 --min-side 0 --dt 6")

#: Identical to the ARD-MAV pair in every respect that is not the dataset, so a reader can
#: attribute a difference between the two corpora to the corpora. Photometric augmentation
#: is off on BOTH arms: it is forced off on the temporal one (hue/saturation jitter remixes
#: moments rather than colours) and applied to the control too so augmentation is not a
#: second variable.
_NPS_AB = dict(
    datasets=("nps",),
    model_cfg="yolov8s-p2.yaml",
    weights="yolov8s.pt",
    imgsz=640, batch=8, epochs=30, seed=0,
    tile_px=640, min_side=0.0,
    nwd=True, nwd_assign_ratio=0.5, nwd_assign_c=16.0,
    nwd_loss_ratio=0.5, nwd_loss_c=2.0,
    aug=NO_PHOTOMETRIC_AUG,
    protocol_key="nps-official",
)

SINGLEFRAME_NPS = ExperimentConfig(
    name="singleframe_nps",
    data=NPS_TILED,
    temporal_stack=False,
    build_command=_BUILD.format(task="nps-train-tiled"),
    tags=("nps", "temporal-ab"),
    notes="Control arm: BGR of frame t. The ablation that says whether the temporal "
          "stack earns its keep on a corpus that is not ARD-MAV.",
    **_NPS_AB,
)

TEMPORAL_NPS = ExperimentConfig(
    name="temporal_nps",
    data=NPS_TEMPORAL,
    temporal_stack=True,
    stack_dt=6,
    build_command=_BUILD.format(task="nps-temporal-tiled"),
    tags=("nps", "temporal-ab", "headline"),
    notes="Treatment arm: ego-stabilised grays at t-12/t-6/t, dt=6 to match ARD-MAV and "
          "the shipped detector.\n"
          "NPS is a harder test of the mechanism than ARD-MAV in one specific way: its "
          "cameras move differently and its targets are larger, so the motion cue is "
          "worth less relative to appearance. On ARD-MAV the stack paid ~0.016 AP on the "
          "ordinary condition to buy +0.155 on the small one; if NPS is mostly 'ordinary' "
          "in that sense, the honest expected result here is a small LOSS, and that is "
          "worth publishing as readily as a win.",
    **_NPS_AB,
)
