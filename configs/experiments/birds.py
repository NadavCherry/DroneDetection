"""Drone-vs-bird: the claim this project cannot currently make at all.

Every detector in this repo is single-class and has never been shown a bird. "It detects
drones" is therefore an untested statement about what it does *not* fire on, and the
reviewer's first question -- "what is your bird false-alarm rate?" -- has no answer,
because the training corpus contains no negative class to have an answer about.

`uav_smid` is the fix that is available without an agreement: 13,928 stills across five
deliberately balanced classes (~3,200 each), CC BY 4.0, direct download. Halmstad adds
video-rate birds, airplanes and helicopters under CC0. Two classes, not five, because the
claim being defended is "drone vs not-drone-in-the-sky"; helicopter/aeroplane/bomb fold
into the negative class rather than diluting the positive one.

STATUS: NOT RUNNABLE. Neither dataset is on disk and neither has a builder. See `missing`.
"""

from __future__ import annotations

from .base import DEFAULT_AUG, ExperimentConfig

BIRDS_2CLASS = ExperimentConfig(
    name="birds_2class",
    datasets=("uav_smid", "halmstad"),
    data="work/ext_datasets/birds2/data.yaml",
    model_cfg="yolov8s-p2.yaml",
    weights="yolov8s.pt",
    imgsz=640, batch=12, epochs=80, seed=0,
    classes=("drone", "bird"),
    tile_px=640, min_side=0.0,
    aug=DEFAULT_AUG,
    protocol_key="ap50",
    build_command="",
    missing=(
        "tools/make_dataset_birds.py does not exist. It must: (1) download uav_smid from "
        "https://data.mendeley.com/datasets/3k3hjc7rkt/2 and remap its five classes to "
        "{drone -> 0, bird -> 1, helicopter/aeroplane/bomb -> 1 or dropped -- an "
        "unresolved decision, not an oversight}; (2) decode Halmstad's MATLAB "
        "(mcos) label files and its .xlsx manifest, which no code in this repo can read "
        "yet; (3) DEFINE AND COMMIT a train/val/test split for Halmstad, which publishes "
        "none -- without a committed split file the resulting number is not reproducible "
        "by anyone, including us. Until (3) exists this experiment must not be run, "
        "because its number would not be a benchmark result."),
    tags=("birds", "discrimination", "blocked"),
    notes="Two-class drone/bird on uav_smid plus Halmstad negatives, so that a "
          "bird-attributed false-alarm rate exists to be quoted. min_side 0: these "
          "targets are not few-pixel (stills of birds and drones at readable scale), so "
          "label inflation would only damage the localisation the class head depends on. "
          "Note the modality trap -- uav_smid is STILLS, so nothing trained here can "
          "support a temporal claim; its job is to teach the appearance of a negative.")
