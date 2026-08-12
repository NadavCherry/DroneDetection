"""The temporal-stack ablation: is the whole thesis actually true, on someone else's data?

The project's founding measurement is single-frame mAP50 0.06 -> temporal-stack 0.83,
same network, same recipe -- but that was measured on this repo's own two clips, where a
4 px black drone against a bright sky is close to invisible in one frame. A reviewer is
entitled to ask whether the gap survives on a public corpus with 11.8 px targets, and
that question has never been asked here.

Two configs, one variable. Both arms use the *same* labels, splits, tile size and
schedule, because `tools/make_temporal_combined.py` deliberately reuses
`make_dataset_external`'s parsers and splits. The only intended difference is what the
three input channels contain:

    single : B, G, R of frame t                       (work/ext_datasets/combined_tiled)
    stack  : gray(t-2dt), gray(t-dt), gray(t), dt=3   (work/ext_datasets/combined_temporal)

THREE CONFOUNDS, STATED RATHER THAN HIDDEN
------------------------------------------
1. The single-frame builder emits drone-free negative tiles (`neg_per_frame`); the
   temporal builder emits positives only. The arms therefore differ in negative supply as
   well as in representation, and a gain must not be attributed entirely to motion until
   that is equalised. It is a builder difference, not a config one, so it cannot be fixed
   from here.
2. Both arms run with photometric augmentation off. That is forced on the stack arm --
   hue/saturation jitter would remix *moments*, not colours -- and is applied to the
   single-frame arm too so that the augmentation is not itself a second variable. It does
   mildly handicap the single-frame arm relative to how it would normally be trained; say
   so when the number is reported.
3. `make_dataset_external.py --task black-paste` APPENDS its synthetic tiles into
   `combined_tiled/images/train` -- the single-frame arm's directory, which the temporal
   builder never writes to. A `black-paste` run left over from an earlier round therefore
   silently gives the control arm thousands of positives the treatment arm cannot have,
   and its labels are inflated to 12 px like everything else, so the min_side check cannot
   see it. Rebuild `combined_tiled` from scratch before running this pair, and compare the
   two manifests' `dataset.splits.train.images.n_files` before believing either number.
"""

from __future__ import annotations

from .base import NO_PHOTOMETRIC_AUG, ExperimentConfig

_SHARED = dict(
    datasets=("ardmav", "nps", "local:07_05"),
    model_cfg="yolov8s-p2.yaml",
    weights="yolov8s.pt",
    imgsz=640, batch=8, epochs=60, seed=0,
    tile_px=640, min_side=12.0,
    aug=NO_PHOTOMETRIC_AUG,
    # The corpus is three datasets pooled, one of which is this repo's own clip, so the
    # number is an internal A/B and not comparable to any paper. Centre-distance is the
    # matcher that is honest at 4 px, and no published number is being placed beside it.
    protocol_key="specklock-centre",
    tags=("ablation", "temporal"),
)

TEMPORAL_ABLATION_SINGLE = ExperimentConfig(
    name="temporal_stack_ablation_single",
    data="work/ext_datasets/combined_tiled/data.yaml",
    temporal_stack=False,
    build_command=("PYTHONPATH=. .venv/bin/python tools/make_dataset_external.py "
                   "--task combined-tiled --tile 640 --stride-train 6 --stride-val 12 "
                   "--min-side 12"),
    notes="Control arm: RGB of frame t. Same labels, splits and schedule as the stack "
          "arm. Built with --stride-train 6 to match the temporal builder's default "
          "stride, so the two arms see the same frames rather than merely the same "
          "videos.",
    **_SHARED,
)

TEMPORAL_ABLATION_STACK = ExperimentConfig(
    name="temporal_stack_ablation_stack",
    data="work/ext_datasets/combined_temporal/data.yaml",
    temporal_stack=True,
    stack_dt=3,
    build_command=("PYTHONPATH=. .venv/bin/python tools/make_temporal_combined.py "
                   "--stride-train 6 --stride-val 12"),
    notes="Treatment arm: three ego-aligned grays at t-6, t-3, t. dt=3 because that is "
          "what make_temporal_combined.py builds; the shipped PC detector uses dt=6 and "
          "dt=9 was measured and lost, so dt is a knob with a history, not a default.",
    **_SHARED,
)
