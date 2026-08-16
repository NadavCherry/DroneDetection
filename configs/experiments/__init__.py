"""The experiment registry: every training run this project intends to be able to re-run.

Lookup is by name, and `resolve()` also accepts a GROUP name, because an ablation is not
one run -- it is a pair that only means something when both arms exist at the same seeds.
`tools/train.py --config temporal_stack_ablation` therefore runs two experiments, and
`--config trueextent_ardmav` runs one.

Adding an experiment means adding a dataclass here, not adding flags to a shell command.
Anything the registry rejects at import time (see `ExperimentConfig.validate`) fails in
CI, which is the point: the config is checked before the GPU is booked.
"""

from __future__ import annotations

from .ardmav import (BASELINE_ARDMAV, BASELINE_LOCAL, P2_NO_P5_ARDMAV,
                     SINGLEFRAME_ARDMAV, TEMPORAL_ARDMAV,
                     TRUEEXTENT_ARDMAV, TRUEEXTENT_LOCAL)
from .base import (Augmentation, DEFAULT_AUG, ExperimentConfig, NO_PHOTOMETRIC_AUG,
                   UAV_DETR_B4_640, VRAM_SAFETY_FRACTION, VramReference,
                   YOLOV8S_P2_ESTIMATE, YOLOV8S_P2_MEASURED, check_strides)
from .birds import BIRDS_2CLASS
from .local_video import (SINGLEFRAME_LOCAL_AB,
                          TEMPORAL_LOCAL_AB)
from .nps import SINGLEFRAME_NPS, TEMPORAL_NPS
from .temporal import TEMPORAL_ABLATION_SINGLE, TEMPORAL_ABLATION_STACK

EXPERIMENTS: dict[str, ExperimentConfig] = {c.name: c for c in (
    BASELINE_ARDMAV,
    TRUEEXTENT_ARDMAV,
    P2_NO_P5_ARDMAV,
    BIRDS_2CLASS,
    TEMPORAL_ABLATION_SINGLE,
    TEMPORAL_ABLATION_STACK,
    BASELINE_LOCAL,
    TRUEEXTENT_LOCAL,
    SINGLEFRAME_ARDMAV,
    TEMPORAL_ARDMAV,
    SINGLEFRAME_NPS,
    TEMPORAL_NPS,
    SINGLEFRAME_LOCAL_AB,
    TEMPORAL_LOCAL_AB,
)}

#: Named sets that must be run together, and the reason they must.
GROUPS: dict[str, tuple[str, ...]] = {
    # An ablation arm on its own is not a result.
    "temporal_stack_ablation": ("temporal_stack_ablation_single",
                                "temporal_stack_ablation_stack"),
    # The true-extent claim is a difference; without the control it is an anecdote.
    "ardmav_headline": ("baseline_ardmav", "trueextent_ardmav"),
    "ardmav_all": ("baseline_ardmav", "trueextent_ardmav", "p2_no_p5_ardmav"),
    # The FATAL extent case is this project's own data, not ARD-MAV: at LABEL = 24 px,
    # 0% of 07_05's boxes can reach IoU 0.5. ardmav_headline does not touch it.
    "local_extent": ("baseline_local", "trueextent_local"),
    # Everything the extent defect touches, controls included.
    "extent_all": ("baseline_local", "trueextent_local",
                   "baseline_ardmav", "trueextent_ardmav"),
    # Does the founding claim survive on a public benchmark, on the official split, at
    # the shipped dt=6? The single-frame arm is NOT trueextent_ardmav: it drops
    # photometric augmentation so that the input representation is the only variable.
    # Running the stack arm alone would be an assertion with a number attached.
    "ardmav_temporal": ("singleframe_ardmav", "temporal_ardmav"),
    # The same one-variable A/B on a second corpus. One benchmark is an anecdote.
    "nps_temporal": ("singleframe_nps", "temporal_nps"),
    # And on the task the project actually exists for: an 8 px drone among 6 px birds,
    # train on 07_05, test on the held-out 10_06 flight. Neither public benchmark has
    # distractors that overlap the target in size, so this is the only place the
    # discrimination claim can be measured at all.
    "local_temporal": ("singleframe_local_ab", "temporal_local_ab"),
}


def names() -> list[str]:
    return sorted(EXPERIMENTS)


def get(name: str) -> ExperimentConfig:
    try:
        return EXPERIMENTS[name]
    except KeyError:
        raise KeyError(f"unknown experiment {name!r}; known experiments: {names()}; "
                       f"known groups: {sorted(GROUPS)}") from None


def resolve(name: str) -> list[ExperimentConfig]:
    """One name -> the list of experiments it stands for (a group expands, in order)."""
    if name in GROUPS:
        return [get(n) for n in GROUPS[name]]
    return [get(name)]


def describe_all() -> str:
    """The registry as a table, for `tools/train.py --list`."""
    rows = [("experiment", "datasets", "imgsz", "batch", "ep", "min_side", "nwd",
             "protocol", "status")]
    for c in (EXPERIMENTS[n] for n in names()):
        rows.append((
            c.name,
            ",".join(c.datasets),
            str(c.imgsz), str(c.batch), str(c.epochs),
            f"{c.min_side:g}",
            "yes" if c.nwd else "-",
            c.protocol_key,
            "ready" if c.runnable else "BLOCKED",
        ))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = ["  ".join(c.ljust(w[i]) for i, c in enumerate(rows[0])),
           "  ".join("-" * x for x in w)]
    out += ["  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows[1:]]
    out.append("")
    out.append("groups (run every member, in order):")
    for g, members in sorted(GROUPS.items()):
        out.append(f"  {g}: {', '.join(members)}")
    return "\n".join(out)


__all__ = [
    "Augmentation", "DEFAULT_AUG", "EXPERIMENTS", "ExperimentConfig", "GROUPS",
    "NO_PHOTOMETRIC_AUG", "UAV_DETR_B4_640", "VRAM_SAFETY_FRACTION", "VramReference",
    "YOLOV8S_P2_ESTIMATE", "YOLOV8S_P2_MEASURED",
    "check_strides", "describe_all", "get", "names", "resolve",
]
