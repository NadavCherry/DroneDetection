"""The experiment config type: everything that decides what a checkpoint means.

Why this exists
---------------
`tools/train_yolo.py` sets no seed anywhere, writes no record of what produced a
checkpoint, and keeps its hyperparameters in argparse defaults that are then copied by
hand into shell history. The result is that **no weights file in `work/runs/` can be
traced to the code and the data that made it**, and no headline number can be re-run.
An `ExperimentConfig` is the fix: the run is named, the knobs are values in version
control, and `tools/train.py` stamps the resolved config into
`work/runs/<name>-s<seed>/MANIFEST.json` *before* the first batch.

Why Python dataclasses and not YAML
-----------------------------------
CI installs numpy/scipy/opencv/pytest and nothing else -- there is no PyYAML to parse a
config with, and adding one to run five configs would be a dependency taken on for
punctuation. Dataclasses also mean `validate()` runs at import time, so a config that
contradicts itself fails in the test suite rather than three hours into a run.

Stdlib only (plus `benchmarks.protocol`, which is also stdlib only), so everything here
imports in the torch-free CI job. Nothing in this module may import torch or ultralytics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from benchmarks.catalog import DATASETS
from benchmarks.protocol import BY_KEY, Protocol

# A dataset key that is not in `benchmarks.catalog` must say so out loud: the repo's own
# two clips are a scene, not a benchmark, and a number measured on them cannot be placed
# beside a published one. The prefix makes that visible in the manifest.
LOCAL_PREFIX = "local:"


# --------------------------------------------------------------------------- VRAM guard
@dataclass(frozen=True)
class VramReference:
    """One anchor point for the OOM guard: `gib` peak at `batch` x `imgsz`.

    `measured` is the whole point of the type. The only VRAM figure this project has
    actually observed is UAV-DETR's, and extrapolating a DETR's activation cost onto a
    YOLO would over-warn on every single config -- so the YOLO anchor is labelled an
    estimate, and every warning prints which anchor it used and whether anyone ever saw
    it happen. `tools/train.py` records `torch.cuda.max_memory_allocated()` into
    RESULT.json precisely so the first real run can replace the estimate with a fact.
    """

    gib: float
    batch: int
    imgsz: int
    source: str
    measured: bool

    @property
    def gib_per_batch_megapixel(self) -> float:
        return self.gib / (self.batch * (self.imgsz ** 2) / 1e6)


#: The one number this project has measured. Quoted in every OOM warning as the anchor.
UAV_DETR_B4_640 = VramReference(
    gib=5.14, batch=4, imgsz=640, measured=True,
    source="UAV-DETR reported peak training memory, batch 4 @ 640 px")

#: MEASURED 2026-08-12 on an RTX 4080 Laptop (sm_89, 12 GiB), yolov8s-p2 @ 640 px, one
#: epoch per point, peak `torch.cuda.max_memory_reserved()`:
#:
#:     batch  8 ->  3.73 GiB (31.1 %)
#:     batch 16 ->  7.08 GiB (59.1 %)
#:     batch 32 -> 10.39 GiB (86.7 %)
#:
#: which fits 0.37 GiB fixed + 0.419 GiB per sample. The estimate this replaces claimed
#: 6.55 GiB at batch 16 and was ~8 % optimistic -- close, but this type exists to carry
#: facts and now it carries one. Note that batch 32 lands ABOVE VRAM_SAFETY_FRACTION on
#: this card: it runs, but it is not a thing to schedule unattended for six hours.
YOLOV8S_P2_MEASURED = VramReference(
    gib=7.08, batch=16, imgsz=640, measured=True,
    source="MEASURED 2026-08-12, RTX 4080 Laptop 12 GiB, yolov8s-p2 @ 640 px, peak "
           "max_memory_reserved over one epoch; fits 0.37 GiB + 0.419 GiB/sample. "
           "Raw readings: work/_vram_probe.json")

#: The former name, kept so nothing that imported it breaks. It now points at the fact.
YOLOV8S_P2_ESTIMATE = YOLOV8S_P2_MEASURED

#: Headroom left for the CUDA context, cuDNN workspaces and fragmentation. A run that
#: fits at exactly 8.0 GiB in arithmetic does not fit on an 8 GB card.
VRAM_SAFETY_FRACTION = 0.85


# ------------------------------------------------------------------------ augmentation
@dataclass(frozen=True)
class Augmentation:
    """Ultralytics augmentation knobs, as data rather than as argparse defaults."""

    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.08
    scale: float = 0.25
    shear: float = 0.0
    fliplr: float = 0.5
    flipud: float = 0.0
    mosaic: float = 0.3
    close_mosaic: int = 12
    mixup: float = 0.0
    erasing: float = 0.0


DEFAULT_AUG = Augmentation()

#: For any input whose channels are not colour. A temporal stack's three channels are
#: stabilized grays at t-2dt / t-dt / t, so hue and saturation jitter do not change the
#: appearance of a drone -- they remix *when* it was. The v3 rounds pass 0 0 for this
#: reason, and `ExperimentConfig.validate()` now enforces it instead of trusting the CLI.
NO_PHOTOMETRIC_AUG = Augmentation(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)


# -------------------------------------------------------------------------- the config
@dataclass(frozen=True)
class ExperimentConfig:
    """One training run, completely specified.

    Every field is either a hyperparameter that changes the result or a claim about what
    the result means (`datasets`, `min_side`, `protocol_key`). The second kind is why
    this is not just a dict of kwargs: `protocol()` turns `min_side` into the
    `label_inflation_px` field of a `benchmarks.protocol.Protocol`, so that comparing a
    label-inflated run against a published IoU number *automatically* reports that the
    comparison is capped by geometry rather than by the detector.
    """

    name: str
    datasets: tuple[str, ...]
    data: str                                  # repo-relative path to the YOLO data.yaml
    model_cfg: str                             # ultralytics model yaml (builtin or configs/)
    weights: str                               # checkpoint to transfer from ('' = scratch)

    imgsz: int = 640
    batch: int = 8
    epochs: int = 60
    seed: int = 0
    patience: int = 25
    workers: int = 4
    cos_lr: bool = True
    lr0: float | None = None                   # None keeps ultralytics' own default
    freeze: int | None = None
    deterministic: bool = True

    channels: int = 3                          # >3 needs dronedet.mc_data (.npy inputs)
    classes: tuple[str, ...] = ("drone",)
    tile_px: int = 640                         # side of a training tile, for label checks
    min_side: float = 12.0                     # label inflation applied when building
    temporal_stack: bool = False
    stack_dt: int | None = None                # channel spacing in frames, if stacked

    nwd: bool = False
    nwd_assign_ratio: float = 0.5
    nwd_assign_c: float = 16.0
    nwd_loss_ratio: float = 0.5
    nwd_loss_c: float = 2.0

    aug: Augmentation = DEFAULT_AUG
    expected_strides: tuple[int, ...] = (4, 8, 16, 32)
    unverified_model_cfg: bool = False
    vram_ref: VramReference = YOLOV8S_P2_ESTIMATE

    protocol_key: str = "specklock-centre"
    build_command: str = ""                    # exact command that produces `data`
    missing: str = ""                          # non-empty => not runnable, and why
    notes: str = ""
    tags: tuple[str, ...] = ()

    # ------------------------------------------------------------------ validation
    def __post_init__(self) -> None:
        errs = self.validate()
        if errs:
            raise ValueError(f"invalid experiment config {self.name!r}:\n  - "
                             + "\n  - ".join(errs))

    def validate(self) -> list[str]:
        """Every way this config contradicts itself. Empty list means it is coherent."""
        e: list[str] = []
        if not self.name or not all(c.isalnum() or c in "_-" for c in self.name):
            e.append(f"name {self.name!r} must be a non-empty [A-Za-z0-9_-] run-dir name")
        if not self.datasets:
            e.append("datasets is empty: a run that cannot name its corpus is untraceable")
        for k in self.datasets:
            if k.startswith(LOCAL_PREFIX):
                continue
            if k not in DATASETS:
                e.append(f"unknown dataset key {k!r}: add it to benchmarks/catalog.py, or "
                         f"prefix it {LOCAL_PREFIX!r} if it is not a published benchmark")
        if not self.data:
            e.append("data (path to data.yaml) is required")
        if not self.model_cfg:
            e.append("model_cfg is required")

        if self.imgsz <= 0 or self.imgsz % 32:
            e.append(f"imgsz {self.imgsz} must be positive and a multiple of 32")
        if self.batch <= 0:
            e.append(f"batch {self.batch} must be positive")
        if self.epochs <= 0:
            e.append(f"epochs {self.epochs} must be positive")
        if self.seed < 0:
            e.append(f"seed {self.seed} must be >= 0")
        if self.patience < 0:
            e.append("patience must be >= 0")
        if self.tile_px <= 0:
            e.append("tile_px must be positive")
        if self.min_side < 0:
            e.append("min_side must be >= 0 (0 means true extents, no inflation)")
        if self.min_side >= self.tile_px:
            e.append(f"min_side {self.min_side} >= tile_px {self.tile_px}: every label "
                     f"would fill the tile")
        if not self.classes:
            e.append("classes is empty")
        if self.channels < 1:
            e.append("channels must be >= 1")

        for fname in ("nwd_assign_ratio", "nwd_loss_ratio"):
            v = getattr(self, fname)
            if not 0.0 <= v <= 1.0:
                e.append(f"{fname} {v} must be in [0, 1] (it is a blend weight)")
        for fname in ("nwd_assign_c", "nwd_loss_c"):
            if getattr(self, fname) <= 0:
                e.append(f"{fname} must be > 0 (it is the NWD normalising constant)")

        a = self.aug
        for fname in ("hsv_h", "hsv_s", "hsv_v", "translate", "scale", "fliplr",
                      "flipud", "mosaic", "mixup", "erasing"):
            v = getattr(a, fname)
            if not 0.0 <= v <= 1.0:
                e.append(f"aug.{fname} {v} out of range [0, 1]")
        if a.close_mosaic < 0:
            e.append("aug.close_mosaic must be >= 0")
        if a.close_mosaic > self.epochs:
            e.append(f"aug.close_mosaic {a.close_mosaic} > epochs {self.epochs}: mosaic "
                     f"would never be on")

        # The invariant from the v3 rounds, promoted from a CLI habit to a rule.
        if self.temporal_stack and (a.hsv_h or a.hsv_s or a.hsv_v):
            e.append("temporal_stack inputs are stabilized grays at t-2dt/t-dt/t; HSV "
                     "jitter remixes the moments semantically. Use NO_PHOTOMETRIC_AUG.")
        if self.temporal_stack and not self.stack_dt:
            e.append("temporal_stack needs stack_dt: the channel spacing is part of the "
                     "experiment (DT=9 was measured and lost; DT=6 and DT=3 shipped)")
        if not self.temporal_stack and self.stack_dt:
            e.append("stack_dt is set but temporal_stack is False")

        if self.channels != 3 and not self.data.endswith(".yaml"):
            e.append("multi-channel training loads .npy via dronedet.mc_data and still "
                     "needs a data.yaml")
        if self.protocol_key not in BY_KEY:
            e.append(f"unknown protocol_key {self.protocol_key!r}; "
                     f"known: {sorted(BY_KEY)}")
        if not self.notes:
            e.append("notes is required: a config with no stated rationale is a config "
                     "nobody can decide whether to trust")
        if not self.missing and not self.build_command:
            e.append("build_command is required for a runnable config: it is the only "
                     "record of how `data` was produced")
        return e

    # ------------------------------------------------------------------ derived facts
    def run_name(self, seed: int | None = None) -> str:
        """`<name>-s<seed>`. The seed is in the directory name because three seeds of the
        same experiment are three checkpoints that must not overwrite each other."""
        return f"{self.name}-s{self.seed if seed is None else seed}"

    def protocol(self) -> Protocol:
        """The protocol this run's numbers will carry, with label inflation stamped in.

        This is what makes the true-extent experiment self-documenting: a run built with
        `min_side=12` produces a protocol whose `mismatches_with(ARDMAV_OFFICIAL)` says
        the IoU comparison is capped by the labels, and a `min_side=0` run's does not.
        """
        base = BY_KEY[self.protocol_key]
        return replace(base, label_inflation_px=(self.min_side or None))

    @property
    def external_dataset_keys(self) -> tuple[str, ...]:
        """Only the catalogued benchmarks -- what an external claim may cite."""
        return tuple(k for k in self.datasets if not k.startswith(LOCAL_PREFIX))

    @property
    def runnable(self) -> bool:
        return not self.missing

    def require_runnable(self) -> None:
        """Raise if this config names work that has not been done yet.

        Deliberately `NotImplementedError` and not a friendly warning: a config whose
        dataset builder does not exist must not silently degrade into training on
        whatever happens to be on disk under that path.
        """
        if self.missing:
            raise NotImplementedError(
                f"experiment {self.name!r} is not runnable yet: {self.missing}")

    def with_overrides(self, **kw) -> "ExperimentConfig":
        """A copy with fields replaced, re-validated. Used for CLI overrides so that an
        override cannot produce a config the registry would have rejected."""
        return replace(self, **kw)

    # ------------------------------------------------------------------ the VRAM guard
    def estimated_vram_gib(self, batch: int | None = None,
                           imgsz: int | None = None) -> float:
        b = self.batch if batch is None else batch
        s = self.imgsz if imgsz is None else imgsz
        return self.vram_ref.gib_per_batch_megapixel * b * (s ** 2) / 1e6

    def max_batch_for(self, budget_gib: float) -> int:
        """Largest batch whose estimate stays inside the safety fraction of `budget_gib`."""
        per_image = self.estimated_vram_gib(batch=1)
        if per_image <= 0:
            return self.batch
        return max(1, int((budget_gib * VRAM_SAFETY_FRACTION) // per_image))

    def vram_warning(self, budget_gib: float = 8.0) -> str | None:
        """A loud, self-doubting OOM warning, or None if the run looks like it fits.

        It quotes the anchor it reasoned from and says whether anyone has observed it,
        because the reader's first question is "how would you know?".

        That answer improved on 2026-08-12. It used to be "from one measurement of a
        *different* architecture" -- UAV-DETR's batch-4 figure, extrapolated onto a YOLO.
        yolov8s-p2 has since been measured directly on this hardware at three batch
        sizes, so a yolov8s-p2 config is now warned about on the strength of yolov8s-p2
        numbers. The DETR anchor is still quoted when it is the one in use.
        """
        est = self.estimated_vram_gib()
        limit = budget_gib * VRAM_SAFETY_FRACTION
        if est <= limit:
            return None
        ref = self.vram_ref
        return (
            f"LIKELY OOM: {self.name} at batch {self.batch} x {self.imgsz} px estimates "
            f"{est:.1f} GiB against a {budget_gib:.0f} GiB card "
            f"({limit:.1f} GiB usable after context/fragmentation).\n"
            f"  basis: {ref.source} -> {ref.gib_per_batch_megapixel:.2f} GiB per "
            f"batch-megapixel"
            f"{'  [MEASURED]' if ref.measured else '  [UNMEASURED ESTIMATE]'}.\n"
            f"  measured anchors this project owns: "
            f"{YOLOV8S_P2_MEASURED.gib:.2f} GiB at batch {YOLOV8S_P2_MEASURED.batch} x "
            f"{YOLOV8S_P2_MEASURED.imgsz} px (yolov8s-p2, this repo, 2026-08-12); "
            f"{UAV_DETR_B4_640.gib:.2f} GiB at batch {UAV_DETR_B4_640.batch} x "
            f"{UAV_DETR_B4_640.imgsz} px ({UAV_DETR_B4_640.source}).\n"
            f"  suggestion: --batch {self.max_batch_for(budget_gib)}, or keep the batch "
            f"and pass ultralytics `nbs` gradient accumulation.")


def check_strides(actual: Iterable[float], expected: Iterable[int]) -> str | None:
    """Compare a built model's detection-head strides with what the config expects.

    A hand-written architecture yaml that drops a head is exactly the kind of edit that
    loads without error and trains for three hours at the wrong resolution. `tools/train.py`
    runs this the moment the model is instantiated, before the first batch.
    """
    a = [int(x) for x in actual]
    b = [int(x) for x in expected]
    if a == b:
        return None
    return (f"detection-head strides {a} do not match the config's expected {b}: the "
            f"model yaml is not the architecture this experiment claims to be testing")
