#!/usr/bin/env python3
"""Run a registered experiment, and make the checkpoint it produces traceable.

The problem this replaces
-------------------------
`tools/train_yolo.py` sets no random seed anywhere, records nothing about what produced a
checkpoint, and keeps its hyperparameters in argparse defaults that get overridden from
shell history. Six months later `work/runs/combined-m-p2-640/weights/best.pt` is a file
with a number attached to it and no way to establish which code, which dataset build or
which augmentation produced either. That is not a reproducibility inconvenience; it means
every headline number in the repo rests on an artifact nobody can regenerate.

What this does instead
----------------------
1. The run is a **named config** (`configs/experiments/`), not a command line.
2. Seeds are set for python, numpy and torch, and the seed is in the run directory name,
   because three seeds of one experiment are three different checkpoints.
3. `work/runs/<name>-s<seed>/MANIFEST.json` is written **before the first batch** -- git
   SHA, dirty flag and a hash of the uncommitted diff, the fully resolved config, the
   dataset's file-listing hashes and measured label statistics, package versions, GPU,
   and the exact command line. A run that crashes in epoch 3 is still traceable; a
   manifest written at the end would not have been.
4. The dataset on disk is **checked against the config's claim about it** before
   training. `min_side` is the field that decides what a label means, and the ARD-MAV
   builder writes both the inflated and the true-extent build into the same directory --
   so this samples the label files and refuses to start if the inflation on disk is not
   the inflation the experiment says it is testing.
5. An 8 GB VRAM guard warns before the run instead of after the OOM.

    python tools/train.py --list
    python tools/train.py --config trueextent_ardmav --seeds 3
    python tools/train.py --config temporal_stack_ablation --dry-run   # a group: 2 runs

CONVENTION: everything at module scope here is stdlib. torch and ultralytics are imported
inside `_ultralytics_trainer` and `gpu_info` only, so that this module -- and the tests
that exercise its manifest, seeding and validation logic with a fake trainer -- import in
the torch-free CI job.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from configs.experiments import ExperimentConfig, check_strides, describe_all, resolve  # noqa: E402

MANIFEST_SCHEMA_VERSION = 2

#: Below this, a "true extent" ARD-MAV build should certainly have some boxes: its median
#: target is 11.8 px and 21.3% of its boxes are very-tiny (2-8 px). If the smallest box in
#: a sample is bigger than this, the labels on disk were inflated by a builder run the
#: config does not know about.
TRUE_EXTENT_MAX_MIN_SIDE_PX = 8.0

#: Label sides are written with 6 decimal places, so a normalised side round-trips to
#: within ~0.001 px on a 640 px tile. Half a pixel of slack is generous.
INFLATION_TOLERANCE_PX = 0.5


# ------------------------------------------------------------------------- small utils
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_git(repo: Path, *args: str) -> str | None:
    """Run a git command, or return None if its output could not be read.

    None means "unknown", never "empty". The distinction matters: `git status
    --porcelain` returning "" means the tree is CLEAN, and reporting a failed read as ""
    would stamp a manifest with a clean-tree provenance for a dirty tree.

    `out.stdout` can be None even when returncode is 0. `capture_output` with a `timeout`
    reads the pipes on helper threads, and under heavy disk load one of those threads can
    die -- pytest surfaced it as PytestUnhandledThreadExceptionWarning while a 40,000-tile
    dataset build saturated the disk, and `out.stdout.strip()` then raised AttributeError
    from inside build_manifest(). That is a crash in provenance bookkeeping taking down a
    training run that was otherwise fine, which is the wrong way round.
    """
    try:
        out = subprocess.run(["git", "-C", str(repo), *args],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or out.stdout is None:
        return None
    return out.stdout.strip()


def git_provenance(repo: Path = REPO) -> dict[str, Any]:
    """SHA, branch, dirty flag -- and a hash of the uncommitted diff.

    The diff hash is the part that matters. Every real training run is launched from a
    dirty tree, so `dirty: true` alone tells you the checkpoint is untraceable; a hash of
    `git diff HEAD` at least tells you whether two runs were launched from the *same*
    dirty tree, and lets a saved patch be matched to the run it produced.
    """
    status = _run_git(repo, "status", "--porcelain")
    diff = _run_git(repo, "diff", "HEAD")
    dirty_files = [ln[3:] for ln in (status or "").splitlines() if ln.strip()]
    return {
        "sha": _run_git(repo, "rev-parse", "HEAD"),
        "sha_short": _run_git(repo, "rev-parse", "--short", "HEAD"),
        "branch": _run_git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _run_git(repo, "describe", "--always", "--dirty", "--tags"),
        "dirty": bool(dirty_files),
        "dirty_file_count": len(dirty_files),
        "dirty_files": dirty_files[:50],
        "uncommitted_diff_sha256": sha256_text(diff) if diff else None,
        "uncommitted_diff_bytes": len(diff.encode()) if diff else 0,
    }


def package_versions() -> dict[str, str | None]:
    """Versions WITHOUT importing the packages -- importlib.metadata reads the dist-info.

    Importing torch here would cost ~4 s and a CUDA context on every dry run, and would
    break this module in the torch-free CI job for no gain.
    """
    from importlib.metadata import PackageNotFoundError, version
    out: dict[str, str | None] = {}
    for pkg in ("torch", "torchvision", "ultralytics", "numpy", "scipy",
                "opencv-python", "opencv-python-headless"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def peak_vram() -> dict[str, Any]:
    """Peak VRAM this process reached, so the OOM guard can be corrected by evidence.

    `configs.experiments.base.VramReference` exists to hold a *measured* anchor and its
    docstring says this is where the measurement comes from -- but nothing was writing
    it, so six completed runs left the anchor an estimate. Recorded per run now, in the
    same units the guard reasons in.

    `reserved` is the number that matters for "will it OOM": the allocator holds cached
    blocks it has not handed out, and the card has to have room for those too.

    Reports only when CUDA is ALREADY initialised, and that guard is the point.
    `torch.cuda.max_memory_allocated()` initialises CUDA as a side effect if nothing
    else has, and this runs at the end of every run including the fake-trainer tests --
    so without the guard a pure-CPU test would build a CUDA context it never wanted:
    seconds of startup, a context on a card another run may be holding, and a reporting
    call that can block on a busy GPU. If no training touched the GPU there is no peak
    to report, and saying so is the correct answer.
    """
    try:
        import torch
        if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
            return {"available": False, "reason": "cuda not initialised in this process"}
        return {
            "available": True,
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3),
        }
    except Exception as exc:                                   # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def gpu_info() -> dict[str, Any]:
    """Device name and memory, or an explicit statement of why they are unknown."""
    try:
        import torch
    except Exception as exc:                                   # noqa: BLE001
        return {"available": False, "reason": f"torch not importable: {exc}"}
    try:
        if not torch.cuda.is_available():
            return {"available": False, "reason": "torch.cuda.is_available() is False",
                    "torch": torch.__version__}
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        return {
            "available": True,
            "name": props.name,
            "total_memory_gib": round(props.total_memory / 2 ** 30, 2),
            "capability": f"{props.major}.{props.minor}",
            "count": torch.cuda.device_count(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    except Exception as exc:                                   # noqa: BLE001
        return {"available": False, "reason": f"query failed: {exc}"}


# ------------------------------------------------------------------------ data.yaml I/O
def parse_data_yaml(path: Path) -> dict[str, Any]:
    """Parse the *narrow* data.yaml shape this repo writes -- not YAML in general.

    `make_dataset_external.write_data_yaml` emits exactly `path:`, `train:`, `val:`, an
    optional `channels:`, and a `names:` block of `  <int>: <name>` lines. PyYAML is not
    installed in CI and is not worth a dependency for five keys, so this reads those and
    raises on anything it does not recognise rather than guessing. The raw text is hashed
    into the manifest as well, so the parse is never the only record.
    """
    text = path.read_text(encoding="utf-8")
    out: dict[str, Any] = {"names": {}}
    in_names = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith((" ", "\t")):
            if not in_names:
                raise ValueError(f"{path}: indented line outside a names: block: {raw!r}")
            k, _, v = line.strip().partition(":")
            out["names"][int(k)] = v.strip()
            continue
        in_names = False
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key == "names":
            in_names = True
            if val:                                   # inline list form: names: [a, b]
                out["names"] = {i: s.strip().strip("'\"")
                                for i, s in enumerate(val.strip("[]").split(","))
                                if s.strip()}
            continue
        out[key] = val
    for required in ("train", "val"):
        if required not in out:
            raise ValueError(f"{path}: no '{required}:' key")
    return out


def _labels_dir(images_dir: Path) -> Path:
    """Ultralytics' own images/ -> labels/ rule, applied to the last matching component."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir.parent / "labels"


def label_box_stats(labels_dir: Path, tile_px: int, sample: int = 2000) -> dict[str, Any]:
    """Measure what the labels on disk actually say, from a deterministic sample.

    Sampling, not a full scan, because a tiled ARD-MAV build is ~50k label files and this
    runs before every training launch. The sample is every k-th file of the sorted
    listing, so it is reproducible and spread across videos rather than concentrated in
    the first one.
    """
    if not labels_dir.is_dir():
        return {"present": False, "labels_dir": str(labels_dir)}
    files = sorted(p.name for p in labels_dir.iterdir() if p.suffix == ".txt")
    n = len(files)
    step = max(1, n // sample) if sample > 0 else 1
    chosen = files[::step][:sample] if sample > 0 else files

    sides_px: list[float] = []
    class_ids: set[int] = set()
    n_boxes = n_empty = 0
    for name in chosen:
        body = (labels_dir / name).read_text(encoding="utf-8").strip()
        if not body:
            n_empty += 1
            continue
        for line in body.splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            class_ids.add(int(float(f[0])))
            w, h = float(f[3]) * tile_px, float(f[4]) * tile_px
            sides_px.append(min(w, h))
            n_boxes += 1

    stats: dict[str, Any] = {
        "present": True,
        "labels_dir": str(labels_dir),
        "n_label_files": n,
        "n_sampled_files": len(chosen),
        "n_empty_sampled": n_empty,
        "n_boxes_sampled": n_boxes,
        "class_ids": sorted(class_ids),
        "tile_px_assumed": tile_px,
    }
    if sides_px:
        sides_px.sort()
        stats["min_side_px"] = round(sides_px[0], 3)
        stats["median_side_px"] = round(sides_px[len(sides_px) // 2], 3)
        stats["max_side_px"] = round(sides_px[-1], 3)
    return stats


def listing_hash(directory: Path) -> dict[str, Any]:
    """Hash the sorted (name, size) listing of a split, not its bytes.

    The question this answers is "is this the same dataset build as last time?", and a
    listing hash answers it in milliseconds on 50k files. Hashing the pixels would answer
    a stronger question nobody is asking often enough to pay for it -- `--hash-contents`
    is there for when they are.
    """
    if not directory.is_dir():
        return {"present": False, "dir": str(directory)}
    entries = sorted((e.name, e.stat().st_size)
                     for e in os.scandir(directory) if e.is_file())
    body = "\n".join(f"{n} {s}" for n, s in entries)
    return {"present": True, "dir": str(directory), "n_files": len(entries),
            "total_bytes": sum(s for _, s in entries), "listing_sha256": sha256_text(body)}


def dataset_manifest(data_yaml: Path, tile_px: int, sample: int = 2000,
                     hash_contents: bool = False) -> dict[str, Any]:
    """Everything that identifies the dataset build a run consumed."""
    raw = data_yaml.read_text(encoding="utf-8")
    parsed = parse_data_yaml(data_yaml)
    root = Path(parsed.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    out: dict[str, Any] = {
        "present": True,
        "data_yaml": str(data_yaml.resolve()),
        "data_yaml_sha256": sha256_text(raw),
        "parsed": {k: v for k, v in parsed.items()},
        "root": str(root),
        "splits": {},
    }
    for split in ("train", "val"):
        rel = parsed.get(split)
        if not rel:
            continue
        img_dir = Path(rel)
        if not img_dir.is_absolute():
            img_dir = root / rel
        entry = {"images": listing_hash(img_dir)}
        lbl_dir = _labels_dir(img_dir)
        entry["labels"] = listing_hash(lbl_dir)
        entry["label_stats"] = label_box_stats(lbl_dir, tile_px, sample=sample)
        if hash_contents and img_dir.is_dir():
            h = hashlib.sha256()
            for p in sorted(img_dir.iterdir()):
                if p.is_file():
                    h.update(sha256_file(p).encode())
            entry["images"]["content_sha256"] = h.hexdigest()
        out["splits"][split] = entry
    return out


def check_label_inflation(stats: dict[str, Any], min_side: float) -> str | None:
    """Refuse to train when the labels on disk contradict the config's `min_side`.

    This exists because of a concrete trap: the ARD-MAV tiled builder writes both the
    inflated (min_side 12) and the true-extent (min_side 0) datasets to the SAME
    directory. Running the true-extent experiment against a stale inflated build produces
    a perfectly plausible training curve and a result that answers a different question.
    Nothing else in the pipeline can notice, because both builds are valid YOLO datasets.
    """
    if not stats.get("present") or "min_side_px" not in stats:
        return None
    observed = stats["min_side_px"]
    if min_side > 0:
        if observed < min_side - INFLATION_TOLERANCE_PX:
            return (f"config claims labels inflated to min_side {min_side:g} px, but the "
                    f"smallest sampled box is {observed:.2f} px -- the dataset on disk "
                    f"was built with a smaller --min-side")
        return None
    if observed >= TRUE_EXTENT_MAX_MIN_SIDE_PX:
        return (f"config claims TRUE EXTENTS (min_side 0), but the smallest box in a "
                f"sample of {stats.get('n_boxes_sampled')} is {observed:.2f} px "
                f"(>= {TRUE_EXTENT_MAX_MIN_SIDE_PX:g}) -- this looks like an inflated "
                f"build. Rebuild with --min-side 0 before running this experiment")
    return None


def check_class_count(parsed: dict[str, Any], stats: dict[str, Any],
                      classes: tuple[str, ...]) -> str | None:
    """Refuse to train when the dataset's classes are not the classes the config claims.

    Same failure shape as the inflation check, one field over: ultralytics takes `nc` from
    the data.yaml and never consults the config, so a two-class experiment pointed at a
    one-class build trains a perfectly healthy single-class detector and reports it under a
    name that promises a bird false-alarm rate. `--data` makes that one flag away.

    Only the *declared* names are an error; a split whose sampled labels happen to contain
    no instance of a class is normal (drone-free negative tiles, a rare second class), so
    an unexpected class ID is the error and a missing one is not.
    """
    names = parsed.get("names") or {}
    if names and len(names) != len(classes):
        return (f"data.yaml declares {len(names)} class(es) {sorted(names.values())}, but "
                f"the config claims {len(classes)}: {list(classes)}")
    if stats.get("present"):
        stray = [c for c in stats.get("class_ids", []) if not 0 <= c < len(classes)]
        if stray:
            return (f"labels contain class id(s) {stray}, outside the config's "
                    f"{len(classes)} class(es): {list(classes)}")
    return None


# ---------------------------------------------------------------------------- seeding
def set_seeds(seed: int) -> dict[str, Any]:
    """Seed python, numpy and torch, and record exactly what was done.

    Two honest caveats, both recorded in the manifest rather than implied:

    * `PYTHONHASHSEED` only affects string hashing if it is set *before* the interpreter
      starts. Setting it here fixes it for the dataloader worker processes ultralytics
      forks, not for this process. Export it in the shell for full coverage.
    * `torch.use_deterministic_algorithms` is deliberately NOT called here. Ultralytics
      calls it itself from its `deterministic=` train argument, with the warn-only
      handling its ops need; calling it first turns a slow-but-working run into a crash
      in a backward kernel three epochs in.
    """
    record: dict[str, Any] = {"seed": seed}
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required by cuBLAS for reproducible reductions once torch determinism is on.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    record["env"] = {"PYTHONHASHSEED": os.environ["PYTHONHASHSEED"],
                     "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]}

    import random
    random.seed(seed)
    record["python_random"] = True

    try:
        import numpy as np
        np.random.seed(seed)
        record["numpy"] = True
    except Exception as exc:                                   # noqa: BLE001
        record["numpy"] = f"not seeded: {exc}"

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        record["torch"] = True
        record["torch_use_deterministic_algorithms"] = "left to ultralytics deterministic="
    except Exception as exc:                                   # noqa: BLE001
        record["torch"] = f"not seeded: {exc}"
    return record


# ------------------------------------------------------------------- trainer interface
def ultralytics_kwargs(cfg: ExperimentConfig, seed: int, run_dir: Path,
                       device: str | None = None,
                       data: str | Path | None = None) -> dict[str, Any]:
    """The complete ultralytics `model.train()` call, as data.

    Pure and torch-free on purpose: it is the seam the tests use to assert that the seed,
    the augmentation and the run directory reach the trainer, without a GPU in sight.
    """
    a = cfg.aug
    kw: dict[str, Any] = {
        "data": str(data if data is not None else cfg.data),
        "imgsz": cfg.imgsz,
        "epochs": cfg.epochs,
        "batch": cfg.batch,
        "seed": seed,
        "deterministic": cfg.deterministic,
        # An absolute project path: ~/.config/Ultralytics/settings.json redirects
        # runs_dir, and a relative project= is then nested under another workspace.
        "project": str(run_dir.parent.resolve()),
        "name": run_dir.name,
        # tools/train.py has already created run_dir and written MANIFEST.json into it.
        # Without exist_ok ultralytics treats that as a collision and writes the weights
        # to '<name>2', orphaning the manifest from the checkpoint it describes.
        "exist_ok": True,
        "patience": cfg.patience,
        "workers": cfg.workers,
        "cos_lr": cfg.cos_lr,
        "plots": True,
        "hsv_h": a.hsv_h, "hsv_s": a.hsv_s, "hsv_v": a.hsv_v,
        "degrees": a.degrees, "translate": a.translate, "scale": a.scale,
        "shear": a.shear, "fliplr": a.fliplr, "flipud": a.flipud,
        "mosaic": a.mosaic, "close_mosaic": a.close_mosaic,
        "mixup": a.mixup, "erasing": a.erasing,
    }
    if cfg.lr0 is not None:
        kw["lr0"] = cfg.lr0
    if cfg.freeze is not None:
        kw["freeze"] = cfg.freeze
    if device:
        kw["device"] = device
    return kw


def _ultralytics_trainer(cfg: ExperimentConfig, seed: int, run_dir: Path,
                         kwargs: dict[str, Any]) -> dict[str, Any]:
    """The real trainer. Every heavy import lives in this function and nowhere else."""
    if cfg.channels != 3:
        from dronedet.mc_data import enable_multichannel
        enable_multichannel()
    if cfg.nwd:
        from dronedet.nwd import enable_nwd
        enable_nwd(cfg.nwd_assign_ratio, cfg.nwd_assign_c,
                   cfg.nwd_loss_ratio, cfg.nwd_loss_c)

    from ultralytics import YOLO
    model = YOLO(cfg.model_cfg)
    if cfg.weights:
        model = model.load(cfg.weights)

    # Fail before the first batch, not after three hours: a hand-written architecture
    # yaml that silently keeps or drops a head trains perfectly happily at the wrong
    # resolution and only looks wrong in the final table.
    try:
        strides = [int(s) for s in model.model.model[-1].stride.tolist()]
    except Exception as exc:                                   # noqa: BLE001
        strides = None
        print(f"WARNING: could not read detection-head strides ({exc}); the "
              f"expected_strides={cfg.expected_strides} check was skipped", file=sys.stderr)
    if strides is not None:
        problem = check_strides(strides, cfg.expected_strides)
        if problem:
            raise SystemExit(f"ABORT: {problem}\n  model_cfg: {cfg.model_cfg}")

    results = model.train(**kwargs)

    out: dict[str, Any] = {"strides": strides}
    try:
        import torch
        if torch.cuda.is_available():
            out["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    except Exception:                                          # noqa: BLE001
        pass
    try:
        out["metrics"] = {k: float(v) for k, v in
                          dict(results.results_dict).items()}   # type: ignore[attr-defined]
    except Exception:                                          # noqa: BLE001
        out["metrics"] = None
    return out


Trainer = Callable[[ExperimentConfig, int, Path, dict], dict]


# ---------------------------------------------------------------------------- manifest
def build_manifest(cfg: ExperimentConfig, seed: int, run_dir: Path,
                   dataset: dict[str, Any], kwargs: dict[str, Any],
                   seeding: dict[str, Any], vram_warning: str | None,
                   argv: list[str] | None = None) -> dict[str, Any]:
    """Everything needed to answer 'what made this checkpoint?', assembled before training."""
    argv = list(sys.argv if argv is None else argv)
    proto = cfg.protocol()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "started",
        "experiment_name": cfg.name,
        "run_name": cfg.run_name(seed),
        "run_dir": str(run_dir),
        "seed": seed,
        "config": dataclasses.asdict(cfg),
        "protocol": {
            "key": cfg.protocol_key,
            "describe": proto.describe(),
            "fields": dataclasses.asdict(proto),
        },
        "command": {
            "argv": argv,
            "shell": " ".join(shlex.quote(a) for a in argv),
            "cwd": os.getcwd(),
            "executable": sys.executable,
        },
        "git": git_provenance(),
        "packages": package_versions(),
        "platform": {
            "python": sys.version.split()[0],
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "node": platform.node(),
        },
        "gpu": gpu_info(),
        "seeding": seeding,
        "dataset": dataset,
        "vram": {
            "estimate_gib": round(cfg.estimated_vram_gib(), 2),
            "reference": dataclasses.asdict(cfg.vram_ref),
            "warning": vram_warning,
        },
        "trainer_kwargs": kwargs,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- the runner
def train_one(cfg: ExperimentConfig, seed: int, args: argparse.Namespace,
              trainer: Trainer | None = None) -> Path:
    """Prepare, record, then train one (experiment, seed). Returns the run directory."""
    cfg.require_runnable()
    trainer = trainer or _ultralytics_trainer

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (REPO / out_root)
    run_dir = (out_root / cfg.run_name(seed)).resolve()
    manifest_path = run_dir / "MANIFEST.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"ABORT: {manifest_path} already exists -- that run has already "
                         f"been recorded. Pass --force to overwrite it, or change --seed.")

    data_path = Path(args.data) if args.data else Path(cfg.data)
    if not data_path.is_absolute():
        data_path = REPO / data_path

    if data_path.exists():
        dataset = dataset_manifest(data_path, cfg.tile_px, sample=args.label_sample,
                                   hash_contents=args.hash_contents)
        if not args.skip_data_check:
            for split, entry in dataset["splits"].items():
                problem = (check_label_inflation(entry["label_stats"], cfg.min_side)
                           or check_class_count(dataset["parsed"], entry["label_stats"],
                                                cfg.classes))
                if problem:
                    raise SystemExit(f"ABORT: dataset/{split} disagrees with the config: "
                                     f"{problem}\n  build it with: {cfg.build_command}")
    else:
        dataset = {"present": False, "data_yaml": str(data_path),
                   "note": "dataset absent at manifest time"}
        if not args.dry_run:
            raise SystemExit(f"ABORT: no dataset at {data_path}\n  build it with: "
                             f"{cfg.build_command}")
        print(f"WARNING: {data_path} does not exist; dry run continues without it",
              file=sys.stderr)

    warning = cfg.vram_warning(args.vram_gib)
    if warning:
        print("\n" + "!" * 78, file=sys.stderr)
        print(warning, file=sys.stderr)
        print("!" * 78 + "\n", file=sys.stderr)
    if cfg.unverified_model_cfg:
        print(f"NOTE: {cfg.model_cfg} has never been instantiated in this repo; the "
              f"strides {cfg.expected_strides} are checked before the first batch.",
              file=sys.stderr)

    seeding = set_seeds(seed)
    kwargs = ultralytics_kwargs(cfg, seed, run_dir, device=args.device, data=data_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(cfg, seed, run_dir, dataset, kwargs, seeding, warning)
    write_json(manifest_path, manifest)
    print(f"[{cfg.run_name(seed)}] manifest -> {manifest_path}")

    if args.dry_run:
        print(f"[{cfg.run_name(seed)}] --dry-run: not training")
        return run_dir

    started = time.time()
    try:
        result = trainer(cfg, seed, run_dir, kwargs)
    except BaseException as exc:                                # noqa: BLE001
        write_json(run_dir / "RESULT.json", {
            "run_name": cfg.run_name(seed), "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.time() - started, 1),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        raise

    best = run_dir / "weights" / "best.pt"
    write_json(run_dir / "RESULT.json", {
        "run_name": cfg.run_name(seed),
        "status": "ok",
        "duration_s": round(time.time() - started, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_weights": str(best) if best.exists() else None,
        "best_weights_sha256": sha256_file(best) if best.exists() else None,
        "vram": peak_vram(),
        "trainer": result,
    })
    print(f"[{cfg.run_name(seed)}] done in {time.time() - started:.0f} s -> {run_dir}")
    return run_dir


def seeds_for(cfg: ExperimentConfig, args: argparse.Namespace) -> list[int]:
    """`--seeds N` runs N consecutive seeds from the base, so mean+-std is one command.

    Consecutive rather than arbitrary, because the seeds a headline number was averaged
    over have to be written down somewhere, and 'base+0,1,2' is a rule a reader can check
    against the run directory names.
    """
    base = cfg.seed if args.seed is None else args.seed
    return [base + i for i in range(max(1, args.seeds))]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="experiment or group name (see --list)")
    p.add_argument("--list", action="store_true", help="print the registry and exit")
    p.add_argument("--seed", type=int, default=None,
                   help="base seed (default: the config's own)")
    p.add_argument("--seeds", type=int, default=1,
                   help="run N consecutive seeds sequentially, for mean+-std")
    p.add_argument("--out-root", default="work/runs")
    p.add_argument("--device", default=None, help="ultralytics device string, e.g. '0'")
    p.add_argument("--dry-run", action="store_true",
                   help="write the manifest and stop before training")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing MANIFEST.json")
    p.add_argument("--vram-gib", type=float, default=8.0,
                   help="VRAM budget the OOM guard warns against")
    p.add_argument("--label-sample", type=int, default=2000,
                   help="label files to sample for the dataset statistics (0 = all)")
    p.add_argument("--hash-contents", action="store_true",
                   help="also hash every image's bytes (slow; the listing hash is usually enough)")
    p.add_argument("--skip-data-check", action="store_true",
                   help="do not verify the on-disk label inflation against min_side")
    # Overrides. Each one is re-validated through ExperimentConfig, so an override cannot
    # produce a config the registry would have rejected.
    p.add_argument("--data", default=None, help="override the data.yaml path")
    p.add_argument("--weights", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    return p


def apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    over = {k: getattr(args, k) for k in ("epochs", "batch", "imgsz", "weights")
            if getattr(args, k) is not None}
    if args.data is not None:
        over["data"] = args.data
    return cfg.with_overrides(**over) if over else cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print(describe_all())
        return 0
    if not args.config:
        build_parser().error("--config is required (or --list)")

    resolved = resolve(args.config)
    # A group's arms are a comparison. Pointing all of them at one dataset does not
    # override the experiment, it deletes it -- the temporal ablation's two arms differ in
    # nothing *but* their data, so `--data` would silently turn it into the same run twice.
    if args.data and len({c.data for c in resolved}) > 1:
        build_parser().error(
            f"--data applies to every arm, but {args.config!r} expands to arms that use "
            f"different datasets ({', '.join(sorted({c.data for c in resolved}))}). "
            f"Run the arms one at a time if you really mean to override the data.")

    configs = [apply_overrides(c, args) for c in resolved]
    plan = [(c, s) for c in configs for s in seeds_for(c, args)]
    print(f"plan: {len(plan)} run(s) -> " + ", ".join(c.run_name(s) for c, s in plan))
    for cfg, seed in plan:
        train_one(cfg, seed, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
