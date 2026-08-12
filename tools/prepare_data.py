#!/usr/bin/env python3
"""Turn a downloaded dataset into the two things every later stage needs.

    python tools/prepare_data.py ardmav --out work/prepared/ardmav
    python tools/prepare_data.py uav_smid --out work/prepared/uav_smid --stride 1
    python tools/prepare_data.py yolo_dir --root some/dir --out work/prepared/x
    python tools/prepare_data.py ardmav --dry-run          # parse + report, write nothing

Outputs, under ``--out``::

    gt/<seq>.json      dronedet ground truth, TRUE extent, one file per sequence
    yolo/images/{train,val}/…  + yolo/labels/{train,val}/…  + yolo/data.yaml
    splits.json        which sequence went where, and where the split came from
    manifest.json      everything a reader needs to know what these numbers can mean

The parse lives in `benchmarks/adapters/`; this file is only the two builders and the
report. What it exists to prevent:

**1. Test-set leakage.** The split comes from `Dataset.official_test` / `official_val` via
the adapter, and **test sequences are never exported as training images** (``--export-test``
writes them to ``images/test``, which ``data.yaml`` does not reference). The bug this
guards against actually shipped: `make_dataset_external.combined_splits()` defined the
official ARD-MAV test list and then re-split by position, so rounds 5–7 trained on most of
the official test set and their ARD-MAV numbers cannot sit beside MGMD's.

**2. Silent label inflation.** ``--min-side`` **defaults to 0 — true extent** — and every
run prints the achievable-IoU table for the value chosen. The default in
`make_dataset_external.py` is 12 and `make_datasets_v3.py` hardcodes 24, and at 24 px
**zero per cent** of this repo's own boxes can reach IoU 0.5 (median achievable IoU 0.110),
which makes COCO AP not low but arithmetically impossible
(docs/research/verified-measurements-2026-08.md §6b). Inflation is a training device with
an evaluation cost, so it is opt-in, printed, and written into the manifest — never a
default nobody sees.

Two conventions worth stating because they are load-bearing:

* **Ground truth is never inflated**, whatever ``--min-side`` says. `--min-side` reaches
  the YOLO labels only. An inflated ground truth is not a device, it is a wrong answer.
* **Ground truth keeps non-target classes as ``ignore`` objects.** `dronedet.metrics`
  counts a hit on one as a *distractor* rather than discarding it, which is the only way
  "N hits on 3,162 labelled bird instances" becomes a number. The YOLO labels drop those
  boxes, so a bird image trains as a hard negative.

Video is read strictly sequentially through `dronedet.video.frames` (never seek): both of
this repo's own videos hide their opening seconds behind an MP4 edit list, and inter-frame
codecs make positioned reads unreliable near stream ends.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import numpy as np  # noqa: E402

from benchmarks.adapters import ADAPTERS, Box, build  # noqa: E402
from dataset_stats import _iou_after_inflation  # noqa: E402
from dataset_stats import summarise as print_size_report  # noqa: E402
from dronedet.metrics import SIZE_BINS, size_bin  # noqa: E402

#: Candidate inflations always shown next to the chosen one, so the cost of the repo's two
#: historical defaults (12 in make_dataset_external, 24 in make_datasets_v3) is on screen
#: even when this run used neither.
CONTEXT_MIN_SIDES = (0.0, 8.0, 12.0, 16.0, 24.0)


@dataclass
class SplitCounts:
    images: int = 0
    boxes: int = 0
    negatives: int = 0          # images written with an empty label file


@dataclass
class Report:
    """Everything the manifest is built from. Assembled during the run so that a crash
    halfway leaves a partial manifest rather than a directory nobody can interpret."""

    key: str
    root: str
    out: str
    split_source: str = ""
    splits: dict[str, list[str]] = field(default_factory=dict)
    sequences: dict[str, dict] = field(default_factory=dict)
    yolo: dict[str, SplitCounts] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  !! {msg}")


# --------------------------------------------------------------------------- YOLO labels
def _inflate(box: Box, min_side: float, W: int, H: int
             ) -> tuple[float, float, float, float]:
    """``Box`` -> ``(cx, cy, w, h)`` in pixels, each side at least `min_side`.

    Inflation is concentric and then nudged to keep the box inside the frame; with
    ``min_side=0`` this is exactly the true extent, which is the default and the only
    setting whose ground truth and labels agree.
    """
    w, h = max(box.w, min_side), max(box.h, min_side)
    cx = min(max(box.cx, w / 2), max(W - w / 2, w / 2))
    cy = min(max(box.cy, h / 2), max(H - h / 2, h / 2))
    return cx, cy, w, h


def _label_lines(boxes: list[Box], min_side: float, W: int, H: int,
                 x0: int = 0, y0: int = 0, tw: int | None = None, th: int | None = None
                 ) -> list[str]:
    """YOLO lines for one written image. ``(x0, y0, tw, th)`` describe a crop window; the
    default is the whole image.

    Only boxes whose *centre* falls inside the window are kept. Clipping a partially
    visible target instead would teach the detector a truncated appearance and a wrong
    extent — and at 4 px there is no such thing as a partially visible target.
    """
    tw = tw if tw is not None else W
    th = th if th is not None else H
    lines = []
    for b in boxes:
        if not (x0 <= b.cx <= x0 + tw and y0 <= b.cy <= y0 + th):
            continue
        cx, cy, w, h = _inflate(b, min_side, W, H)
        lines.append(f"0 {(cx - x0) / tw:.6f} {(cy - y0) / th:.6f} {w / tw:.6f} {h / th:.6f}")
    return lines


def _windows(boxes: list[Box], W: int, H: int, tile: int, neg_per_image: int,
             jitter: float, rng: random.Random) -> list[tuple[int, int, int, int]]:
    """Crop windows for one image: one jittered window per target, plus `neg_per_image`
    target-free windows.

    Native-scale crops rather than a resized full frame, because that resize is the whole
    problem: an 11.8 px ARD-MAV target in a 1920-wide frame becomes 3.9 px at 640, i.e.
    below the size at which any of this works. Same lesson as the repo's tiled builder.

    The window is always the full `tile`, never clipped to the image. `_emit` pads a short
    crop up to `tile` at the ORIGIN, so window coordinates and written-image coordinates
    coincide and `_label_lines` may normalise by the window. Clipping the window to
    ``min(tile, W/H)`` instead — which is what this did — normalised by the crop while the
    image on disk was the padded tile: on a 1600x400 source, a target at y=200 was labelled
    at y=320 with its height 60 % too large. Silent, and only in the one direction (an image
    shorter or narrower than the tile) that no fixture happened to cover.
    """
    tw = th = tile
    out: list[tuple[int, int, int, int]] = []
    for b in boxes:
        jx = rng.uniform(-jitter, jitter) * tw
        jy = rng.uniform(-jitter, jitter) * th
        x0 = int(min(max(b.cx + jx - tw / 2, 0), max(W - tw, 0)))
        y0 = int(min(max(b.cy + jy - th / 2, 0), max(H - th, 0)))
        out.append((x0, y0, tw, th))
    for _ in range(neg_per_image):
        x0 = rng.randint(0, max(W - tw, 0))
        y0 = rng.randint(0, max(H - th, 0))
        if all(not (x0 <= b.cx <= x0 + tw and y0 <= b.cy <= y0 + th) for b in boxes):
            out.append((x0, y0, tw, th))
    return out


def _write_image(img, path: Path, quality: int) -> None:
    import cv2

    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def _emit(img, boxes: list[Box], stem: str, img_dir: Path, lbl_dir: Path,
          args, counts: SplitCounts, rng: random.Random) -> None:
    """Write one source image as one or more YOLO samples."""
    H, W = img.shape[:2]
    if args.tile and (W > args.tile or H > args.tile):
        wins = _windows(boxes, W, H, args.tile, args.neg_per_image, args.jitter, rng)
        samples = []
        for k, (x0, y0, tw, th) in enumerate(wins):
            crop = img[y0:y0 + th, x0:x0 + tw]
            if crop.shape[0] != args.tile or crop.shape[1] != args.tile:
                pad = np.zeros((args.tile, args.tile, 3), crop.dtype)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad
            samples.append((f"{stem}_{k}", crop,
                            _label_lines(boxes, args.min_side, W, H, x0, y0, tw, th)))
    else:
        samples = [(stem, img, _label_lines(boxes, args.min_side, W, H))]

    for name, image, lines in samples:
        _write_image(image, img_dir / f"{name}.jpg", args.jpeg_quality)
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines))
        counts.images += 1
        counts.boxes += len(lines)
        if not lines:
            counts.negatives += 1


def _select_frames(by_frame: dict[int, list[Box]], targets: dict[int, list[Box]],
                   stride: int, neg_stride: int) -> list[int]:
    """Which annotated frames to export.

    Positives are subsampled because consecutive frames of one track are near-duplicates
    (that is also why the split is by whole sequence). Annotated *empty* frames are kept
    at a coarser stride: a clip's own drone-free background, at its own exposure and its
    own compression artefacts, is the best hard negative available, and dropping them is
    how a detector learns that "sky" means "drone".
    """
    pos = sorted(f for f in by_frame if targets.get(f))
    neg = sorted(f for f in by_frame if not targets.get(f))
    return sorted(set(pos[::max(stride, 1)]) | set(neg[::max(neg_stride, 1)]))


# --------------------------------------------------------------------------- the run
def prepare(args) -> Report:
    ad = build(args.key, args.root)
    out = Path(args.out)
    rep = Report(key=args.key, root=str(ad.root), out=str(out))

    seqs = ad.sequences()
    if args.sequences:
        want = {s.strip() for s in args.sequences.split(",") if s.strip()}
        missing = want - set(seqs)
        if missing:
            sys.exit(f"no such sequence(s): {sorted(missing)}")
        seqs = [s for s in seqs if s in want]
    if args.limit:
        seqs = seqs[:args.limit]
    if not seqs:
        sys.exit(f"adapter {args.key!r} found no sequences under {ad.root}")

    rep.split_source = ad.split_source()
    print(f"\n{args.key}: {len(seqs)} sequence(s) under {ad.root}")
    print(f"split source: {rep.split_source}")

    # ---------------------------------------------------------------- ground truth first
    all_wh: list[tuple[float, float]] = []
    parsed: dict[str, tuple[dict[int, list[Box]], dict[int, list[Box]]]] = {}
    for seq in seqs:
        by_frame = ad.boxes(seq)
        targets = {f: [b for b in bs if b.cls.lower() in ad.positive_classes]
                   for f, bs in by_frame.items()}
        parsed[seq] = (by_frame, targets)
        split = ad.split_of(seq)
        rep.splits.setdefault(split, []).append(seq)
        n_t = sum(len(v) for v in targets.values())
        n_d = sum(len(v) for v in by_frame.values()) - n_t
        rep.sequences[seq] = {
            "split": split,
            "annotated_frames": len(by_frame),
            "target_boxes": n_t,
            "distractor_boxes": n_d,
            "conditions": [c.value for c in ad.conditions(seq)],
        }
        all_wh.extend((b.w, b.h) for bs in targets.values() for b in bs)
        if not args.dry_run:
            # Hand the parse back rather than making the adapter redo it: ARD-MAV is
            # 107,497 XML files and this loop would otherwise read them all twice.
            ad.ground_truth(seq, boxes=by_frame).save(out / "gt" / f"{seq}.json")

    rep.splits = {k: sorted(v) for k, v in sorted(rep.splits.items())}
    for split in ("train", "val", "test"):
        rep.splits.setdefault(split, [])
    for split, members in rep.splits.items():
        print(f"  {split:5s} {len(members):4d} sequence(s)")
    if not rep.splits["test"]:
        rep.warn("no test sequences: nothing here can be scored without leaking training data")
    if not args.dry_run:
        (out / "splits.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "splits.json").write_text(json.dumps(
            {"split_source": rep.split_source, "splits": rep.splits}, indent=1))

    # ---------------------------------------------------------------- the extent report
    size_stats = _size_report(all_wh, args, rep)

    if args.gt_only or args.dry_run:
        counts_json = {}
    else:
        counts_json = _build_yolo(ad, parsed, rep, out, args)

    if not args.dry_run:
        manifest = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "argv": sys.argv,
            "adapter": args.key,
            "root": rep.root,
            "out": rep.out,
            "split_source": rep.split_source,
            "splits": rep.splits,
            "official_protocol": _protocol_note(ad),
            "min_side": float(args.min_side),
            "label_extent": ("true extent (no inflation)" if not args.min_side
                             else f"inflated to a minimum side of {args.min_side:g} px"),
            "gt_extent": "true extent always -- --min-side never reaches gt/",
            "tile": args.tile,
            "stride": args.stride,
            "neg_stride": args.neg_stride,
            "seed": args.seed,
            "sequences": rep.sequences,
            "size_distribution": size_stats,
            "yolo": counts_json,
            "warnings": rep.warnings,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"\nwrote {out}/manifest.json")
    return rep


def _protocol_note(ad) -> str | None:
    d = ad.dataset
    if d is None or d.official_protocol is None:
        return None
    return d.official_protocol.describe()


def _size_report(all_wh: list[tuple[float, float]], args, rep: Report) -> dict:
    """Print the size distribution and the achievable-IoU table, and return them as data.

    This is not optional output. `--min-side` is a number whose consequence is invisible
    until evaluation, months later, on a machine that is not this one; printing the cost
    at build time is the only place it can still change a decision.
    """
    if not all_wh:
        rep.warn("no target boxes parsed -- size report skipped "
                 "(check the class table: are the positives named something unexpected?)")
        return {}

    wh = np.asarray(all_wh, dtype=float)
    sides = np.sqrt(wh[:, 0] * wh[:, 1])
    thresholds = tuple(sorted(set(CONTEXT_MIN_SIDES) | {float(args.min_side)}))
    print_size_report(sides, wh, f"{args.key} (target boxes, true extent)",
                      thresholds=thresholds)
    print(f"    the row in use for this build is min_side = {args.min_side:g}.")

    table = []
    for ms in thresholds:
        ious = np.array([_iou_after_inflation(w, h, ms) for w, h in wh])
        table.append({
            "min_side": ms,
            "boxes_grown_pct": round(100 * float(np.mean((wh[:, 0] < ms) | (wh[:, 1] < ms))), 2),
            "median_best_iou": round(float(np.median(ious)), 4),
            "pct_can_reach_iou_050": round(100 * float(np.mean(ious >= 0.5)), 2),
            "pct_can_reach_iou_025": round(100 * float(np.mean(ious >= 0.25)), 2),
        })

    chosen = next(r for r in table if r["min_side"] == float(args.min_side))
    if args.min_side:
        rep.warn(f"--min-side {args.min_side:g} inflates {chosen['boxes_grown_pct']:.1f}% of "
                 f"boxes; a perfectly-centred prediction then caps at median IoU "
                 f"{chosen['median_best_iou']:.3f}, and only "
                 f"{chosen['pct_can_reach_iou_050']:.1f}% of boxes can reach IoU 0.5. Any "
                 f"IoU-based score from this build is bounded by geometry, not by the "
                 f"detector. Record it in the protocol as label_inflation_px.")

    return {
        "n_target_boxes": int(len(wh)),
        "sqrt_area_px": {
            "min": round(float(sides.min()), 2),
            "p25": round(float(np.percentile(sides, 25)), 2),
            "median": round(float(np.median(sides)), 2),
            "p75": round(float(np.percentile(sides, 75)), 2),
            "max": round(float(sides.max()), 2),
        },
        "ai_tod_bins_pct": {
            name: round(100 * float(np.mean([size_bin(s, s) == name for s in sides])), 2)
            for name, _, _ in SIZE_BINS
        },
        "achievable_iou": table,
    }


def _build_yolo(ad, parsed, rep: Report, out: Path, args) -> dict:
    """Export images + labels for train/val (and test only on request).

    Test images are withheld by default and, when exported, land in ``images/test``, which
    ``data.yaml`` never references. That is deliberate belt-and-braces: the leak this repo
    actually suffered was not malice, it was a builder that could reach the test videos at
    all.
    """
    from dronedet.video import frames as video_frames

    rng = random.Random(args.seed)
    root = out / "yolo"
    wanted_splits = ["train", "val"] + (["test"] if args.export_test else [])
    print(f"\nYOLO export -> {root}  (splits: {', '.join(wanted_splits)}"
          f"{'' if args.export_test else '; test withheld'})")

    for seq, (by_frame, targets) in parsed.items():
        split = ad.split_of(seq)
        if split not in wanted_splits:
            continue
        counts = rep.yolo.setdefault(split, SplitCounts())
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        try:
            src = ad.image_source(seq)
        except (FileNotFoundError, NotImplementedError) as exc:
            rep.warn(f"{seq}: no pixels ({exc}); ground truth written, no training images")
            continue

        stride = args.stride if args.stride is not None else (4 if src.kind == "video" else 1)
        neg_stride = args.neg_stride if args.neg_stride is not None else stride * 4
        want = set(_select_frames(by_frame, targets, stride, neg_stride))
        if not want:
            continue

        before = counts.images
        if src.kind == "video":
            if not src.video.exists():
                rep.warn(f"{seq}: video missing at {src.video}; ground truth only")
                continue
            last = max(want)
            for idx, frame in video_frames(str(src.video)):     # sequential; never seek
                if idx > last:
                    break
                if idx in want:
                    _emit(frame, targets.get(idx, []), f"{ad.key or 'yolo'}__{seq}__{idx:06d}",
                          img_dir, lbl_dir, args, counts, rng)
        else:
            import cv2

            for idx, path in src.images:
                if idx not in want:
                    continue
                img = cv2.imread(str(path))
                if img is None:
                    rep.warn(f"{seq}: unreadable image {path}")
                    continue
                _emit(img, targets.get(idx, []), f"{ad.key or 'yolo'}__{seq}__{idx:06d}",
                      img_dir, lbl_dir, args, counts, rng)
        print(f"  [{split}] {seq}: {counts.images - before} images")

    _write_data_yaml(root, rep)
    counts_json = {k: {"images": v.images, "boxes": v.boxes, "empty_labels": v.negatives}
                   for k, v in sorted(rep.yolo.items())}
    for split, c in counts_json.items():
        print(f"  {split:5s} {c['images']:6d} images / {c['boxes']:6d} boxes / "
              f"{c['empty_labels']:6d} negatives")
    if not rep.yolo.get("val", SplitCounts()).images:
        rep.warn("no val images: training will have no held-out set to select on")
    return counts_json


def _write_data_yaml(root: Path, rep: Report) -> None:
    """One class, always.

    Non-target classes were dropped from the labels on purpose (see the module docstring):
    a bird image is a hard negative, not a bird detection task. Written by hand rather than
    with PyYAML, which is not installed in the torch-free CI job.
    """
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# generated by tools/prepare_data.py from adapter '{rep.key}'",
        f"# split source: {rep.split_source}",
        "# 'test' is deliberately absent: no training entry point may reach it.",
        f"path: {root.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: drone",
    ]
    (root / "data.yaml").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key", choices=sorted(ADAPTERS), help="which adapter to run")
    ap.add_argument("--root", type=Path, default=None,
                    help="dataset root (default: benchmarks.adapters.DEFAULT_ROOTS)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-side", type=float, default=0.0,
                    help="minimum side, in px, for YOLO TRAINING labels. DEFAULT 0 = "
                         "preserve true extent. Any value above 0 caps the achievable IoU "
                         "-- the printed table says by how much. Never reaches gt/.")
    ap.add_argument("--tile", type=int, default=640,
                    help="native-resolution crop size for images larger than it; 0 writes "
                         "whole frames (which downscale a 12 px target to 4 at train time)")
    ap.add_argument("--stride", type=int, default=None,
                    help="keep every Nth annotated frame that contains a target "
                         "(default 4 for video, 1 for stills)")
    ap.add_argument("--neg-stride", type=int, default=None,
                    help="keep every Nth annotated frame with no target (default 4*stride)")
    ap.add_argument("--neg-per-image", type=int, default=1,
                    help="target-free crops per source image, when tiling")
    ap.add_argument("--jitter", type=float, default=0.35,
                    help="crop-centre jitter as a fraction of the tile")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sequences", default="", help="comma-separated subset")
    ap.add_argument("--limit", type=int, default=0, help="first N sequences only (smoke run)")
    ap.add_argument("--gt-only", action="store_true", help="write gt/ + manifest, no images")
    ap.add_argument("--export-test", action="store_true",
                    help="also write images/test. data.yaml still never references it.")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse and print the report; write nothing")
    args = ap.parse_args(argv)

    rep = prepare(args)
    if rep.warnings:
        print(f"\n{len(rep.warnings)} warning(s) recorded in the manifest:")
        for w in rep.warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
