#!/usr/bin/env python3
"""Rebuild an NPS split the way YOLOMG's own data-prep script builds it, and measure
what that choice alone is worth in AP.

WHY THIS EXISTS
---------------
YOLOMG reports AP 0.95 on NPS-Drones. Trained from their code, on their recipe, and
scored by their own val.py, our reproduction reaches 0.79 on a video-disjoint validation
split and 0.53 on a video-disjoint test split. `tools/protocol_sweep.py` already ruled out
the evaluator: frame selection, 101-point interpolation, matcher, confidence floor and
aggregation together move the number by about 0.01, not 0.42.

That leaves the split. Their `data/split_train_val.py` reads:

    trainval_percent = 1.0
    train_percent    = 0.85
    total_xml = os.listdir(xmlfilepath)        # a FLAT list of every frame image
    trainval  = random.sample(list_index, tv)  # sampled per FRAME, not per VIDEO

Two consequences follow, and neither is visible from the paper:

  1. The partition is per-frame across all videos. Frame t can train while frame t+1
     validates. On NPS the camera and the target move a few pixels between consecutive
     sampled frames, so a validation frame is very nearly a training frame with noise.
     This is temporal leakage; the model can score well by recognising a background it
     has already memorised rather than by detecting a drone.
  2. `trainval_percent = 1.0` makes `tv == num`, so every index lands in `trainval` and
     the `else` branch that writes `test.txt` never executes. Their test list is EMPTY.
     Whatever the pipeline reports as a held-out number is the 15 % validation slice --
     the one interleaved with training frames.

So this script builds the leak split and nothing else changes: the same images, the same
labels, the same masks, the same model, the same 100 epochs at 1280 px. Only the rule
that assigns a frame to train or val differs. Whatever AP moves is what the rule was
worth.

CONTROLLING THE OBVIOUS CONFOUND
--------------------------------
A leak split drawn from all 50 clips would also give the model footage from clips the
disjoint arm never sees, confounding "leakage" with "more diverse data". `--pool
disjoint-clips` therefore restricts the pool to exactly the clips the disjoint arm
already had (train + val, clips 1-40) and re-partitions only those. That arm ends up with
FEWER training images than the disjoint arm (85 % of 8,330 = 7,080 against 8,019), so it
is strictly handicapped on data volume. If it still scores higher, the partition rule is
the only thing left to credit.

`--pool all-clips` reproduces their script faithfully instead, pooling all 50 clips. Both
are worth having: the first isolates the mechanism, the second measures the protocol as
actually published.

COST
----
None worth mentioning. Nothing is extracted, converted or copied -- the image and label
files already exist and keep their paths. This writes list files, so the whole experiment
is a re-partition of text.

    python tools/sota/make_leak_split.py \
        --dataset work/ext_datasets/yolomg_nps --style yolomg \
        --pool disjoint-clips --seed 0 --out work/ext_datasets/yolomg_nps_leak
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def _read_list(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def _clip_of(path: str) -> str:
    """'Clip_001_000056.jpg' -> 'Clip_001'. Used only for the audit, never for the split.

    The whole point of the leak arm is that the split does NOT respect this key; it is
    recorded so the report can state how many clips each side draws from.
    """
    stem = Path(path).stem
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def build(dataset: Path, style: str, pool: str, seed: int, out: Path,
          train_frac: float) -> dict:
    """Write a leak-split dataset definition beside the original. Returns an audit dict.

    Both paths are resolved to absolute FIRST. Everything written here is read by a
    process with a different working directory -- YOLOMG's train.py runs from
    third_party/YOLOMG, and ultralytics joins its `path:` key onto `train:`/`val:` -- so a
    relative path written into a yaml or an image list resolves against the wrong root.
    Writing them relative produced 'Dataset not found.' for the competitor and a
    doubled-up '.../leak_ctl/work/ext_datasets/leak_ctl/val.txt' for ours.
    """
    dataset = dataset.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if style == "yolomg":
        # Two parallel streams: images/ is RGB, images2/ is the motion mask. The lists
        # must stay index-aligned, so both are permuted by the SAME shuffle.
        names = ["train", "val"] + (["test"] if pool == "all-clips" else [])
        rgb = [ln for n in names for ln in _read_list(dataset / f"{n}.txt")]
        msk = [ln for n in names for ln in _read_list(dataset / f"{n}2.txt")]
        if len(rgb) != len(msk):
            raise SystemExit(f"stream length mismatch: {len(rgb)} rgb vs {len(msk)} mask")
        # Pair before shuffling so a frame's mask can never be separated from its image.
        pairs = list(zip(rgb, msk))
        for a, b in pairs:
            if Path(a).name != Path(b).name:
                raise SystemExit(f"stream misalignment: {a} vs {b}")
    else:
        names = ["train", "val"]
        rgb = [str(p) for n in names
               for p in sorted((dataset / "images" / n).glob("*.jpg"))]
        pairs = [(p, None) for p in rgb]

    rng = random.Random(seed)
    if style == "ours":
        # Our images are TILES -- 'Clip_001_00056_3.jpg' is tile 3 of frame 56. Shuffling
        # them flat would split tiles of one frame across train and val, which leaks a
        # second way that the competitor's frame-level arm does not. Keeping a frame's
        # tiles together makes both arms measure exactly one thing: frame-level leakage.
        # (It is also the harder setting for us, so it cannot flatter our arm.)
        by_frame: dict[str, list] = {}
        for a, b in pairs:
            by_frame.setdefault(Path(a).stem.rsplit("_", 1)[0], []).append((a, b))
        groups = sorted(by_frame.values(), key=lambda g: g[0][0])
        rng.shuffle(groups)
        cut_g = int(len(groups) * train_frac)
        tr = [p for g in groups[:cut_g] for p in g]
        va = [p for g in groups[cut_g:] for p in g]
    else:
        rng.shuffle(pairs)                   # THE leak: a flat shuffle over frames
        cut = int(len(pairs) * train_frac)
        tr, va = pairs[:cut], pairs[cut:]

    (out / "train.txt").write_text("\n".join(a for a, _ in tr) + "\n")
    (out / "val.txt").write_text("\n".join(a for a, _ in va) + "\n")
    if style == "yolomg":
        (out / "train2.txt").write_text("\n".join(b for _, b in tr) + "\n")
        (out / "val2.txt").write_text("\n".join(b for _, b in va) + "\n")
        # Their loader also wants test/test2 keys present; point them at val so the yaml
        # is loadable. Nothing in this experiment reads them -- and note that pointing
        # them at val is exactly what their own empty test.txt effectively amounts to.
        (out / "test.txt").write_text("\n".join(a for a, _ in va) + "\n")
        (out / "test2.txt").write_text("\n".join(b for _, b in va) + "\n")
        (out / "nps.yaml").write_text(
            f"train: {out}/train.txt\ntrain2: {out}/train2.txt\n"
            f"val: {out}/val.txt\nval2: {out}/val2.txt\n"
            f"test: {out}/test.txt\ntest2: {out}/test2.txt\n\nnc: 1\nnames: ['Drone']\n")
    else:
        ### No `path:` key. Ultralytics joins `path` onto `train`/`val`, so supplying both
        ### an absolute path and absolute list files is one redundancy too many; the list
        ### files are absolute and that is sufficient.
        (out / "data.yaml").write_text(
            f"train: {out}/train.txt\nval: {out}/val.txt\n"
            "names:\n  0: drone\n")

    ### Verify the lists point at files that exist, here, rather than discovering it from
    ### a GPU job's traceback 15 seconds after it allocated a 4090. Sampling the ends and
    ### the middle catches both a wrong root and a partially-built source dataset.
    for label, rows in (("train", tr), ("val", va)):
        if not rows:
            raise SystemExit(f"{label} split is empty")
        for idx in {0, len(rows) // 2, len(rows) - 1}:
            img = Path(rows[idx][0])
            if not img.is_absolute():
                raise SystemExit(f"{label}[{idx}] is not absolute: {img}")
            if not img.exists():
                raise SystemExit(f"{label}[{idx}] does not exist: {img}")
            lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
            if not lbl.exists():
                raise SystemExit(f"label missing for {label}[{idx}]: {lbl}")
            if rows[idx][1] is not None and not Path(rows[idx][1]).exists():
                raise SystemExit(f"mask missing for {label}[{idx}]: {rows[idx][1]}")

    tr_clips = {_clip_of(a) for a, _ in tr}
    va_clips = {_clip_of(a) for a, _ in va}
    audit = {
        "dataset": str(dataset), "style": style, "pool": pool, "seed": seed,
        "train_frac": train_frac,
        "n_train": len(tr), "n_val": len(va), "n_total": len(pairs),
        "train_clips": len(tr_clips), "val_clips": len(va_clips),
        # The number that makes the leak concrete: clips appearing on BOTH sides.
        "clips_in_both": len(tr_clips & va_clips),
        "val_clips_unseen_in_train": sorted(va_clips - tr_clips),
    }
    (out / "SPLIT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--style", required=True, choices=("yolomg", "ours"))
    ap.add_argument("--pool", default="disjoint-clips",
                    choices=("disjoint-clips", "all-clips"),
                    help="disjoint-clips: re-partition ONLY the clips the disjoint arm "
                         "already trained and validated on, so the leak arm gets no new "
                         "footage and in fact less training data. all-clips: reproduce "
                         "their script exactly, pooling all 50.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.85,
                    help="0.85 is their train_percent")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    audit = build(a.dataset, a.style, a.pool, a.seed, a.out, a.train_frac)
    print(json.dumps(audit, indent=2))
    print(f"\n{audit['clips_in_both']} clips appear in BOTH train and val "
          f"({audit['n_train']} train / {audit['n_val']} val frames).")
    if audit["clips_in_both"]:
        print("That is the leak, stated as a count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
