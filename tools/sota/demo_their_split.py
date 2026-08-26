#!/usr/bin/env python3
"""Run YOLOMG's own split script, unmodified, and report what it produces.

The claim this turns into evidence: their `data/split_train_val.py` cannot produce a
held-out test set, and the split it does produce is per-frame rather than per-video.
Reading the code is enough to see why, but a reader should not have to take that on
trust when the script can simply be executed.

    trainval_percent = 1.0
    tv = int(num * trainval_percent)      # == num
    trainval = random.sample(list_index, tv)   # therefore EVERY index
    ...
    if i in trainval:  ...  else:  file_test.write(name)   # never reached

So `test.txt` is created and left empty. Whatever their pipeline reports is the 15 %
`val.txt` slice -- which, because `total_xml` is a flat `os.listdir` over every frame of
every video, contains frames drawn from the same flights as `train.txt`.

This copies their script verbatim (never imports it -- it executes argparse at module
scope), points it at a directory holding the real NPS frame names, runs it, and prints
the resulting line counts plus how many videos each split touches.

    python tools/sota/demo_their_split.py --names-from work/ext_datasets/yolomg_nps
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _clip(stem: str) -> str:
    p = stem.split("_")
    return "_".join(p[:2]) if len(p) >= 2 else stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", type=Path,
                    default=REPO / "third_party/YOLOMG/data/split_train_val.py")
    ap.add_argument("--names-from", type=Path, required=True,
                    help="a built yolomg dataset; its train/val/test lists supply the "
                         "real frame filenames so the demo runs on OUR actual data")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    names: list[str] = []
    for part in ("train", "val", "test"):
        f = a.names_from / f"{part}.txt"
        if f.exists():
            names += [Path(ln).name for ln in f.read_text().split() if ln.strip()]
    if not names:
        raise SystemExit(f"no frame names found under {a.names_from}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        imgs = tmp / "images"
        imgs.mkdir()
        for n in names:
            (imgs / n).touch()          # os.listdir only needs the names to exist
        shutil.copy(a.script, tmp / "split_train_val.py")

        # Run it exactly as their README does, from the directory holding images/.
        proc = subprocess.run([sys.executable, "split_train_val.py",
                               "--xml_path", "images",
                               "--txt_path", "./ImageSets/Main"],
                              cwd=tmp, capture_output=True, text=True)
        main_dir = tmp / "ImageSets" / "Main"
        counts, clips = {}, {}
        for part in ("trainval", "train", "val", "test"):
            f = main_dir / f"{part}.txt"
            rows = [ln for ln in f.read_text().split() if ln.strip()] if f.exists() else []
            counts[part] = len(rows)
            clips[part] = len({_clip(r) for r in rows})

    report = {
        "script": str(a.script), "n_input_frames": len(names),
        "returncode": proc.returncode,
        "counts": counts, "distinct_videos_per_split": clips,
    }
    print(json.dumps(report, indent=2))
    print()
    print(f"test.txt lines: {counts['test']}  "
          f"<- their script's held-out test set")
    print(f"val.txt  lines: {counts['val']} drawn from {clips['val']} distinct videos")
    print(f"train.txt lines: {counts['train']} drawn from {clips['train']} distinct videos")
    if counts["test"] == 0:
        print("\nCONFIRMED: the test split is empty. trainval_percent = 1.0 sends every "
              "index into trainval, so the branch that writes test.txt never executes.")
    if clips["val"] and clips["val"] == clips["train"]:
        print("CONFIRMED: train and val draw from the SAME "
              f"{clips['train']} videos -- the split is per-frame, not per-video.")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
