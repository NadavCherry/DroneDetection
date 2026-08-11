"""Evaluate the sliced (ft2) and copy-paste (ft3) fine-tunes end-to-end."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

PY = ".venv/bin/python"


def sh(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def rename_method(path: str, new_name: str) -> None:
    p = Path(path)
    d = json.loads(p.read_text())
    d["method"] = new_name
    p.write_text(json.dumps(d))


def run_variant(tag: str, weights: str) -> list[str]:
    outs = []
    # SAHI-style tiled inference at the exact training scale (640 slices)
    out = f"work/det/{tag}-sahi.json"
    sh(PY, "-m", "dronedet", "detect", "--video", "data/videos/07_05.mp4",
       "--method", "yolo-ft-sahi", "--weights", weights, "--out", out)
    rename_method(out, f"{tag}-sahi")
    outs.append(out)
    # hybrid with 1:1-scale verification crops (640 window, no zoom)
    out = f"work/det/{tag}-hybrid.json"
    sh(PY, "-m", "dronedet", "detect", "--video", "data/videos/07_05.mp4",
       "--method", "yolo-ft-hybrid", "--weights", weights,
       "--method-kw", '{"crop_half": 320, "zoom": 1}', "--out", out)
    rename_method(out, f"{tag}-hybrid")
    outs.append(out)
    return outs


def main() -> None:
    ft3_run = Path("work/runs/ft3-p2-640-cp")
    shutil.copy(ft3_run / "weights/best.pt", "work/models/yolo-ft3-best.pt")
    shutil.copy(ft3_run / "results.csv", "work/models/ft3_results.csv")

    dets = []
    dets += run_variant("ft2", "work/models/yolo-ft2-best.pt")
    dets += run_variant("ft3", "work/models/yolo-ft3-best.pt")

    all_dets = [f"work/det/{m}.json" for m in
                ["yolo-640", "yolo-1280", "yolo-sahi", "motion-median",
                 "motion-mog2", "hybrid", "yolo-ft", "yolo-ft-hybrid"]] + dets

    sh(PY, "-m", "dronedet", "eval", "--gt", "work/gt.json",
       "--dets", *all_dets, "--frames", "342:571", "--out", "work/eval_val.md")
    sh(PY, "-m", "dronedet", "eval", "--gt", "work/gt.json",
       "--dets", *all_dets, "--out", "work/eval_full.md")

    # trackers on the strongest ft3 detector, both operating modes
    sh(PY, "-m", "dronedet", "track", "--video", "data/videos/07_05.mp4",
       "--dets", "work/det/ft3-hybrid.json",
       "--out", "work/tracks/ft3-hybrid-all.json", "--min-score", "0.2")
    sh(PY, "-m", "dronedet", "track", "--video", "data/videos/07_05.mp4",
       "--dets", "work/det/ft3-hybrid.json",
       "--out", "work/tracks/ft3-hybrid-confirmed.json", "--min-score", "0.55")
    sh(PY, "tools/eval_tracks.py", "--tracks",
       "work/tracks/ft3-hybrid-all.json", "work/tracks/ft3-hybrid-confirmed.json")

    sh(PY, "-m", "dronedet", "render", "--video", "data/videos/07_05.mp4",
       "--dets", "work/det/ft3-hybrid.json",
       "--out", "work/vis/ft3_hybrid_dets.mp4", "--min-score", "0.5")
    sh(PY, "-c",
       "from dronedet.render import render_tracks; "
       "render_tracks('data/videos/07_05.mp4', 'work/tracks/ft3-hybrid-confirmed.json', "
       "'work/vis/ft3_hybrid_tracks.mp4')")


if __name__ == "__main__":
    main()
