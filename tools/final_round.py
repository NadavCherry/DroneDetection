"""Final round: the tiny-specialist (ft5) + mixture-of-experts hybrid.

Methods added:
  ft5-sahi    tiny specialist alone on 640 tiles (appearance-only, tiny)
  moe-hybrid  motion proposals -> ft5 verification at 1:1 scale, unioned
              with the near/big expert (ft1) full-frame pass

Produces the definitive eval tables, tracker runs and final videos.
"""

from __future__ import annotations

import json
import shutil
import subprocess
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


def main() -> None:
    run = Path("work/runs/ft5-p2-640-tiny")
    shutil.copy(run / "weights/best.pt", "work/models/yolo-ft5-best.pt")
    shutil.copy(run / "results.csv", "work/models/ft5_results.csv")
    w5 = "work/models/yolo-ft5-best.pt"
    w1 = "work/models/yolo-ft-best.pt"

    out = "work/det/ft5-sahi.json"
    sh(PY, "-m", "dronedet", "detect", "--video", "data/videos/07_05.mp4",
       "--method", "yolo-ft-sahi", "--weights", w5, "--out", out)
    rename_method(out, "ft5-sahi")

    out = "work/det/moe-hybrid.json"
    sh(PY, "-m", "dronedet", "detect", "--video", "data/videos/07_05.mp4",
       "--method", "yolo-ft-hybrid", "--weights", w5,
       "--method-kw",
       json.dumps({"crop_half": 320, "zoom": 1,
                   "full_weights": w1, "full_classes": None}),
       "--out", out)
    rename_method(out, "moe-hybrid")

    all_dets = [f"work/det/{m}.json" for m in
                ["yolo-640", "yolo-1280", "yolo-sahi", "motion-median",
                 "motion-mog2", "hybrid", "yolo-ft", "yolo-ft-hybrid",
                 "ft2-sahi", "ft2-hybrid", "ft3-sahi", "ft3-hybrid",
                 "ft5-sahi", "moe-hybrid"]]

    sh(PY, "-m", "dronedet", "eval", "--gt", "work/gt.json",
       "--dets", *all_dets, "--frames", "342:571", "--out", "work/eval_val.md")
    sh(PY, "-m", "dronedet", "eval", "--gt", "work/gt.json",
       "--dets", *all_dets, "--out", "work/eval_full.md")

    for mode, score in (("all", "0.2"), ("confirmed", "0.55")):
        sh(PY, "-m", "dronedet", "track", "--video", "data/videos/07_05.mp4",
           "--dets", "work/det/moe-hybrid.json",
           "--out", f"work/tracks/moe-{mode}.json", "--min-score", score)
    sh(PY, "tools/eval_tracks.py", "--tracks",
       "work/tracks/moe-all.json", "work/tracks/moe-confirmed.json")

    sh(PY, "-m", "dronedet", "render", "--video", "data/videos/07_05.mp4",
       "--dets", "work/det/moe-hybrid.json",
       "--out", "work/vis/final_dets.mp4", "--min-score", "0.5")
    sh(PY, "-c",
       "from dronedet.render import render_tracks; "
       "render_tracks('data/videos/07_05.mp4', 'work/tracks/moe-confirmed.json', "
       "'work/vis/final_tracks.mp4')")


if __name__ == "__main__":
    main()
