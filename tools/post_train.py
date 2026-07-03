"""After fine-tuning: run ft detectors, evals, trackers, renders."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PY = ".venv/bin/python"
RUN_DIR = Path("/home/nadavc/PycharmProjects/TheAgency_workspace/runs/detect/work/runs/ft-p2-1280")
BEST = Path("work/models/yolo-ft-best.pt")


def sh(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    BEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUN_DIR / "weights/best.pt", BEST)
    shutil.copy(RUN_DIR / "results.csv", "work/models/ft_results.csv")

    # fine-tuned detectors
    sh(PY, "-m", "dronedet", "detect", "--video", "data/videos/07_05.mp4",
       "--method", "yolo-ft", "--weights", str(BEST),
       "--out", "work/det/yolo-ft.json")
    sh(PY, "-m", "dronedet", "detect", "--video", "data/videos/07_05.mp4",
       "--method", "yolo-ft-hybrid", "--weights", str(BEST),
       "--out", "work/det/yolo-ft-hybrid.json")

    dets = [f"work/det/{m}.json" for m in
            ["yolo-640", "yolo-1280", "yolo-sahi", "motion-median",
             "motion-mog2", "hybrid", "yolo-ft", "yolo-ft-hybrid"]]

    # like-for-like on the held-out val segment (the hardest part: 3-5 px)
    sh(PY, "-m", "dronedet", "eval", "--gt", "work/gt.json",
       "--dets", *dets, "--frames", "342:571", "--out", "work/eval_val.md")
    # full video (ft methods saw frames 0..341 during training -- caveat)
    sh(PY, "-m", "dronedet", "eval", "--gt", "work/gt.json",
       "--dets", *dets, "--out", "work/eval_full.md")

    # trackers on the strongest detector, two operating modes
    sh(PY, "-m", "dronedet", "track", "--video", "data/videos/07_05.mp4",
       "--dets", "work/det/yolo-ft-hybrid.json",
       "--out", "work/tracks/ft-hybrid-all.json", "--min-score", "0.2")
    sh(PY, "-m", "dronedet", "track", "--video", "data/videos/07_05.mp4",
       "--dets", "work/det/yolo-ft-hybrid.json",
       "--out", "work/tracks/ft-hybrid-confirmed.json", "--min-score", "0.55")
    sh(PY, "tools/eval_tracks.py", "--tracks",
       "work/tracks/ft-hybrid-all.json", "work/tracks/ft-hybrid-confirmed.json",
       "work/tracks/hybrid.json", "work/tracks/motion-median.json")

    # final annotated videos
    sh(PY, "-m", "dronedet", "render", "--video", "data/videos/07_05.mp4",
       "--dets", "work/det/yolo-ft-hybrid.json",
       "--out", "work/vis/ft_hybrid_dets.mp4", "--min-score", "0.5")
    sh(PY, "-c",
       "from dronedet.render import render_tracks; "
       "render_tracks('data/videos/07_05.mp4', 'work/tracks/ft-hybrid-confirmed.json', "
       "'work/vis/ft_hybrid_tracks.mp4')")


if __name__ == "__main__":
    main()
