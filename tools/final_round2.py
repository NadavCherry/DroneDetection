"""Round 2 (user-GT) finale: new methods, full comparison, trackers,
tracker-feedback method, and videos painted with the user's labels."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

PY = ".venv/bin/python"
FT1 = "work/models/yolo-ft-best.pt"    # near/big expert
FT6 = "work/models/yolo-ft6-best.pt"   # 2-class single-frame tiny specialist
FT7 = "work/models/yolo-ft7-best.pt"   # 2-class temporal (stacked) specialist
GT = "work/gt_user.json"


def sh(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def rename(path: str, new_name: str) -> None:
    p = Path(path)
    d = json.loads(p.read_text())
    d["method"] = new_name
    p.write_text(json.dumps(d))


def main() -> None:
    run = Path("work/runs/ft7-p2-640-stacked")
    shutil.copy(run / "weights/best.pt", FT7)
    shutil.copy(run / "results.csv", "work/models/ft7_results.csv")

    kw2 = json.dumps({"weights": FT6, "full_weights": FT1})
    sh(PY, "-m", "dronedet", "detect", "--video", "07_05.mp4",
       "--method", "moe2-hybrid", "--method-kw", kw2,
       "--out", "work/det2/moe2-hybrid.json")

    kw3 = json.dumps({"weights": FT7, "full_weights": FT1})
    sh(PY, "-m", "dronedet", "detect", "--video", "07_05.mp4",
       "--method", "moe3-stacked", "--method-kw", kw3,
       "--out", "work/det2/moe3-stacked.json")

    kwsr = json.dumps({"sr_model_path": "work/models/FSRCNN_x4.pb"})
    sh(PY, "-m", "dronedet", "detect", "--video", "07_05.mp4",
       "--method", "sr-hybrid", "--method-kw", kwsr,
       "--out", "work/det2/sr-hybrid.json")

    # trackers on both new hybrids; feedback method from the better one
    for tag in ("moe2-hybrid", "moe3-stacked"):
        for mode, score in (("all", "0.2"), ("confirmed", "0.55")):
            sh(PY, "-m", "dronedet", "track", "--video", "07_05.mp4",
               "--dets", f"work/det2/{tag}.json",
               "--out", f"work/tracks2/{tag}-{mode}.json", "--min-score", score)
    sh(PY, "tools/eval_tracks.py", "--gt", GT, "--tracks",
       "work/tracks2/moe2-hybrid-all.json", "work/tracks2/moe2-hybrid-confirmed.json",
       "work/tracks2/moe3-stacked-all.json", "work/tracks2/moe3-stacked-confirmed.json")

    sh(PY, "tools/tracks_to_dets.py", "--tracks", "work/tracks2/moe3-stacked-all.json",
       "--src", "work/det2/moe3-stacked.json",
       "--out", "work/det2/tracked-moe3.json", "--name", "tracked-moe3")

    old = [f"work/det/{m}.json" for m in
           ["yolo-640", "yolo-1280", "yolo-sahi", "motion-median", "motion-mog2",
            "hybrid", "yolo-ft-hybrid", "ft5-sahi", "moe-hybrid"]]
    new = [f"work/det2/{m}.json" for m in
           ["motion-slow", "sr-hybrid", "moe2-hybrid", "moe3-stacked", "tracked-moe3"]]

    sh(PY, "-m", "dronedet", "eval", "--gt", GT, "--dets", *(old + new),
       "--frames", "342:571", "--out", "work/eval_user_val.md")
    sh(PY, "-m", "dronedet", "eval", "--gt", GT, "--dets", *(old + new),
       "--out", "work/eval_user_full.md")

    # videos painted with the user's labels
    sh(PY, "-c",
       "from dronedet.render import render_detections; "
       f"render_detections('07_05.mp4', 'work/det2/moe3-stacked.json', "
       f"'work/vis2/round2_dets.mp4', min_score=0.5, gt_path='{GT}')")
    sh(PY, "-c",
       "from dronedet.render import render_tracks; "
       "render_tracks('07_05.mp4', 'work/tracks2/moe3-stacked-all.json', "
       "'work/vis2/round2_tracks.mp4')")


if __name__ == "__main__":
    main()
