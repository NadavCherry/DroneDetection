"""Round 3 (v3 models) PC finale.

Pipeline: moe3-v3 (dual motion proposals -> ft7-v3 temporal verifier +
ft1 full-frame expert) UNION fullS (yolov8s-p2 @1280 on the full-frame
temporal stack) -> score-aware fusion -> tracker -> track classifier ->
tracked-pcmax. Evaluated on 07_05 val/full (hand labels) and the 10_06
hardened test reference.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PY = ".venv/bin/python"
FT1 = "work/models/yolo-ft-best.pt"
FT7V3 = "work/models/yolo-ft7v3-best.pt"
FTSV3 = "work/models/yolo-ftSv3-best.pt"
GT05 = "work/gt_user.json"
GT06 = "REALTIME/work/gt_1006_v2.json"


def sh(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def run_fullS(video: str, out: str) -> None:
    """Full-frame temporal s-model pass via the REALTIME runner."""
    from REALTIME.pipelines import FullFramePipeline
    from REALTIME.rt_models import Detector
    from REALTIME.runner import run_pipeline

    pipe = FullFramePipeline("fullS-s1280", Detector(FTSV3, 1280), temporal=True)
    run_pipeline(video, pipe, out)


def main() -> None:
    only = set(sys.argv[1:])

    def want(step: str) -> bool:
        return not only or step in only

    if want("weights"):
        shutil.copy("work/runs/v3-ft7-s640/weights/best.pt", FT7V3)
        shutil.copy("work/runs/v3-ftS-s1280/weights/best.pt", FTSV3)

    for video, tag in (("07_05.mp4", "0705"), ("10_06.mp4", "1006")):
        det_dir = Path(f"work/det3/{tag}")
        det_dir.mkdir(parents=True, exist_ok=True)
        if want("moe3"):
            kw = json.dumps({"weights": FT7V3, "full_weights": FT1})
            sh(PY, "-m", "dronedet", "detect", "--video", video,
               "--method", "moe3-stacked", "--method-kw", kw,
               "--out", str(det_dir / "moe3-v3.json"))
        if want("fulls"):
            run_fullS(video, str(det_dir / "fullS"))
            shutil.copy(det_dir / "fullS/dets.json", det_dir / "fullS-s1280.json")
        if want("fuse"):
            sh(PY, "tools/fuse_dets.py", "--dets",
               str(det_dir / "fullS-s1280.json"), str(det_dir / "moe3-v3.json"),
               "--out", str(det_dir / "pc-max.json"), "--name", "pc-max")
        if want("track"):
            tr_dir = Path(f"work/tracks3/{tag}")
            tr_dir.mkdir(parents=True, exist_ok=True)
            for src, nm in (("pc-max", "tracked-pcmax"),
                            ("moe3-v3", "tracked-moe3v3")):
                sh(PY, "-m", "dronedet", "track", "--video", video,
                   "--dets", str(det_dir / f"{src}.json"),
                   "--out", str(tr_dir / f"{src}-all.json"), "--min-score", "0.2")
                sh(PY, "tools/tracks_to_dets.py",
                   "--tracks", str(tr_dir / f"{src}-all.json"),
                   "--src", str(det_dir / f"{src}.json"),
                   "--out", str(det_dir / f"{nm}.json"),
                   "--name", nm, "--classify")

    if want("eval"):
        d05 = [f"work/det3/0705/{m}.json" for m in
               ("moe3-v3", "fullS-s1280", "pc-max", "tracked-moe3v3", "tracked-pcmax")]
        d05 += ["work/det2/moe3-stacked.json", "work/det2/tracked-moe3-cls.json"]
        sh(PY, "-m", "dronedet", "eval", "--gt", GT05, "--dets", *d05,
           "--frames", "342:571", "--out", "work/eval_round3_0705val.md")
        sh(PY, "-m", "dronedet", "eval", "--gt", GT05, "--dets", *d05,
           "--out", "work/eval_round3_0705full.md")
        d06 = [f"work/det3/1006/{m}.json" for m in
               ("moe3-v3", "fullS-s1280", "pc-max", "tracked-moe3v3", "tracked-pcmax")]
        d06 += ["work/infer/10_06/tracked-moe3-cls.json"]
        sh(PY, "-m", "dronedet", "eval", "--gt", GT06, "--dets", *d06,
           "--out", "work/eval_round3_1006test.md")
        for p in ("work/eval_round3_0705val.md", "work/eval_round3_0705full.md",
                  "work/eval_round3_1006test.md"):
            print(f"\n==== {p} ====")
            print(Path(p).read_text())


if __name__ == "__main__":
    main()
