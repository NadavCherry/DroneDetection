"""Run all realtime pipelines on 07_05 (val) and 10_06 (test), evaluate,
benchmark, and emit the comparison tables."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from realtime.pipelines import ClassicPipeline, FullFramePipeline, VerifiedPipeline
from realtime.rt_models import Detector, Expert
from realtime.runner import run_pipeline

M = "realtime/work/models"
OUT = Path("realtime/work/out")


def build(name: str):
    if name == "rt-a-classic":
        return ClassicPipeline()
    if name == "rt-b-verify256":
        return VerifiedPipeline(
            "rt-b-verify256",
            Detector(f"{M}/verifier_n256.engine", 256, fixed_batch=8),
            expert=Expert("work/models/yolo-ft-best.pt"), expert_every=30)
    if name == "rt-c-full1280":
        return FullFramePipeline(
            "rt-c-full1280", Detector(f"{M}/full_temporal_n1280.engine", 1280),
            temporal=True)
    if name == "rt-d-full640":
        return FullFramePipeline(
            "rt-d-full640", Detector(f"{M}/full_temporal_n640.engine", 640),
            temporal=True)
    if name == "rt-e-decimated":
        return VerifiedPipeline(
            "rt-e-decimated",
            Detector(f"{M}/verifier_n256.engine", 256, fixed_batch=8),
            expert=Expert("work/models/yolo-ft-best.pt"), expert_every=30,
            verify_every=2)
    if name == "rt-f-single1280":
        return FullFramePipeline(
            "rt-f-single1280", Detector(f"{M}/full_single_n1280.engine", 1280),
            temporal=False)
    raise ValueError(name)


PIPES = ["rt-a-classic", "rt-b-verify256", "rt-c-full1280", "rt-d-full640",
         "rt-e-decimated", "rt-f-single1280"]


def main() -> None:
    only = set(sys.argv[1:]) or set(PIPES)
    for video, tag in (("data/videos/07_05.mp4", "0705"), ("data/videos/10_06.mp4", "1006")):
        for name in [p for p in PIPES if p in only]:
            pipe = build(name)
            run_pipeline(video, pipe, OUT / tag / name)

    # evaluation
    py = ".venv/bin/python"
    dets05 = [str(OUT / "0705" / n / "dets.json") for n in PIPES]
    subprocess.run([py, "-m", "dronedet", "eval", "--gt", "work/gt_user.json",
                    "--dets", *dets05, "--frames", "342:571",
                    "--out", "realtime/work/eval_0705_val.md"], check=True)
    dets06 = [str(OUT / "1006" / n / "dets.json") for n in PIPES]
    subprocess.run([py, "-m", "dronedet", "eval", "--gt",
                    "realtime/work/gt_1006.json", "--dets", *dets06,
                    "--out", "realtime/work/eval_1006_test.md"], check=True)

    # bench table
    rows = []
    for name in PIPES:
        b = json.loads((OUT / "1006" / name / "bench.json").read_text())
        rows.append((name, b))
    lines = ["| pipeline | fps (5070) | total ms | stage breakdown |", "|---|---|---|---|"]
    for name, b in rows:
        total = b.get("TOTAL ms/frame", 0)
        stages = ", ".join(f"{k} {v:.1f}" for k, v in sorted(b.items())
                           if k not in ("TOTAL ms/frame", "fps_end_to_end"))
        lines.append(f"| {name} | {b['fps_end_to_end']} | {total:.1f} | {stages} |")
    Path("realtime/work/bench.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
