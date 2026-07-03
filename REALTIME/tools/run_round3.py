"""Round 3 edge rebuild: export v3 engines, run RT-C/RT-D on both videos,
produce tracked+classified variants, evaluate against the hand labels
(07_05 val) and the hardened 10_06 reference, and re-bench."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

M = Path("REALTIME/work/models")
OUT = Path("REALTIME/work/out3")
PY = ".venv/bin/python"
GT05 = "work/gt_user.json"
GT06 = "REALTIME/work/gt_1006_v2.json"

PIPES = ["rt-c-full1280", "rt-d-full640"]


def sh(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def export() -> None:
    from ultralytics import YOLO

    best = "work/runs/v3-ftC-n1280/weights/best.pt"
    M.mkdir(parents=True, exist_ok=True)
    for stem, imgsz in (("v3_full_temporal_n1280", 1280),
                        ("v3_full_temporal_n640", 640)):
        # distinct .pt per stem: the engine inherits the .pt stem, so
        # sharing one .pt would overwrite the previous engine
        shutil.copy(best, M / f"{stem}.pt")
        m = YOLO(str(M / f"{stem}.pt"))
        eng = m.export(format="engine", half=True, imgsz=imgsz, batch=1,
                       device=0, verbose=False)
        if Path(eng).resolve() != (M / f"{stem}.engine").resolve():
            shutil.move(eng, M / f"{stem}.engine")
        print(f"{stem}.engine exported")
    # verifier (kept for the rt-b/e patterns + PC parity experiments)
    vb = Path("work/runs/v3-ft8-n256/weights/best.pt")
    if vb.exists():
        shutil.copy(vb, M / "v3_verifier_n256.pt")
        m = YOLO(str(M / "v3_verifier_n256.pt"))
        eng = m.export(format="engine", half=True, imgsz=256, batch=8,
                       device=0, verbose=False)
        shutil.move(eng, M / "v3_verifier_n256.engine")
        print("v3_verifier_n256.engine exported")


def build(name: str):
    from REALTIME.pipelines import FullFramePipeline
    from REALTIME.rt_models import Detector

    if name == "rt-c-full1280":
        return FullFramePipeline(
            "rt-c-full1280", Detector(f"{M}/v3_full_temporal_n1280.engine", 1280),
            temporal=True)
    if name == "rt-d-full640":
        return FullFramePipeline(
            "rt-d-full640", Detector(f"{M}/v3_full_temporal_n640.engine", 640),
            temporal=True)
    raise ValueError(name)


def main() -> None:
    only = set(sys.argv[1:])

    def want(step: str) -> bool:
        return not only or step in only

    if want("export"):
        export()

    if want("run"):
        from REALTIME.runner import run_pipeline

        for video, tag in (("07_05.mp4", "0705"), ("10_06.mp4", "1006")):
            for name in PIPES:
                pipe = build(name)
                run_pipeline(video, pipe, OUT / tag / name)

    if want("track"):
        for video, tag in (("07_05.mp4", "0705"), ("10_06.mp4", "1006")):
            for name in PIPES:
                d = OUT / tag / name
                sh(PY, "-m", "dronedet", "track", "--video", video,
                   "--dets", str(d / "dets.json"),
                   "--out", str(d / "tracks-all.json"), "--min-score", "0.2")
                sh(PY, "tools/tracks_to_dets.py",
                   "--tracks", str(d / "tracks-all.json"),
                   "--src", str(d / "dets.json"),
                   "--out", str(d / f"tracked-{name}.json"),
                   "--name", f"tracked-{name}", "--classify")

    if want("eval"):
        d05, d06 = [], []
        for name in PIPES:
            d05 += [str(OUT / "0705" / name / "dets.json"),
                    str(OUT / "0705" / name / f"tracked-{name}.json")]
            d06 += [str(OUT / "1006" / name / "dets.json"),
                    str(OUT / "1006" / name / f"tracked-{name}.json")]
        sh(PY, "-m", "dronedet", "eval", "--gt", GT05, "--dets", *d05,
           "--frames", "342:571", "--out", "REALTIME/work/eval3_0705_val.md")
        sh(PY, "-m", "dronedet", "eval", "--gt", GT06, "--dets", *d06,
           "--out", "REALTIME/work/eval3_1006_test.md")
        rows = []
        for name in PIPES:
            b = json.loads((OUT / "1006" / name / "bench.json").read_text())
            rows.append((name, b))
        lines = ["| pipeline | fps (5070) | total ms | stage breakdown |",
                 "|---|---|---|---|"]
        for name, b in rows:
            total = b.get("TOTAL ms/frame", 0)
            stages = ", ".join(f"{k} {v:.1f}" for k, v in sorted(b.items())
                               if k not in ("TOTAL ms/frame", "fps_end_to_end"))
            lines.append(f"| {name} | {b['fps_end_to_end']} | {total:.1f} | {stages} |")
        Path("REALTIME/work/bench3.md").write_text("\n".join(lines) + "\n")
        for p in ("REALTIME/work/eval3_0705_val.md",
                  "REALTIME/work/eval3_1006_test.md", "REALTIME/work/bench3.md"):
            print(f"\n==== {p} ====")
            print(Path(p).read_text())


if __name__ == "__main__":
    main()
