"""Export the RT nano models to TensorRT FP16 engines + ONNX (CPU proxy)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

MODELS = Path("realtime/work/models")
RUNS = Path("work/runs")

JOBS = [
    # (run name, out stem, imgsz, batch)
    ("rt-ft8-n256-v2", "verifier_n256", 256, 8),
    ("rt-ftC-n1280-v2", "full_temporal_n1280", 1280, 1),
    ("rt-ftC-n1280-v2", "full_temporal_n640", 640, 1),
    ("rt-ftF-n1280", "full_single_n1280", 1280, 1),
]


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    for run, stem, imgsz, batch in JOBS:
        best = RUNS / run / "weights/best.pt"
        pt = MODELS / f"{stem}.pt"
        shutil.copy(best, pt)
        m = YOLO(str(pt))
        eng = m.export(format="engine", half=True, imgsz=imgsz, batch=batch,
                       device=0, verbose=False)
        shutil.move(eng, MODELS / f"{stem}.engine")
        m2 = YOLO(str(pt))
        onnx = m2.export(format="onnx", imgsz=imgsz, batch=batch, verbose=False)
        shutil.move(onnx, MODELS / f"{stem}.onnx")
        print(f"{stem}: engine + onnx exported (imgsz {imgsz}, batch {batch})")


if __name__ == "__main__":
    main()
