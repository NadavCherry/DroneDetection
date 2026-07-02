"""Micro-benchmarks for the edge estimate: TRT-GPU vs ONNX-CPU per model,
plus the classical stages on CPU. Prints a table used by the README's
Jetson Orin Nano projection (stated assumptions, not measurements)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

M = "REALTIME/work/models"


def bench(fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return 1000 * (time.perf_counter() - t0) / n


def main() -> None:
    from ultralytics import YOLO

    img256 = np.random.randint(0, 255, (256, 256, 3), np.uint8)
    img1280 = np.random.randint(0, 255, (720, 1280, 3), np.uint8)

    rows = []
    trt_v = YOLO(f"{M}/verifier_n256.engine", task="detect")
    rows.append(("verifier n@256 x8 crops", "TRT FP16 (5070)",
                 bench(lambda: trt_v([img256] * 8, imgsz=256, verbose=False))))
    trt_c = YOLO(f"{M}/full_temporal_n1280.engine", task="detect")
    rows.append(("full-frame n@1280", "TRT FP16 (5070)",
                 bench(lambda: trt_c(img1280, imgsz=1280, verbose=False))))
    trt_d = YOLO(f"{M}/full_temporal_n640.engine", task="detect")
    rows.append(("full-frame n@640", "TRT FP16 (5070)",
                 bench(lambda: trt_d(img1280, imgsz=640, verbose=False))))

    onnx_v = YOLO(f"{M}/verifier_n256.onnx", task="detect")
    rows.append(("verifier n@256 x8 crops", "ONNX CPU (32-core laptop)",
                 bench(lambda: onnx_v([img256] * 8, imgsz=256, device="cpu",
                                      verbose=False), n=10)))
    onnx_c = YOLO(f"{M}/full_temporal_n1280.onnx", task="detect")
    rows.append(("full-frame n@1280", "ONNX CPU (32-core laptop)",
                 bench(lambda: onnx_c(img1280, imgsz=1280, device="cpu",
                                      verbose=False), n=10)))
    onnx_d = YOLO(f"{M}/full_temporal_n640.onnx", task="detect")
    rows.append(("full-frame n@640", "ONNX CPU (32-core laptop)",
                 bench(lambda: onnx_d(img1280, imgsz=640, device="cpu",
                                      verbose=False), n=10)))

    # classical stages (CPU)
    from REALTIME.rt_stabilize import LiteStabilizer
    from REALTIME.pipelines import make_slow_channel

    gray = cv2.cvtColor(img1280, cv2.COLOR_BGR2GRAY)
    stab = LiteStabilizer()
    stab.update(gray)
    rows.append(("stabilize (768x448 crop corr)", "CPU",
                 bench(lambda: stab.update(gray))))
    slow = make_slow_channel()
    for _ in range(45):
        slow.process(gray)
    rows.append(("lagged-median motion (amortized)", "CPU",
                 bench(lambda: slow.process(gray), n=40)))

    lines = ["| component | runtime | ms |", "|---|---|---|"]
    for name, rt, ms in rows:
        lines.append(f"| {name} | {rt} | {ms:.1f} |")
    table = "\n".join(lines)
    Path("REALTIME/work/bench_models.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
