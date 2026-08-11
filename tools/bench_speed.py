#!/usr/bin/env python3
"""Benchmark detector inference speed: raw forward throughput at the batch sizes
the SAHI pipeline actually uses (K 640-px tiles per frame), for a .pt model or an
exported TensorRT .engine. Reports ms/frame and fps per configuration.

  python tools/bench_speed.py --weights work/runs/combined-n-p2-640/weights/best.pt
  python tools/bench_speed.py --weights ....engine            # TensorRT
  python tools/bench_speed.py --weights ...best.pt --export-trt   # export then bench
"""
import argparse
import time

import numpy as np


def bench(model, tiles, imgsz, device, half, n=60, warmup=15):
    import torch
    imgs = [np.random.randint(0, 255, (imgsz, imgsz, 3), np.uint8) for _ in range(tiles)]
    for _ in range(warmup):
        model(imgs, imgsz=imgsz, verbose=False, device=device, half=half)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(imgs, imgsz=imgsz, verbose=False, device=device, half=half)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0   # ms per frame (one K-tile forward)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--export-trt", action="store_true", help="export .pt -> TensorRT FP16 engine, then bench it")
    a = ap.parse_args()
    from ultralytics import YOLO

    weights = a.weights
    half = a.half
    if a.export_trt:
        print(f"exporting {weights} -> TensorRT FP16 @ imgsz {a.imgsz} ...", flush=True)
        eng = YOLO(weights).export(format="engine", half=True, imgsz=a.imgsz,
                                   dynamic=True, batch=16, device=a.device, verbose=False)
        weights = str(eng)
        half = True
        print("engine:", weights)

    model = YOLO(weights)
    tag = "TensorRT-FP16" if weights.endswith(".engine") else ("PyTorch-FP16" if half else "PyTorch-FP32")
    print(f"\n== {weights}  [{tag}] imgsz={a.imgsz} ==")
    print("| frame type | tiles/frame | ms/frame | fps |")
    print("|---|---|---|---|")
    for name, tiles in [("full-frame", 1), ("SAHI 1280p-frame", 6), ("SAHI 1920p-frame", 15)]:
        ms = bench(model, tiles, a.imgsz, a.device, half)
        print(f"| {name} | {tiles} | {ms:.1f} | {1000/ms:.1f} |")


if __name__ == "__main__":
    main()
