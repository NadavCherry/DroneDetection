"""Latency and FPS for both arms, measured the same way on the same GPU.

WHY NOT JUST CITE THEIR FPS
---------------------------
YOLOMG publishes an FPS column for every baseline (133 for YOLOv5, 5 for TransVisDrone,
133/35 for itself at 640/1280) measured on an RTX 2080 Ti. Those numbers are fine for
their table and useless for ours: different GPU, different batch, different precision,
different definition of "a frame". Putting our measured number next to their published
one in the same column would be the speed equivalent of comparing AP across protocols,
which this project already refuses to do for accuracy.

So both arms are measured here, in one process, on one GPU, and the numbers only ever
appear beside numbers produced by this file.

WHAT "ONE FRAME" MEANS FOR EACH ARM
-----------------------------------
This is the part that decides whether the comparison is honest, because the two detectors
do different amounts of work per frame and both of them do work outside the network:

  ours      K tiles of 640 px per full frame (K depends on resolution), plus ego
            stabilisation of the two taps. The tiles are ONE frame, so the reported
            ms/frame is the whole K-tile forward, not per tile.
  YOLOMG    one 1280 px forward over two streams, plus building the mask32 -- which is
            two KLT + RANSAC homography estimates on the CPU.

The preprocessing is included for both, reported separately as well as in the total.
Excluding it would flatter whichever method has the more expensive front end, and for
these two methods that is not the same method at both ends of the pipeline.

Reported as p50/p95/p99 over N timed frames after a warm-up, never as a bare mean: the
tail is what a real-time claim lives or dies on, and a mean hides a periodic stall.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def _sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _summarise(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]  # noqa: E731
    return {"n": len(s), "p50_ms": round(q(0.50), 3), "p95_ms": round(q(0.95), 3),
            "p99_ms": round(q(0.99), 3), "mean_ms": round(st.fmean(s), 3),
            "fps_at_p50": round(1000.0 / q(0.50), 1) if q(0.50) > 0 else None}


def bench_ours(weights: str, imgsz: int, tiles: int, half: bool, n: int, warmup: int):
    """Ultralytics path: K tiles in one batched forward = one frame."""
    import torch
    from ultralytics import YOLO

    model = YOLO(weights)
    imgs = [np.random.randint(0, 255, (imgsz, imgsz, 3), np.uint8) for _ in range(tiles)]
    for _ in range(warmup):
        model(imgs, imgsz=imgsz, verbose=False, device=0, half=half)
    _sync(torch)

    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(imgs, imgsz=imgsz, verbose=False, device=0, half=half)
        _sync(torch)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"arm": "ours", "weights": weights, "imgsz": imgsz, "tiles_per_frame": tiles,
            "half": half, "engine": Path(weights).suffix, **_summarise(samples)}


def bench_yolomg(weights: str, imgsz: int, frame_hw: tuple[int, int], half: bool,
                 n: int, warmup: int):
    """Competitor path: mask32 construction (CPU) + one dual-stream forward."""
    import torch

    sys.path.insert(0, str(REPO / "third_party" / "YOLOMG"))
    from models.experimental import attempt_load

    from tools.sota.infer_yolomg import _letterbox, _to_tensor
    from tools.sota.motion_mask import fd5_mask

    device = torch.device("cuda:0")
    model = attempt_load(weights, map_location=device).eval()
    if half:
        model.half()

    h, w = frame_hw
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(5)]

    pre, fwd, tot = [], [], []
    for i in range(warmup + n):
        t0 = time.perf_counter()
        mask = np.clip(fd5_mask(frames[0], frames[2], frames[4]), 0, 255).astype(np.uint8)
        import cv2
        lb_rgb, *_ = _letterbox(frames[2], imgsz)
        lb_msk, *_ = _letterbox(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), imgsz)
        x1, x2 = _to_tensor(lb_rgb, device), _to_tensor(lb_msk, device)
        if half:
            x1, x2 = x1.half(), x2.half()
        _sync(torch)
        t1 = time.perf_counter()
        with torch.no_grad():
            model(x1, x2)
        _sync(torch)
        t2 = time.perf_counter()
        if i >= warmup:
            pre.append((t1 - t0) * 1000.0)
            fwd.append((t2 - t1) * 1000.0)
            tot.append((t2 - t0) * 1000.0)

    out = {"arm": "yolomg", "weights": weights, "imgsz": imgsz, "frame_hw": list(frame_hw),
           "half": half, **_summarise(tot)}
    out["preprocess"] = _summarise(pre)     # mask32: 2x KLT + RANSAC, on the CPU
    out["forward_only"] = _summarise(fwd)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=("ours", "yolomg"))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tiles", type=int, default=6,
                    help="ours: tiles per full frame; one batched forward is ONE frame")
    ap.add_argument("--frame-hw", type=int, nargs=2, default=(1080, 1920),
                    help="yolomg: source frame size the mask is built at")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    if a.arm == "ours":
        r = bench_ours(a.weights, a.imgsz, a.tiles, a.half, a.n, a.warmup)
    else:
        r = bench_yolomg(a.weights, a.imgsz, tuple(a.frame_hw), a.half, a.n, a.warmup)

    try:
        import torch
        r["gpu"] = torch.cuda.get_device_name(0)
        r["torch"] = torch.__version__
    except Exception:
        pass

    print(json.dumps(r, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
