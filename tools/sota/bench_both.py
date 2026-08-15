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
from collections import deque
from pathlib import Path

import cv2
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


def bench_ours(weights: str, imgsz: int, frame_hw: tuple[int, int], overlap: int,
               conf: float, max_det: int, half: bool, n: int, warmup: int):
    """One full frame: stabilise, build the temporal stack, tile it, run every tile.

    The first version of this timed only `model(imgs)` on pre-made random tiles, while
    `bench_yolomg` timed its mask construction inside the total -- so the published table
    charged the competitor for its front end and us for none of ours, in a module whose
    docstring promises the opposite. That is the sharpest possible thumb on the scale in a
    latency comparison, and it was in our favour.

    Two further corrections, both the same mistake of describing rather than measuring:

      * the tile count is DERIVED from `tile_origins` at the real frame size, not passed
        in. It was passed as 6; a 1920x1080 frame is 8 tiles, so our ms/frame was understated
        by about a quarter.
      * `conf` and `max_det` match the evaluated runs (0.001 / 30 per tile). NMS cost is
        dominated by candidate count, and benchmarking at ultralytics' default 0.25 would
        measure a configuration that produced no published AP.
    """
    import torch
    from ultralytics import YOLO

    sys.path.insert(0, str(REPO / "tools"))
    from dronedet.stabilize import Stabilizer
    from tools.infer_tiled import tile_origins
    from tools.make_dataset_external import _stack_aligned_to_now

    h, w = frame_hw
    model = YOLO(weights)
    origins = tile_origins(w, h, imgsz, overlap)
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(13)]

    pre, fwd, tot = [], [], []
    for i in range(warmup + n):
        t0 = time.perf_counter()
        stab = Stabilizer("translation")
        buf: deque = deque(maxlen=13)
        for f in frames:                     # the real front end: stabilise every tap
            m = stab.update(f)
            buf.append((cv2.cvtColor(f, cv2.COLOR_BGR2GRAY),
                        float(m[0, 2]), float(m[1, 2])))
        img = np.dstack(_stack_aligned_to_now(buf, 6))
        crops = [img[y:y + imgsz, x:x + imgsz] for x, y in origins]
        _sync(torch)
        t1 = time.perf_counter()
        model(crops, imgsz=imgsz, conf=conf, max_det=max_det,
              verbose=False, device=0, half=half)
        _sync(torch)
        t2 = time.perf_counter()
        if i >= warmup:
            pre.append((t1 - t0) * 1000.0)
            fwd.append((t2 - t1) * 1000.0)
            tot.append((t2 - t0) * 1000.0)

    out = {"arm": "ours", "weights": weights, "imgsz": imgsz,
           "frame_hw": list(frame_hw), "tiles_per_frame": len(origins),
           "overlap": overlap, "conf": conf, "max_det_per_tile": max_det,
           "half": half, "engine": Path(weights).suffix, **_summarise(tot)}
    out["preprocess"] = _summarise(pre)      # stabilise 13 taps + build the stack + tile
    out["forward_only"] = _summarise(fwd)
    return out


def bench_yolomg(weights: str, imgsz: int, frame_hw: tuple[int, int], half: bool,
                 n: int, warmup: int):
    """Competitor path: mask32 construction (CPU) + one dual-stream forward."""
    import torch

    sys.path.insert(0, str(REPO / "third_party" / "YOLOMG"))
    from models.experimental import attempt_load
    from utils.general import non_max_suppression

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
            pred = model(x1, x2)[0]
            # NMS inside the timed region, at the thresholds infer_yolomg actually uses.
            # Ours goes through ultralytics' predict path, which includes its NMS; timing
            # only the raw forward here would re-introduce the asymmetry from the other side.
            non_max_suppression(pred, 0.001, 0.6, max_det=300)
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
    ap.add_argument("--overlap", type=int, default=128,
                    help="ours: tile overlap. The TILE COUNT is derived from --frame-hw "
                         "rather than passed, because passing it got it wrong: 6 was used "
                         "where a 1920x1080 frame is 8 tiles.")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="must match the evaluated runs -- NMS cost lives in the candidate "
                         "count, so benchmarking at a floor no published AP used is fiction")
    ap.add_argument("--max-det", type=int, default=30, help="ours: per TILE")
    ap.add_argument("--frame-hw", type=int, nargs=2, default=(1080, 1920),
                    help="source frame size, for BOTH arms: ours derives its tile count "
                         "from it, the competitor builds its mask at it")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    if a.arm == "ours":
        r = bench_ours(a.weights, a.imgsz, tuple(a.frame_hw), a.overlap, a.conf,
                       a.max_det, a.half, a.n, a.warmup)
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
