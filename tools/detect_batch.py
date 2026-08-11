#!/usr/bin/env python3
"""Run one detector over a whole corpus of test videos and dump per-video
detection jsons (basename matches the GT json, so tools/eval_external.py can pair
them). The video path is read from each GT json's "video" field.

Stateless detectors (yolo-ft full-frame / SAHI) reuse one loaded model across all
videos. Stateful ones (motion/hybrid keep a temporal background) MUST be rebuilt
per video -- pass --stateful so state never leaks between clips.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time

import numpy as np

from dronedet.methods import build_method
from dronedet.methods.base import run_method
from dronedet.methods.yolo import YoloFullFrame, YoloSahi
from dronedet.detections import DetectionSet
from dronedet.video import frames as video_frames, probe


def run_sampled(video_path, method, frame_stride, stop):
    """Detector-only runner that infers on every `frame_stride`-th frame (decodes
    all, sequentially, never seeks). No stabilization -- for stateless full-frame
    / SAHI detectors used in generalization eval. ~stride x faster inference."""
    info = probe(video_path)
    ds = DetectionSet(video=video_path, method=method.name)
    eye = np.eye(2, 3, dtype=np.float32)
    t0, n = time.perf_counter(), 0
    for idx, frame in video_frames(video_path, stop=stop):
        if idx % frame_stride:
            continue
        ds.add(idx, method.process(idx, frame, eye))
        n += 1
        if n % 100 == 0:
            print(f"  [{method.name}] frame {idx} ({n/(time.perf_counter()-t0):.1f} fps)", flush=True)
    ds.meta = {"fps_end_to_end": round(n / (time.perf_counter() - t0), 2), "n_frames": n,
               "video_fps": info.fps, "video_size": [info.width, info.height],
               "stab_mode": "off", "frame_stride": frame_stride, "shifts": {}}
    method.close()
    return ds


def make_method(a, kw):
    if a.method == "yolo-ft":                       # full frame at chosen imgsz
        return YoloFullFrame("yolo-ft", imgsz=a.imgsz, weights=a.weights,
                             conf=a.conf, drone_classes=None)
    if a.method == "yolo-ft-sahi":                  # native-res tiles (best for tiny)
        return YoloSahi("yolo-ft-sahi", tile=a.tile, overlap=a.overlap,
                        weights=a.weights, conf=a.conf, drone_classes=None)
    if a.weights:
        kw = {**kw, "weights": a.weights}
    return build_method(a.method, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True, help="dir of GT jsons; video paths read from them")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--method", default="yolo-ft")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--method-kw", default="{}")
    ap.add_argument("--stab", default="translation")
    ap.add_argument("--stateful", action="store_true")
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="infer on every Nth frame (detector-only, stab off)")
    ap.add_argument("--limit", type=int, default=None, help="only first N videos")
    a = ap.parse_args()
    kw = json.loads(a.method_kw)
    os.makedirs(a.out_dir, exist_ok=True)

    gts = sorted(glob.glob(os.path.join(a.gt_dir, "*.json")))
    if a.limit:
        gts = gts[:a.limit]
    method = None if a.stateful else make_method(a, kw)
    for i, gp in enumerate(gts):
        stem = Path(gp).stem
        out = os.path.join(a.out_dir, stem + ".json")
        if os.path.exists(out):
            print(f"[{i+1}/{len(gts)}] {stem}: exists, skip")
            continue
        video = json.loads(Path(gp).read_text())["video"]
        if not os.path.exists(video):
            print(f"[{i+1}/{len(gts)}] {stem}: MISSING VIDEO {video}")
            continue
        m = make_method(a, kw) if a.stateful else method
        print(f"[{i+1}/{len(gts)}] {stem}: {a.method} over {os.path.basename(video)}", flush=True)
        if a.frame_stride > 1:
            ds = run_sampled(video, m, a.frame_stride, a.stop)
        else:
            ds = run_method(video, m, stop=a.stop, stab_mode=a.stab)
        ds.save(out)
        if a.stateful:
            m.close()
    print("done")


if __name__ == "__main__":
    main()
