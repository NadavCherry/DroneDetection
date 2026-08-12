#!/usr/bin/env python3
"""Full-frame inference for the ARD-MAV single-frame vs temporal A/B.

Why this exists rather than `--method moe3-stacked`
---------------------------------------------------
`dronedet.methods.hybrid2.Hybrid2` also runs a motion-proposal stage and only verifies
crops around its proposals. That is the right design for the shipped system and the wrong
instrument for this experiment: the two arms would then differ in their input channels
*and* in whether a motion detector chose where to look, and a gain could not be attributed.

This harness does exactly what the training builder did, and nothing else:

    temporal : channels = stabilised gray(t-2*dt), gray(t-dt), gray(t)   [matches
               `make_dataset_external.extract_yolo_tiled_temporal`]
    rgb      : channels = BGR of frame t                                [matches
               `extract_yolo_tiled`]

then tiles the full frame on a fixed overlapping grid, runs the detector on every tile,
maps boxes back to frame coordinates and merges duplicates from the overlap. Both arms
share every line of this file, so the only difference between them is what is in the three
channels -- which is the whole claim under test.

Output is a `dronedet.detections.DetectionSet` JSON per video, which is what
`tools/evaluate.py` scores against `work/ext_datasets/gt/ardmav/<video>.json` under the
`ardmav-official` protocol (IoU 0.25 on the published 15-video split, i.e. MGMD's).

    python tools/infer_tiled.py --weights work/runs/temporal_ardmav-s0/weights/best.pt \
        --mode temporal --gt-dir work/ext_datasets/gt/ardmav --out-dir work/det/temporal
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.detections import Detection, DetectionSet  # noqa: E402

TEMPORAL_DT = 6          # must match make_dataset_external.TEMPORAL_DT


def tile_origins(w: int, h: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Top-left corners of a grid covering the frame, with the last row/column pulled
    back inside the image rather than padded.

    Pulling back rather than padding matters: a padded tile puts a hard black edge in the
    image, and on a 6 px target a black edge is a strong gradient the detector has never
    seen in training, where tiles were always cut from real pixels.
    """
    step = max(1, tile - overlap)
    xs = list(range(0, max(1, w - tile + 1), step))
    ys = list(range(0, max(1, h - tile + 1), step))
    if xs[-1] != w - tile and w > tile:
        xs.append(w - tile)
    if ys[-1] != h - tile and h > tile:
        ys.append(h - tile)
    return [(x, y) for y in ys for x in xs]


def merge_by_centre(dets: list[Detection], dist: float = 6.0) -> list[Detection]:
    """Greedy merge of near-coincident centres, highest score first.

    Centre distance rather than IoU on purpose. The overlap region duplicates a target
    across two tiles, and on a 6 px box two detections of the same object routinely score
    IoU below 0.5 -- an IoU-based NMS would keep both and invent a false positive out of
    the tiling itself.
    """
    out: list[Detection] = []
    for d in sorted(dets, key=lambda x: -x.score):
        if all((d.cx - k.cx) ** 2 + (d.cy - k.cy) ** 2 > dist * dist for k in out):
            out.append(d)
    return out


def frame_source(video: Path, mode: str, dt: int):
    """Yield (index, HxWx3 uint8) in the representation the model was trained on."""
    if mode == "rgb":
        cap = cv2.VideoCapture(str(video))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
        cap.release()
        return

    from dronedet.stabilize import Stabilizer, warp_to_reference
    stab = Stabilizer("translation")
    buf: deque = deque(maxlen=2 * dt + 1)
    cap = cv2.VideoCapture(str(video))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        m = stab.update(frame)
        buf.append(warp_to_reference(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), m))
        n = len(buf)
        taps = [buf[max(0, n - 1 - 2 * dt)], buf[max(0, n - 1 - dt)], buf[n - 1]]
        # Boxes come back in stabilised coordinates; undo the shift so detections are
        # reported in the ORIGINAL frame, which is where the ground truth lives.
        yield (idx, np.dstack(taps), float(m[0, 2]), float(m[1, 2]))
        idx += 1
    cap.release()


def run_video(model, video: Path, mode: str, args) -> DetectionSet:
    ds = DetectionSet(video=video.name, method=f"tiled-{mode}",
                      meta={"mode": mode, "tile": args.tile, "overlap": args.overlap,
                            "conf": args.conf, "dt": args.dt, "weights": args.weights})
    origins = None
    for item in frame_source(video, mode, args.dt):
        if mode == "rgb":
            idx, img = item
            dx = dy = 0.0
        else:
            idx, img, dx, dy = item
        if args.stop and idx >= args.stop:
            break
        h, w = img.shape[:2]
        if origins is None:
            origins = tile_origins(w, h, args.tile, args.overlap)
        crops = [img[y:y + args.tile, x:x + args.tile] for x, y in origins]
        found: list[Detection] = []
        for i in range(0, len(crops), args.batch):
            chunk = crops[i:i + args.batch]
            res = model.predict(chunk, imgsz=args.tile, conf=args.conf, device=args.device,
                                verbose=False, max_det=args.max_det)
            for r, (ox, oy) in zip(res, origins[i:i + args.batch]):
                for b in r.boxes:
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                    found.append(Detection(x1 + ox - dx, y1 + oy - dy,
                                           x2 + ox - dx, y2 + oy - dy,
                                           float(b.conf[0]),
                                           r.names[int(b.cls[0])]))
        ds.add(idx, merge_by_centre(found, args.merge_dist))
    return ds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mode", required=True, choices=["temporal", "rgb"])
    ap.add_argument("--gt-dir", required=True, help="per-video GT jsons; names the videos")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--video-root", default="data/external/ard_mav/ARD-MAV/videos")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--conf", type=float, default=0.01,
                    help="low on purpose: AP integrates over the curve, and a high floor "
                         "silently truncates the low-precision tail that AP is made of")
    ap.add_argument("--merge-dist", type=float, default=6.0)
    ap.add_argument("--max-det", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--dt", type=int, default=TEMPORAL_DT)
    ap.add_argument("--stop", type=int, default=0, help="first N frames only (smoke test)")
    ap.add_argument("--limit", type=int, default=0, help="first N videos only")
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(a.weights)

    gts = sorted(Path(a.gt_dir).glob("*.json"))
    if a.limit:
        gts = gts[:a.limit]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(gts)} videos, mode={a.mode}, tile={a.tile}, overlap={a.overlap}")

    for i, gt in enumerate(gts, 1):
        video = Path(a.video_root) / f"{gt.stem}.mp4"
        if not video.exists():
            print(f"  [{i}/{len(gts)}] {gt.stem}: MISSING {video}")
            continue
        t0 = time.time()
        ds = run_video(model, video, a.mode, a)
        ds.save(out_dir / f"{gt.stem}.json")
        n = sum(len(v) for v in ds.frames.values())
        print(f"  [{i}/{len(gts)}] {gt.stem}: {len(ds.frames)} frames, {n} dets, "
              f"{time.time() - t0:.0f}s", flush=True)
    print(f"wrote -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
