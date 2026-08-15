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
`ardmav-official` protocol -- **IoU 0.5** on the published 15-video split, which is GLAD's
(arXiv 2312.11008: "We set the intersection over union (IOU) threshold between predictions
and ground truths to 0.5"), where the bar is AP 0.80 overall and 0.58 on its small-MAV
condition. It is NOT MGMD's IoU 0.25 -- that number is on a split MGMD never enumerates.

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
from dronedet.console import use_utf8_stdio  # noqa: E402

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


def centre_in_frame(det: Detection, w: int, h: int) -> bool:
    """Is this detection inside the ORIGINAL frame, after undoing stabilisation?

    Stabilisation warps each frame onto a same-sized canvas, so the region vacated by the
    pan is filled with a constant and carries a hard straight edge that a detector trained
    on real pixels will happily fire on. Undoing the shift then places those boxes outside
    the frame: measured on phantom05, detections at x = 2091 on a 1920 px frame, 171 px
    into a border containing no scene at all.

    Ground truth lives in original-frame coordinates and inside the frame, so every one of
    these is a false positive manufactured by our own preprocessing. On a benchmark where
    precision is the scarce resource they cost AP with nothing to show for it.
    """
    return 0.0 <= det.cx < w and 0.0 <= det.cy < h


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

    # Imported from the builder so training and inference cannot drift apart: the stack
    # must be assembled by the SAME code that made the training tiles, or the model is
    # shown a representation at test time that it was never trained on.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_mde", Path(__file__).resolve().parent / "make_dataset_external.py")
    mde = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mde
    spec.loader.exec_module(mde)

    from dronedet.stabilize import Stabilizer
    stab = Stabilizer("translation")
    buf: deque = deque(maxlen=2 * dt + 1)
    cap = cv2.VideoCapture(str(video))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        m = stab.update(frame)
        buf.append((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    float(m[0, 2]), float(m[1, 2])))
        # Taps are aligned to the CURRENT frame, so the stack is already in original-frame
        # coordinates and detections need no un-shifting -- which is also why the
        # stabilisation border is now bounded by dt frames of camera motion rather than by
        # the whole video's accumulated drift.
        yield idx, np.dstack(mde._stack_aligned_to_now(buf, dt))
        idx += 1
    cap.release()


def run_video(model, video: Path, mode: str, args) -> DetectionSet:
    ds = DetectionSet(video=video.name, method=f"tiled-{mode}",
                      meta={"mode": mode, "tile": args.tile, "overlap": args.overlap,
                            "conf": args.conf, "dt": args.dt, "weights": args.weights})
    origins = None
    dx = dy = 0.0          # both modes now yield original-frame coordinates
    for idx, img in frame_source(video, mode, args.dt):
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
                                verbose=False, max_det=args.max_det, half=args.half)
            for r, (ox, oy) in zip(res, origins[i:i + args.batch]):
                for b in r.boxes:
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                    d = Detection(x1 + ox - dx, y1 + oy - dy,
                                  x2 + ox - dx, y2 + oy - dy,
                                  float(b.conf[0]), r.names[int(b.cls[0])])
                    if centre_in_frame(d, w, h):     # see the docstring: border artefacts
                        found.append(d)
        ds.add(idx, merge_by_centre(found, args.merge_dist))
    return ds


#: Container extensions seen across the corpora this repo scores. ARD-MAV ships .mp4 and
#: NPS ships .mov, and case varies between releases -- `Clip_5.mov` vs `Clip_005.MOV`.
#: Hard-coding ".mp4" here cost six NPS scorecards that all read AP = 0.000.
_VIDEO_EXTS = (".mp4", ".mov", ".MOV", ".MP4", ".avi", ".AVI", ".m4v", ".mkv")


def _resolve_video(root: Path, stem: str) -> Path | None:
    """The video for a GT stem, whatever container and case it happens to use."""
    for ext in _VIDEO_EXTS:
        p = root / f"{stem}{ext}"
        if p.exists():
            return p
    # Some NPS releases drop the zero padding: Clip_041 on disk as Clip_41.
    if "_" in stem:
        head, _, tail = stem.rpartition("_")
        if tail.isdigit():
            for cand in (f"{head}_{int(tail)}", f"{head}_{int(tail):03d}"):
                for ext in _VIDEO_EXTS:
                    p = root / f"{cand}{ext}"
                    if p.exists():
                        return p
    hits = sorted(root.glob(f"{stem}.*"))
    return hits[0] if hits else None


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mode", required=True, choices=["temporal", "rgb"])
    ap.add_argument("--gt-dir", required=True, help="per-video GT jsons; names the videos")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--video-root", default="data/external/ard_mav/ARD-MAV/videos")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--conf", type=float, default=0.05,
                    help="score floor. Low on purpose -- AP integrates over the curve and "
                         "a high floor truncates the low-precision tail AP is made of -- "
                         "but not arbitrarily low: at 0.01 ultralytics' internal NMS hit "
                         "its 2.4 s time limit and gave up mid-frame, which silently drops "
                         "boxes and makes the run non-deterministic. 0.05 measured clean")
    ap.add_argument("--merge-dist", type=float, default=6.0)
    ap.add_argument("--max-det", type=int, default=30,
                    help="per TILE, not per frame; 8 tiles cover a 1920x1080 frame")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--half", action="store_true",
                    help="FP16 inference. Roughly 2x on Ada and a detector this small is "
                         "not precision-limited at 6 px; off by default so the reference "
                         "number is FP32")
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

    missing = []
    for i, gt in enumerate(gts, 1):
        video = _resolve_video(Path(a.video_root), gt.stem)
        if video is None:
            print(f"  [{i}/{len(gts)}] {gt.stem}: MISSING under {a.video_root}")
            missing.append(gt.stem)
            continue
        t0 = time.time()
        ds = run_video(model, video, a.mode, a)
        ds.save(out_dir / f"{gt.stem}.json")
        n = sum(len(v) for v in ds.frames.values())
        print(f"  [{i}/{len(gts)}] {gt.stem}: {len(ds.frames)} frames, {n} dets, "
              f"{time.time() - t0:.0f}s", flush=True)
    print(f"wrote -> {out_dir}")

    # A missing video is not a warning. Downstream, `tools/evaluate.py` correctly scores a
    # sequence with no detections as a TOTAL MISS rather than skipping it -- so a wrong
    # --video-root does not crash, it produces a complete, plausible scorecard reading
    # AP = 0.000. That happened: NPS ships .mov and this looked only for .mp4, and six
    # runs' worth of scorecards came back at zero for a reason having nothing to do with
    # the models. Refuse to exit 0 on it.
    if missing:
        print(f"\nABORT: {len(missing)} of {len(gts)} videos could not be resolved under "
              f"{a.video_root} (e.g. {missing[:3]}). Scoring would report these as total "
              f"misses and hand you a believable AP of 0.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
