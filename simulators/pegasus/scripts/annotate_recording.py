#!/usr/bin/env python3
"""Turn a raw two-drone recording into something a human can actually read.

A 25-pixel drone in a 1440x840 frame is invisible at any playback size that
fits on a screen, which makes a raw recording useless for the one question you
want to answer while looking at it -- "is the target where it should be?".

This adds, per frame:

* a ground-truth box on the observer's pane, from the recorded geometry
  (``frames.json``), not from any detector;
* a magnified inset of the region around the target, so the aircraft is legible
  at normal playback size;
* a readout of range, pixel span and simulation time.

It runs on the **host**, not in the container -- it is ordinary OpenCV over
files, and the host has a real ffmpeg (via ``imageio-ffmpeg``) where the
container has none.

    python simulators/pegasus/scripts/annotate_recording.py <recording-dir> \\
        --out annotated.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

INSET_SCALE = 3
INSET_SRC_PX = 55


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", help="a directory written by run_two_drone.py")
    p.add_argument("--out", default=None, help="output MP4 (default: <recording>/annotated.mp4)")
    p.add_argument("--fps", type=float, default=None, help="override the recorded fps")
    p.add_argument("--scale", type=float, default=0.5,
                   help="output scale relative to the composed frame (default 0.5)")
    p.add_argument("--codec", default="h264", choices=["h264", "mp4v"],
                   help="h264 needs ffmpeg on PATH or imageio-ffmpeg installed")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rec = Path(args.recording)
    frames = json.loads((rec / "frames.json").read_text())
    meta = json.loads((rec / "meta.json").read_text())
    fps = args.fps or meta.get("fps", 20.0)
    out_path = Path(args.out) if args.out else rec / "annotated.mp4"

    left_dir, right_dir = rec / "left", rec / "right"
    n = min(len(frames), len(list(left_dir.glob("*.jpg"))))
    if n == 0:
        raise SystemExit(f"no frames under {rec}")

    first = cv2.imread(str(left_dir / "000000.jpg"))
    h, w = first.shape[:2]
    comp_w, comp_h = w * 2, h
    out_w = _even(int(comp_w * args.scale))
    out_h = _even(int(comp_h * args.scale))

    writer, proc = None, None
    if args.codec == "h264":
        proc = _ffmpeg(out_path, out_w, out_h, fps)
    if proc is None:
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (out_w, out_h))

    for i in range(n):
        L = cv2.imread(str(left_dir / f"{i:06d}.jpg"))
        R = cv2.imread(str(right_dir / f"{i:06d}.jpg"))
        if L is None or R is None:
            break
        row = frames[i]
        L = _annotate_observer(L, row)
        R = _annotate_target(R, row, meta)
        canvas = np.hstack([L, R])
        canvas = _footer(canvas, row, meta)
        canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if proc is not None:
            proc.stdin.write(canvas.tobytes())
        else:
            writer.write(canvas)
        if i % 100 == 0:
            print(f"  {i}/{n}", flush=True)

    if proc is not None:
        proc.stdin.close()
        proc.wait()
    else:
        writer.release()
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({n} frames, {size_mb:.1f} MB)")
    return 0


def _annotate_observer(img, row):
    """Ground-truth box plus a magnified inset of the target."""
    img = _caption(img, "DRONE 1 - observer camera (hovering)")
    uv = row.get("target_uv")
    if uv is None:
        return img

    h, w = img.shape[:2]
    x, y = int(round(uv[0])), int(round(uv[1]))
    span = max(float(row.get("target_px", 20.0)), 8.0)
    half = int(span * 0.8) + 6

    # The inset is taken BEFORE the box is drawn, so the magnified view shows the
    # aircraft rather than the annotation drawn over it.
    x0, y0 = max(0, x - INSET_SRC_PX), max(0, y - INSET_SRC_PX)
    x1, y1 = min(w, x + INSET_SRC_PX), min(h, y + INSET_SRC_PX)
    patch = img[y0:y1, x0:x1].copy()

    cv2.rectangle(img, (x - half, y - half), (x + half, y + half), (0, 255, 255), 2)
    cv2.line(img, (x - half - 14, y), (x - half - 4, y), (0, 255, 255), 2)
    cv2.line(img, (x + half + 4, y), (x + half + 14, y), (0, 255, 255), 2)

    if patch.size:
        ins = cv2.resize(patch, (patch.shape[1] * INSET_SCALE, patch.shape[0] * INSET_SCALE),
                         interpolation=cv2.INTER_NEAREST)
        ih, iw = ins.shape[:2]
        # Top-right: the target rides in the upper-middle of the observer's frame
        # (the calibration's principal point is well above centre), and the
        # bottom-right is where the footer readout goes.
        px, py = w - iw - 16, 62
        img[py:py + ih, px:px + iw] = ins
        cv2.rectangle(img, (px, py), (px + iw, py + ih), (0, 255, 255), 2)
        cv2.putText(img, f"x{INSET_SCALE}", (px + 6, py + ih + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def _annotate_target(img, row, meta):
    """Mark drone 1 in drone 2's pane.

    The observer's position in the target's image is not recorded -- only the
    forward direction is -- but it is recoverable: both world positions are in
    ``frames.json``, and the target is pointed straight at the observer, so its
    heading is the bearing between them. Recomputing here rather than re-flying
    is the whole reason the recording carries full geometry per frame.
    """
    img = _caption(img, "DRONE 2 - target camera (looking back at drone 1)")
    obs = row.get("observer_xyz")
    tgt = row.get("target_xyz")
    if not obs or not tgt or meta.get("target_faces") != "observer":
        return img

    intr = meta.get("intrinsics") or {}
    fx, fy = intr.get("fx"), intr.get("fy")
    cx, cy = intr.get("cx"), intr.get("cy")
    if None in (fx, fy, cx, cy):
        return img

    import math

    yaw = math.atan2(obs[1] - tgt[1], obs[0] - tgt[0])
    dx, dy, dz = (obs[0] - tgt[0], obs[1] - tgt[1], obs[2] - tgt[2])
    c, s = math.cos(-yaw), math.sin(-yaw)
    fwd = dx * c - dy * s
    left = dx * s + dy * c
    if fwd <= 1e-6:
        return img

    u = int(round(cx + fx * (-left) / fwd))
    v = int(round(cy + fy * (-dz) / fwd))
    span = max(float(row.get("target_px", 20.0)), 8.0)
    half = int(span * 0.8) + 6
    h, w = img.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return img

    cv2.rectangle(img, (u - half, v - half), (u + half, v + half), (0, 200, 255), 2)
    cv2.putText(img, "drone 1", (u - half, v - half - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)

    x0, y0 = max(0, u - INSET_SRC_PX), max(0, v - INSET_SRC_PX)
    patch = img[y0:min(h, v + INSET_SRC_PX), x0:min(w, u + INSET_SRC_PX)]
    if patch.size:
        ins = cv2.resize(patch, (patch.shape[1] * INSET_SCALE, patch.shape[0] * INSET_SCALE),
                         interpolation=cv2.INTER_NEAREST)
        ih, iw = ins.shape[:2]
        px, py = w - iw - 16, 62
        img[py:py + ih, px:px + iw] = ins
        cv2.rectangle(img, (px, py), (px + iw, py + ih), (0, 200, 255), 2)
    return img


def _caption(img, text):
    cv2.rectangle(img, (10, 10), (18 + 15 * len(text), 48), (0, 0, 0), -1)
    cv2.putText(img, text, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _footer(canvas, row, meta):
    h, w = canvas.shape[:2]
    cv2.line(canvas, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
    bits = [f"t={row['t']:6.2f}s", f"range={row.get('range_m', 0):5.1f}m",
            f"target={row.get('target_px', 0):4.1f}px"]
    if row.get("target_in_frame") is not None:
        bits.append("IN FRAME" if row["target_in_frame"] else "out of frame")
    text = "   ".join(bits)
    cv2.rectangle(canvas, (10, h - 46), (26 + 17 * len(text), h - 8), (0, 0, 0), -1)
    cv2.putText(canvas, text, (18, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _even(n):
    return n if n % 2 == 0 else n + 1


def _ffmpeg(out_path, w, h, fps):
    """An ffmpeg pipe, or None if no ffmpeg is available."""
    exe = "ffmpeg"
    try:
        subprocess.run([exe, "-version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            print("no ffmpeg available, falling back to mp4v")
            return None
    cmd = [exe, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", str(out_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


if __name__ == "__main__":
    raise SystemExit(main())
