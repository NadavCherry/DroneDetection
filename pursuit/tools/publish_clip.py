#!/usr/bin/env python3
"""Turn a recorded engagement into something a README can show.

The in-loop writer is OpenCV's ``mp4v``, which is large and which GitHub will
not play inline. This re-encodes to H.264 and, optionally, renders a GIF -- the
only format that animates in a Markdown ``<img>`` without a click.

    .venv/bin/python -m pursuit.tools.publish_clip work/pursuit/city/headline.mp4 \\
        --out docs/media/pursuit/city_intercept --gif --speed 2 --width 900

Two things worth keeping right, because both were got wrong once:

* **A GIF needs a generated palette.** The default 256-colour quantiser picks
  its palette from the first frame, and the first frame of one of these clips is
  mostly sky -- so the city arrives banded and the 3-pixel target, whose whole
  signature is being slightly darker than its background, disappears entirely.
  ``palettegen``/``paletteuse`` over the *whole* clip is the fix.
* **Speeding a clip up must drop frames, not just relabel them.** ``setpts``
  alone leaves a frame rate no player honours in a GIF.

A third, learnt from the files this had already shipped: ffmpeg writes the moov
atom *after* the payload unless told otherwise, so a browser has to fetch the
whole file before it can show the first frame. ``-movflags +faststart`` moves
it to the front, and ``-g`` keeps the keyframes close enough that scrubbing
lands where it was asked to.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run(args: list) -> None:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        raise SystemExit(f"ffmpeg failed: {' '.join(args[:6])} ...")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=12, help="GIF frame rate")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--crf", type=int, default=23)
    a = ap.parse_args(argv)

    src = Path(a.clip)
    if not src.is_file():
        raise SystemExit(f"no clip at {src}")
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fx = ffmpeg()

    vf = f"setpts={1.0 / a.speed:.4f}*PTS,scale={a.width}:-2:flags=lanczos"
    mp4 = out.with_suffix(".mp4")
    run([fx, "-y", "-i", str(src), "-vf", vf, "-an", "-c:v", "libx264",
         "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p",
         "-g", "40", "-movflags", "+faststart", str(mp4)])
    print(f"wrote {mp4}  ({mp4.stat().st_size / 1e6:.1f} MB)")

    if a.gif:
        gif = out.with_suffix(".gif")
        pal = out.parent / f".{out.name}_palette.png"
        gvf = (f"setpts={1.0 / a.speed:.4f}*PTS,fps={a.fps},"
               f"scale={a.width}:-2:flags=lanczos")
        run([fx, "-y", "-i", str(src), "-vf", f"{gvf},palettegen=stats_mode=diff",
             str(pal)])
        run([fx, "-y", "-i", str(src), "-i", str(pal), "-lavfi",
             f"{gvf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
             "-loop", "0", str(gif)])
        pal.unlink(missing_ok=True)
        print(f"wrote {gif}  ({gif.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
