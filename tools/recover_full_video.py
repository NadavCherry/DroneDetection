"""Losslessly recover the frames an MP4 edit list hides.

Both source videos were exported with an edit list that trims their first
seconds: every decoder (players, OpenCV, PyAV, ffmpeg) honours it and
silently drops the pre-roll, which is why "readable frames" < the
container's sample count. data/videos/07_05.mp4 carries 741 frames but delivers the
last 571 (pre-roll = 170 frames / 5.7 s); data/videos/10_06.mp4 carries 591 and
delivers 361 (pre-roll = 230 frames / 7.7 s).

This remuxes with the edit list ignored -- pure stream copy, so the
recovered frames are bit-identical to the originals and old frame index
``i`` maps to ``i + <pre-roll>`` in the recovered file.

    python tools/recover_full_video.py --video data/videos/10_06.mp4
    # -> work/video/10_06_full.mp4 (591 readable frames instead of 361)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.video import probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None,
                    help="default: work/video/<stem>_full.mp4")
    a = ap.parse_args()

    import imageio_ffmpeg

    out = Path(a.out or f"work/video/{Path(a.video).stem}_full.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner",
                    "-loglevel", "warning", "-y", "-ignore_editlist", "1",
                    "-i", a.video, "-map", "0:v:0", "-c:v", "copy", "-an",
                    "-avoid_negative_ts", "make_zero", str(out)], check=True)

    n_in = probe(a.video, count_frames=True).frame_count
    n_out = probe(str(out), count_frames=True).frame_count
    print(f"{a.video}: {n_in} readable frames -> {out}: {n_out} "
          f"(recovered pre-roll: {n_out - n_in} frames; old frame i = "
          f"new frame i + {n_out - n_in})")


if __name__ == "__main__":
    main()
