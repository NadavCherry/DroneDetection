"""Write both drones' onboard views into one side-by-side MP4.

Left half is the observer's camera, right half is the target's -- both are the
aircraft's **own** forward-facing camera, not an external view of the aircraft.
That distinction is the entire point of this module: an external "chase camera"
recording shows you a drone flying around and tells you nothing about what the
drone could see, and what the drone could see is the only thing a detector ever
gets.

Frames are also written out individually. The MP4 is for looking at; the JPEGs
are what the detector reads, because every strong detector in this project is
temporal and needs to be fed consecutive frames with monotonically increasing
indices -- decoding those back out of a compressed video adds an artefact to
the exact signal (small inter-frame differences) those models key on.

**There is no ffmpeg in the isaac-sim container**, so encoding is OpenCV's
``mp4v`` writer. Re-encode on the host if you need H.264.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class SplitScreenRecorder:
    """Records two onboard camera streams as one side-by-side video.

    Args:
        out_dir: Directory to write into. Created if absent.
        size: ``(width, height)`` of ONE pane. The video is twice as wide.
        fps: Frame rate to stamp the video with.
        labels: ``(left, right)`` captions burned into each pane.
        save_frames: Also write per-pane JPEGs under ``left/`` and ``right/``.
    """

    def __init__(self, out_dir, size, fps: float = 20.0,
                 labels=("DRONE 1 - observer", "DRONE 2 - target"),
                 save_frames: bool = True):
        import cv2

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.width, self.height = int(size[0]), int(size[1])
        self.fps = float(fps)
        self.labels = labels
        self.save_frames = save_frames
        self.frames = 0
        self._meta_rows = []

        if save_frames:
            (self.out_dir / "left").mkdir(exist_ok=True)
            (self.out_dir / "right").mkdir(exist_ok=True)

        # Both dimensions even: an odd width is legal for mp4v but breaks any
        # later yuv420p re-encode on the host, and it surfaces as a broken pipe
        # from ffmpeg rather than as anything that names the width.
        vid_w = _even(self.width * 2)
        vid_h = _even(self.height)
        self.video_path = self.out_dir / "split_view.mp4"
        self._writer = cv2.VideoWriter(
            str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps, (vid_w, vid_h))
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open a video writer at {self.video_path}")
        self._vid_size = (vid_w, vid_h)

    def capture(self, left_rgb, right_rgb, stamp_s: float, extra: dict = None) -> bool:
        """Append one frame from each camera.

        Args:
            left_rgb: Observer's ``(H, W, 3)`` uint8 RGB, or None.
            right_rgb: Target's frame, same shape, or None.
            stamp_s: Simulation time of this frame.
            extra: Per-frame facts to record in ``frames.json`` (positions,
                range, whether the target was inside the observer's frustum).

        Returns:
            True if a frame was written. A camera that has not warmed up yet
            returns None rather than raising, and a dropped frame here is
            normal during warm-up and a bug afterwards -- hence the return
            value rather than a silent skip.
        """
        import cv2

        if left_rgb is None or right_rgb is None:
            return False

        left = _fit(left_rgb, self.width, self.height)
        right = _fit(right_rgb, self.width, self.height)

        if self.save_frames:
            cv2.imwrite(str(self.out_dir / "left" / f"{self.frames:06d}.jpg"),
                        left[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(self.out_dir / "right" / f"{self.frames:06d}.jpg"),
                        right[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])

        canvas = np.hstack([left, right])
        canvas = _label(canvas, self.labels[0], self.labels[1], stamp_s)
        canvas = _fit(canvas, self._vid_size[0], self._vid_size[1])
        self._writer.write(canvas[:, :, ::-1])  # RGB -> BGR for cv2

        row = {"frame": self.frames, "t": round(float(stamp_s), 4)}
        if extra:
            row.update(extra)
        self._meta_rows.append(row)
        self.frames += 1
        return True

    def finish(self, meta: dict = None) -> dict:
        """Close the video and write the sidecar metadata.

        Returns:
            A stats dict, also written to ``meta.json``.
        """
        self._writer.release()

        (self.out_dir / "frames.json").write_text(json.dumps(self._meta_rows, indent=1), encoding="utf-8")

        stats = {
            "frames": self.frames,
            "fps": self.fps,
            "pane_size": [self.width, self.height],
            "video_size": list(self._vid_size),
            "video": str(self.video_path),
            "duration_s": round(self.frames / self.fps, 3) if self.fps else None,
        }
        if meta:
            stats.update(meta)
        (self.out_dir / "meta.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        return stats


def _even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def _fit(img, width: int, height: int):
    """Resize/pad ``img`` to exactly ``width`` x ``height``."""
    import cv2

    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return img
    if (w, h) != (width, height) and abs(w - width) <= 2 and abs(h - height) <= 2:
        # An off-by-one from the even-width rounding: pad rather than resample,
        # so the imagery a detector sees is pixel-identical to what was rendered.
        out = np.zeros((height, width, 3), dtype=img.dtype)
        out[:min(h, height), :min(w, width)] = img[:min(h, height), :min(w, width)]
        return out
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _label(canvas, left_text: str, right_text: str, stamp_s: float):
    """Burn pane captions and a clock into the composed frame."""
    import cv2

    canvas = np.ascontiguousarray(canvas)
    h, w = canvas.shape[:2]
    half = w // 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.line(canvas, (half, 0), (half, h), (255, 255, 255), 2)
    for text, x0 in ((left_text, 0), (right_text, half)):
        cv2.rectangle(canvas, (x0 + 6, 6), (x0 + 10 + 11 * len(text), 34), (0, 0, 0), -1)
        cv2.putText(canvas, text, (x0 + 10, 27), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    clock = f"t = {stamp_s:6.2f}s"
    cv2.rectangle(canvas, (w - 150, h - 34), (w - 6, h - 6), (0, 0, 0), -1)
    cv2.putText(canvas, clock, (w - 144, h - 13), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
