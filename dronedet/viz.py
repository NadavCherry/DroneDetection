"""Overlay drawing and annotated-video writing."""

from __future__ import annotations

import cv2
import numpy as np

from .detections import Detection

COLORS = {
    "det": (0, 60, 255),      # orange-red
    "gt": (60, 220, 60),      # green
    "track": (255, 160, 0),   # blue-ish (BGR)
    "coast": (0, 200, 255),   # yellow
}


def draw_box(img: np.ndarray, d: Detection, color: tuple, label: str | None = None,
             pad: float = 6.0, thickness: int = 1) -> None:
    """Tiny targets get a padded box so they stay visible at video scale."""
    x1, y1, x2, y2 = d.x1 - pad, d.y1 - pad, d.x2 + pad, d.y2 + pad
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    if label:
        cv2.putText(img, label, (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def zoom_inset(img: np.ndarray, cx: float, cy: float, half: int = 28,
               scale: int = 5, corner: str = "tr") -> None:
    """Paste a magnified crop around (cx, cy) into a corner of the frame."""
    h, w = img.shape[:2]
    x0 = int(np.clip(cx - half, 0, w - 2 * half))
    y0 = int(np.clip(cy - half, 0, h - 2 * half))
    crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
    z = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    zh, zw = z.shape[:2]
    cv2.rectangle(z, (0, 0), (zw - 1, zh - 1), (255, 255, 255), 2)
    if corner == "tr":
        img[8:8 + zh, w - 8 - zw:w - 8] = z
    else:
        img[8:8 + zh, 8:8 + zw] = z


class VideoSink:
    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        if not self.writer.isOpened():
            raise IOError(f"cannot open video writer: {path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()
