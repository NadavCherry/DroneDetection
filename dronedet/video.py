"""Sequential video reading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int  # actual readable frames (container metadata can lie)


def probe(path: str, count_frames: bool = False) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    info = VideoInfo(
        path=path,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if count_frames:
        n = 0
        while cap.grab():
            n += 1
        info.frame_count = n
    cap.release()
    return info


def frames(path: str, start: int = 0, stop: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, BGR frame) sequentially. Seeking is avoided on purpose:
    inter-frame codecs make positioned reads unreliable near stream ends."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (stop is not None and idx >= stop):
                return
            if idx >= start:
                yield idx, frame
            idx += 1
    finally:
        cap.release()
