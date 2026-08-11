"""Cheap-but-correct stabilizer for edge hardware.

v1 phase-correlated a downscaled frame; its estimates drifted by several
pixels (low correlation response at reduced scale -> the accumulate
fallback compounds errors) which silently destroyed the background model
downstream (val proposal recall 0.64 -> 0.16). v2 correlates a fixed
FULL-RESOLUTION central crop against frame 0: translation of the crop ==
global translation, precision matches the full-frame stabilizer, cost is
proportional to crop area (~1/3 of the frame).
"""

from __future__ import annotations

import cv2
import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


class LiteStabilizer:
    def __init__(self, crop_w: int = 768, crop_h: int = 448,
                 min_response: float = 0.3):
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.min_response = min_response
        self.scale = 1.0  # shifts are full-scale (runner compatibility)
        self._acc = (0.0, 0.0)
        self._ref: np.ndarray | None = None
        self._prev: np.ndarray | None = None
        self._win: np.ndarray | None = None
        self._roi: tuple[int, int] | None = None

    def _crop(self, gray: np.ndarray) -> np.ndarray:
        if self._roi is None:
            h, w = gray.shape
            self._roi = ((w - self.crop_w) // 2, (h - self.crop_h) // 2)
        x0, y0 = self._roi
        return gray[y0:y0 + self.crop_h, x0:x0 + self.crop_w].astype(np.float32)

    def update(self, gray_full: np.ndarray) -> np.ndarray:
        c = self._crop(gray_full)
        if self._ref is None:
            self._ref = c
            self._prev = c
            self._win = cv2.createHanningWindow((self.crop_w, self.crop_h),
                                                cv2.CV_32F)
            return IDENTITY.copy()
        (dx, dy), resp = cv2.phaseCorrelate(c, self._ref, self._win)
        if resp < self.min_response:
            (sdx, sdy), _ = cv2.phaseCorrelate(c, self._prev, self._win)
            dx = self._acc[0] + sdx
            dy = self._acc[1] + sdy
        self._acc = (dx, dy)
        self._prev = c
        m = IDENTITY.copy()
        m[0, 2] = dx
        m[1, 2] = dy
        return m
