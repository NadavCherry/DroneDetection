"""Stabilized background-model motion detector for tiny targets.

Backend ``median``: temporal median background + per-pixel robust noise
(MAD -> sigma). A pixel is foreground when its deviation exceeds
``k_sigma`` times its *own* historical noise level, so wind-blown foliage
and flickering high-contrast edges (which have high historical variance)
are automatically suppressed, while a drone against calm sky (tiny sigma)
is detectable at very low contrast.

Backend ``mog2``: OpenCV MOG2 on the stabilized frame, same post-processing.

Detections are returned in *stabilized* (reference-frame) coordinates;
callers map them back with the inverse stabilization transform.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .detections import Detection


class MotionDetector:
    def __init__(
        self,
        backend: str = "median",
        window: int = 150,
        sample_stride: int = 5,
        lag: int = 0,
        k_sigma: float = 5.0,
        min_diff: float = 10.0,
        sigma_floor: float = 1.5,
        min_area: int = 2,
        max_area: int = 4000,
        warmup: int = 12,
        bg_update_every: int = 10,
        border: int = 4,
        flicker_alpha: float = 0.02,
        flicker_rate0: float = 0.10,
        flicker_boost: float = 6.0,
        max_dets: int = 60,
    ):
        # ``lag``: the background is built only from frames at least ``lag``
        # old. A slowly drifting target (<1 px/frame) stays out of its own
        # background model instead of being absorbed by it; static scenery
        # is unaffected. lag=0 reproduces the original behaviour.
        if backend not in ("median", "mog2"):
            raise ValueError(f"unknown motion backend: {backend}")
        self.backend = backend
        self.k_sigma = k_sigma
        self.min_diff = min_diff
        self.sigma_floor = sigma_floor
        self.min_area = min_area
        self.max_area = max_area
        self.warmup = warmup
        self.sample_stride = sample_stride
        self.bg_update_every = bg_update_every
        self.border = border
        self.flicker_alpha = flicker_alpha
        self.flicker_rate0 = flicker_rate0
        self.flicker_boost = flicker_boost
        self.max_dets = max_dets
        self.lag = lag
        self._pending: deque[tuple[int, np.ndarray]] = deque()
        self._samples: deque[np.ndarray] = deque(maxlen=max(2, window // sample_stride))
        self._bg: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._flicker: np.ndarray | None = None
        self._n_seen = 0
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=14, detectShadows=False
        )

    def process(self, gray_stab: np.ndarray) -> list[Detection]:
        g = gray_stab.astype(np.float32)
        self._n_seen += 1
        if self.backend == "mog2":
            fg = self._mog2.apply(gray_stab)
            if self._n_seen <= self.warmup:
                return []
            mask = (fg > 0).astype(np.uint8)
            return self._extract(mask, np.full_like(g, 20.0), np.full_like(g, 4.0))

        if self._n_seen % self.sample_stride == 1 or not (self._samples or self._pending):
            if self.lag > 0:
                self._pending.append((self._n_seen, g))
            else:
                self._samples.append(g)
        while self._pending and self._pending[0][0] <= self._n_seen - self.lag:
            self._samples.append(self._pending.popleft()[1])
        if self._n_seen < self.warmup or len(self._samples) < 3:
            return []
        if self._bg is None or self._n_seen % self.bg_update_every == 0:
            stack = np.stack(self._samples)
            self._bg = np.median(stack, axis=0)
            mad = np.median(np.abs(stack - self._bg), axis=0)
            self._sigma = np.maximum(1.4826 * mad, self.sigma_floor)

        diff = np.abs(g - self._bg)
        snr = diff / self._sigma
        base = (snr > self.k_sigma) & (diff > self.min_diff)

        # Flicker suppression: vegetation and flickering edges keep firing at
        # the same pixels, a transiting drone does not. Pixels with a high
        # historical foreground rate get a raised local threshold.
        if self._flicker is None:
            self._flicker = np.zeros_like(g)
        boost = 1.0 + self.flicker_boost * np.minimum(self._flicker / self.flicker_rate0, 3.0)
        mask = (base & (snr > self.k_sigma * boost)).astype(np.uint8)
        # slight spatial spread so the *neighbourhood* of chronic movers is
        # suppressed too (leaves sway a few pixels)
        cur = cv2.dilate(base.astype(np.float32), np.ones((5, 5), np.float32))
        self._flicker += self.flicker_alpha * (cur - self._flicker)
        return self._extract(mask, diff, self._sigma)

    def _extract(self, mask: np.ndarray, diff: np.ndarray, sigma: np.ndarray) -> list[Detection]:
        b = self.border
        mask[:b, :] = 0
        mask[-b:, :] = 0
        mask[:, :b] = 0
        mask[:, -b:] = 0
        # merge fragmented parts (rotor arms) without erasing 2-px targets:
        # dilate for connectivity only, then measure on the raw mask
        merged = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
        dets: list[Detection] = []
        for i in range(1, n):
            x, y, w, h, _area = stats[i]
            raw_area = int((mask[y:y + h, x:x + w] > 0).sum())
            if raw_area < self.min_area or raw_area > self.max_area:
                continue
            if w > 120 or h > 120:  # tiny-target detector: reject huge regions
                continue
            region_snr = (diff[y:y + h, x:x + w] / sigma[y:y + h, x:x + w])
            peak = float(region_snr.max())
            dets.append(Detection(float(x), float(y), float(x + w), float(y + h),
                                  score=peak, label="motion"))
        dets.sort(key=lambda d: -d.score)
        return dets[: self.max_dets]
