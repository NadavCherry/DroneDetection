"""O(1)-per-frame motion detectors for edge hardware.

``LaggedEMA`` replaces the PC pipeline's periodic-median background: the
background/variance are exponential moving averages fed with a DELAYED
frame (a small ring buffer provides the frame from ~``lag`` frames ago),
so a slow drifter never enters its own background model -- the property
that made the PC slow-mover channel work -- at a constant ~5 ms/frame
with no recompute spikes.

``FrameDiff3`` is the classical aligned three-frame differencing detector
(min of |t - (t-k)| and |t - (t-2k)|), the cheapest possible mover finder.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from dronedet.detections import Detection


def _extract(mask: np.ndarray, score_img: np.ndarray, min_area: int,
             max_side: int, max_dets: int, border: int = 4) -> list[Detection]:
    mask[:border], mask[-border:], mask[:, :border], mask[:, -border:] = 0, 0, 0, 0
    merged = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    dets = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or w > max_side or h > max_side:
            continue
        peak = float(score_img[y:y + h, x:x + w].max())
        dets.append(Detection(float(x), float(y), float(x + w), float(y + h),
                              score=peak, label="motion"))
    dets.sort(key=lambda d: -d.score)
    return dets[:max_dets]


class LaggedEMA:
    def __init__(self, lag: int = 96, stride: int = 6, alpha: float = 0.06,
                 k_sigma: float = 5.0, min_diff: float = 10.0,
                 sigma_floor: float = 1.6, min_area: int = 2,
                 max_side: int = 120, max_dets: int = 40,
                 flicker_alpha: float = 0.005, flicker_rate0: float = 0.30,
                 flicker_boost: float = 3.0, warmup: int = 40):
        self.ring: deque[np.ndarray] = deque(maxlen=max(2, lag // stride))
        self.stride = stride
        self.alpha = alpha
        self.k_sigma = k_sigma
        self.min_diff = min_diff
        self.sigma_floor = sigma_floor
        self.min_area = min_area
        self.max_side = max_side
        self.max_dets = max_dets
        self.fa, self.fr0, self.fb = flicker_alpha, flicker_rate0, flicker_boost
        self.warmup = warmup
        self.bg: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.flicker: np.ndarray | None = None
        self.n = 0

    def process(self, gray_stab: np.ndarray) -> list[Detection]:
        g = gray_stab.astype(np.float32)
        self.n += 1
        if self.n % self.stride == 1:
            self.ring.append(g)
        if self.bg is None:
            self.bg = g.copy()
            self.var = np.full_like(g, 25.0)
            self.flicker = np.zeros_like(g)
            return []
        # feed the EMA with the OLDEST ringed frame (≈ lag frames delayed)
        if len(self.ring) == self.ring.maxlen and self.n % self.stride == 0:
            old = self.ring[0]
            d = old - self.bg
            self.bg += self.alpha * d
            self.var += self.alpha * (d * d - self.var)
        if self.n < self.warmup:
            return []
        sigma = np.sqrt(self.var)
        np.maximum(sigma, self.sigma_floor, out=sigma)
        diff = np.abs(g - self.bg)
        snr = diff / sigma
        boost = 1.0 + self.fb * np.minimum(self.flicker / self.fr0, 3.0)
        base = (snr > self.k_sigma) & (diff > self.min_diff)
        mask = (base & (snr > self.k_sigma * boost)).astype(np.uint8)
        cur = cv2.dilate(base.astype(np.float32), np.ones((5, 5), np.float32))
        self.flicker += self.fa * (cur - self.flicker)
        return _extract(mask, snr, self.min_area, self.max_side, self.max_dets)


class FrameDiff3:
    """Aligned 3-frame differencing: response = min(|t-(t-k)|, |t-(t-2k)|)
    (suppresses 'revealed background' ghosts). Cheapest mover finder."""

    def __init__(self, k: int = 6, thresh: float = 14.0, min_area: int = 2,
                 max_side: int = 120, max_dets: int = 40):
        self.k = k
        self.thresh = thresh
        self.min_area = min_area
        self.max_side = max_side
        self.max_dets = max_dets
        self.buf: deque[np.ndarray] = deque(maxlen=2 * k + 1)

    def process(self, gray_stab: np.ndarray) -> list[Detection]:
        g = gray_stab.astype(np.float32)
        self.buf.append(g)
        if len(self.buf) < self.buf.maxlen:
            return []
        d1 = np.abs(g - self.buf[self.k])
        d2 = np.abs(g - self.buf[0])
        resp = np.minimum(d1, d2)
        mask = (resp > self.thresh).astype(np.uint8)
        return _extract(mask, resp, self.min_area, self.max_side, self.max_dets)
