"""Global camera-motion estimation.

The camera sits on a hovering drone: motion is a small global drift.
Two estimators are provided:

* ``translation`` -- phase correlation of the full gray frame against the
  first frame (windowed). Sub-pixel accurate, immune to small moving objects,
  and drift-free because every frame is registered to the same reference.
* ``affine`` -- sparse LK optical flow on a fixed grid + RANSAC partial
  affine, composed frame-to-frame. Use when rotation/zoom is expected.

Both return a 2x3 affine matrix mapping the *current* frame into the
*reference* (frame 0) coordinate system.
"""

from __future__ import annotations

import cv2
import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


class Stabilizer:
    def __init__(self, mode: str = "translation", min_response: float = 0.35,
                 scale: float = 1.0):
        if mode not in ("translation", "affine", "off"):
            raise ValueError(f"unknown stabilizer mode: {mode}")
        self.mode = mode
        self.min_response = min_response
        self.scale = scale          # estimate on a downscaled gray (affine is
        #                             scale-invariant); only translation rescales
        self._ref_gray: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self._window: np.ndarray | None = None
        self._accum = IDENTITY.copy()  # affine mode: prev-frame -> reference

    def update(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return 2x3 matrix M such that warpAffine(frame, M) aligns it to frame 0."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.mode == "off":
            return IDENTITY.copy()
        if self.scale != 1.0:
            gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)
        if self._ref_gray is None:
            self._ref_gray = gray.astype(np.float32)
            self._prev_gray = gray
            self._window = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)
            return IDENTITY.copy()
        m = (self._update_translation(gray) if self.mode == "translation"
             else self._update_affine(gray))
        if self.scale != 1.0:            # rescale translation back to full-res pixels
            m = m.copy()
            m[0, 2] /= self.scale
            m[1, 2] /= self.scale
        return m

    def _update_translation(self, gray: np.ndarray) -> np.ndarray:
        (dx, dy), response = cv2.phaseCorrelate(
            gray.astype(np.float32), self._ref_gray, self._window
        )
        if response < self.min_response:
            # scene changed too much for direct registration; fall back to
            # frame-to-frame accumulation for this step
            (sdx, sdy), _ = cv2.phaseCorrelate(
                gray.astype(np.float32), self._prev_gray.astype(np.float32), self._window
            )
            dx = self._accum[0, 2] + sdx
            dy = self._accum[1, 2] + sdy
        m = IDENTITY.copy()
        m[0, 2] = dx
        m[1, 2] = dy
        self._accum = m
        self._prev_gray = gray
        return m

    def _update_affine(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        xs = np.linspace(w * 0.05, w * 0.95, 24)
        ys = np.linspace(h * 0.05, h * 0.95, 14)
        pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        good = status.ravel() == 1
        step = IDENTITY.copy()
        if good.sum() >= 12:
            m, _ = cv2.estimateAffinePartial2D(
                nxt[good], pts[good], method=cv2.RANSAC, ransacReprojThreshold=1.5
            )
            if m is not None:
                step = m
        # compose: current -> prev -> reference
        a = np.vstack([self._accum, [0, 0, 1]])
        s = np.vstack([step, [0, 0, 1]])
        self._accum = (a @ s)[:2]
        self._prev_gray = gray
        return self._accum.copy()


def warp_to_reference(frame: np.ndarray, m: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def shift_of(m: np.ndarray) -> tuple[float, float]:
    return float(m[0, 2]), float(m[1, 2])


def apply_to_points(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine to an (N,2) array of points."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ m[:, :2].T + m[:, 2]


def invert(m: np.ndarray) -> np.ndarray:
    full = np.vstack([m, [0, 0, 1]])
    return np.linalg.inv(full)[:2]
