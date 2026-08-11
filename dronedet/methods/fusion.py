"""RGB+motion fusion detector (inference side of the YOLOMG-style model).

The model is a ch=4 P2-YOLO trained on [B, G, R, ego-motion-difference] tiles
(see tools/make_fusion_combined.py). At inference we reconstruct the same 4-channel
input per frame -- RGB from the current frame, channel 4 = min over {t-dt, t-2dt}
of |gray_t - warp(gray_past)| after grid-LK+RANSAC ego-registration -- tile it at
native resolution (SAHI), and run one batched forward.

ultralytics' high-level predictor assumes 3-channel BGR (it does BGR->RGB and
letterbox on 3 ch), so it cannot drive a 4-channel model; we call the raw nn.Module
and decode with ultralytics' own NMS. Tiles are already `tile`x`tile`, so boxes come
back in tile pixels directly -- no letterbox rescale.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from ..detections import Detection, nms
from .base import BaseMethod
from .yolo import tile_origins


def _register(src_gray, dst_gray, gx=30, gy=17):
    """Homography mapping src onto dst (grid LK + RANSAC); None if too weak."""
    h, w = dst_gray.shape
    xs = np.linspace(w * 0.05, w * 0.95, gx)
    ys = np.linspace(h * 0.05, h * 0.95, gy)
    pts = np.array([[x, y] for y in ys for x in xs], np.float32).reshape(-1, 1, 2)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
        src_gray, dst_gray, pts, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    good = st.ravel() == 1
    if good.sum() < 12:
        return None
    H, _ = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 3.0)
    return H


class FusionDetector(BaseMethod):
    def __init__(self, name: str, weights: str, tile: int = 640, overlap: float = 0.25,
                 conf: float = 0.02, iou: float = 0.7, dt: int = 3, device: int | str = 0,
                 max_buffer: int = 8):
        from ultralytics import YOLO
        self.name = name
        self.tile, self.overlap = tile, overlap
        self.conf, self.iou, self.dt = conf, iou, dt
        self.device = f"cuda:{device}" if isinstance(device, int) else device
        # trained ch=4 module (checkpoint saved the 4-channel first conv)
        self.net = YOLO(weights).model.to(self.device).eval()
        self._buf: dict[int, np.ndarray] = {}    # idx -> gray(t)   (for motion)
        self._max = max_buffer

    def _motion_map(self, idx: int) -> np.ndarray:
        g = self._buf.get(idx)
        h, w = g.shape
        diffs = []
        for k in (self.dt, 2 * self.dt):
            gp = self._buf.get(idx - k)
            if gp is None:
                continue
            H = _register(gp, g)
            warp = (cv2.warpPerspective(gp, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
                    if H is not None else gp)
            diffs.append(cv2.absdiff(g, warp))
        if not diffs:
            return np.zeros((h, w), np.uint8)
        return diffs[0] if len(diffs) == 1 else np.minimum(diffs[0], diffs[1])

    @torch.no_grad()
    def process(self, idx, frame_bgr, m_stab):
        self._buf[idx] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._buf.pop(idx - self._max, None)
        h, w = frame_bgr.shape[:2]
        motion = self._motion_map(idx)                       # H,W uint8
        frame4 = np.dstack([frame_bgr, motion])              # H,W,4 (B,G,R,motion)

        xs = tile_origins(self.tile, w, self.overlap)
        ys = tile_origins(self.tile, h, self.overlap)
        tiles, offs = [], []
        for y0 in ys:
            for x0 in xs:
                t = frame4[y0:y0 + self.tile, x0:x0 + self.tile]
                if t.shape[:2] != (self.tile, self.tile):    # pad boundary tiles
                    pad = np.zeros((self.tile, self.tile, 4), t.dtype)
                    pad[:t.shape[0], :t.shape[1]] = t
                    t = pad
                tiles.append(t)
                offs.append((x0, y0))

        batch = np.stack(tiles).transpose(0, 3, 1, 2)        # N,4,tile,tile
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device).float() / 255.0
        out = self.net(x)
        preds = out[0] if isinstance(out, (list, tuple)) else out
        from ultralytics.utils.nms import non_max_suppression
        nmsed = non_max_suppression(preds, self.conf, self.iou, classes=None, max_det=300)

        dets: list[Detection] = []
        for det, (dx, dy) in zip(nmsed, offs):
            if det is None or not len(det):
                continue
            for x1, y1, x2, y2, sc, _cls in det.cpu().numpy():
                dets.append(Detection(float(x1) + dx, float(y1) + dy,
                                      float(x2) + dx, float(y2) + dy, float(sc), "drone"))
        return nms(dets, dist_thresh=10.0)
