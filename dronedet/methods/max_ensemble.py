"""MAX v2 detector: multi-expert ensemble = RGB appearance (SAHI) + a TEMPORAL
expert (SAHI over ego-aligned 3-frame motion-trail stacks), fused by NMS.

The temporal expert is motion-driven and colour-blind: a moving drone leaves a
bright trail across the stacked channels while registered background is grey, so
it supplies confident 'drone' detections exactly where the RGB model fails (a
tiny black drone). Both experts detect at native scale via tiling; the union is
deduplicated. On an aggressively moving camera the temporal expert is weighted
down (parallax over 3-D terrain leaks into the stack), mirroring the pipeline's
appearance-first stance there.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..detections import Detection, nms
from .base import BaseMethod
from .yolo import tile_origins


class MaxEnsemble(BaseMethod):
    def __init__(self, name, weights, temporal_weights=None, tile=640, overlap=0.25,
                 near_static=True, dt=3, conf=0.02, device=0):
        from ultralytics import YOLO
        self.name = name
        self.app = YOLO(weights)
        self.temporal = YOLO(temporal_weights) if temporal_weights else None
        self.tile, self.overlap, self.conf, self.device = tile, overlap, conf, device
        self.dt = dt
        self.near_static = near_static
        self.tw = 1.0 if near_static else 0.6      # temporal-expert score weight by regime
        self.buf: dict[int, np.ndarray] = {}
        # classical ego-motion frame-differencing: most sensitive to a slow near-static
        # mover (the black drone), so it joins the ensemble only on a near-static camera
        # (on a moving camera it is parallax clutter).
        self.mc = None
        if near_static:
            from .mc_hybrid import _MCMotion
            self.mc = _MCMotion()

    def _sahi(self, model, img, label):
        h, w = img.shape[:2]
        xs, ys = tile_origins(self.tile, w, self.overlap), tile_origins(self.tile, h, self.overlap)
        tiles, offs = [], []
        for y0 in ys:
            for x0 in xs:
                tiles.append(np.ascontiguousarray(img[y0:y0 + self.tile, x0:x0 + self.tile]))
                offs.append((x0, y0))
        out = []
        for r, (dx, dy) in zip(model(tiles, imgsz=self.tile, conf=self.conf, device=self.device,
                                     verbose=False), offs):
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                out.append(Detection(x1 + dx, y1 + dy, x2 + dx, y2 + dy, float(b.conf[0]), label))
        return out

    def _register(self, src, dst):
        h, w = dst.shape
        xs = np.linspace(w * 0.05, w * 0.95, 30)
        ys = np.linspace(h * 0.05, h * 0.95, 17)
        pts = np.array([[x, y] for y in ys for x in xs], np.float32).reshape(-1, 1, 2)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(src, dst, pts, None, winSize=(21, 21), maxLevel=3,
                                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        good = st.ravel() == 1
        if good.sum() < 12:
            return None
        H, _ = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 3.0)
        return H

    def _stack(self, idx, g):
        h, w = g.shape
        chans = []
        for k in (2 * self.dt, self.dt, 0):
            gp = self.buf.get(idx - k)
            if gp is None or k == 0:
                chans.append(g)
                continue
            H = self._register(gp, g)
            chans.append(cv2.warpPerspective(gp, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
                         if H is not None else gp)
        return cv2.merge(chans)

    def process(self, idx, frame_bgr, m_stab):
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.buf[idx] = g
        self.buf.pop(idx - 2 * self.dt - 2, None)
        dets = self._sahi(self.app, frame_bgr, "drone")               # RGB appearance expert
        if self.temporal is not None and idx >= 2 * self.dt:
            stack = self._stack(idx, g)                               # motion-trail input
            for d in self._sahi(self.temporal, stack, "drone-temporal"):
                dets.append(Detection(d.x1, d.y1, d.x2, d.y2, d.score * self.tw, d.label))
        if self.mc is not None:                                        # classical ego-motion
            for c in self.mc.process(frame_bgr):
                s = min(1.0, c.score / 25.0)
                dets.append(Detection(c.x1, c.y1, c.x2, c.y2, 0.30 + 0.40 * s, "motion"))
        return nms(dets, dist_thresh=10.0)
