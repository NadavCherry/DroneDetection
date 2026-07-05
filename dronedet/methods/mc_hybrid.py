"""Moving-camera motion-guided detection.

The static-camera pipeline (temporal-median background on frames stabilized to
frame 0) collapses when the camera itself flies: registration to a distant
reference leaves huge invalid borders and the "background" is never still. The
top methods on ARD-MAV / ARD100 (GLAD, YOLOMG) instead do **ego-motion-compensated
frame differencing**: register *consecutive* frames (small motion, easy to
estimate), warp, and difference — what moves against the compensated background is
the drone, regardless of its colour.

`_MCMotion` implements 3-frame compensated differencing:
  * grid LK optical flow t-1->t and t-2->t, RANSAC homography (fallback: affine,
    then identity for low-texture sky),
  * warp the two past frames onto t, take min(|t - w1|, |t - w2|) so a blob must
    differ from *both* pasts at its current location (kills ghosts / revealed
    background), threshold at mean+k*std over the valid region, morphology,
    connected components filtered to drone-sized blobs.

Two registry methods use it:
  * ``mc-motion``  — proposals only (colour-invariant, high recall, no appearance).
  * ``mc-hybrid``  — proposals -> zoomed YOLO verification, unioned with a
    full-frame (or SAHI-tiled) YOLO pass. This is the moving-camera analogue of
    ``hybrid``.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..detections import Detection, nms
from .base import BaseMethod
from .yolo import COCO_DRONE_LIKE, YoloSahi, tile_origins


class _MCMotion:
    def __init__(self, k_sigma: float = 4.0, min_area: int = 3, max_area: int = 4000,
                 open_ksize: int = 3, dilate_ksize: int = 5, blur: int = 3,
                 grid_x: int = 32, grid_y: int = 18):
        self.k = k_sigma
        self.min_area, self.max_area = min_area, max_area
        self.open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        self.dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        self.blur = blur
        self.grid_x, self.grid_y = grid_x, grid_y
        self._g1: np.ndarray | None = None   # t-1
        self._g2: np.ndarray | None = None   # t-2

    def _register(self, src, dst):
        """Homography mapping src onto dst from grid LK + RANSAC; robust fallbacks."""
        h, w = dst.shape
        xs = np.linspace(w * 0.04, w * 0.96, self.grid_x)
        ys = np.linspace(h * 0.04, h * 0.96, self.grid_y)
        pts = np.array([[x, y] for y in ys for x in xs], np.float32).reshape(-1, 1, 2)
        nxt, st, err = cv2.calcOpticalFlowPyrLK(src, dst, pts, None, winSize=(21, 21),
                                                maxLevel=3,
                                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        good = (st.ravel() == 1)
        if good.sum() < 12:
            return None
        p0, p1 = pts[good], nxt[good]
        H, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
        if H is None or inl is None or int(inl.sum()) < 10:
            M, _ = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                               ransacReprojThreshold=3.0)
            if M is None:
                return None
            H = np.vstack([M, [0, 0, 1]])
        return H

    def process(self, frame_bgr: np.ndarray) -> list[Detection]:
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.blur:
            g = cv2.GaussianBlur(g, (self.blur, self.blur), 0)
        h, w = g.shape
        if self._g1 is None or self._g2 is None:
            self._g2, self._g1 = self._g1, g
            return []
        diffs, valid = [], np.ones((h, w), np.uint8)
        for past in (self._g1, self._g2):
            H = self._register(past, g)
            if H is None:
                warp = past
                vm = np.ones((h, w), np.uint8)
            else:
                warp = cv2.warpPerspective(past, H, (w, h), flags=cv2.INTER_LINEAR,
                                           borderValue=0)
                vm = cv2.warpPerspective(np.ones((h, w), np.uint8), H, (w, h)) > 0
            diffs.append(cv2.absdiff(g, warp).astype(np.float32))
            valid &= vm.astype(np.uint8)
        motion = np.minimum(diffs[0], diffs[1])
        valid = cv2.erode(valid, self.dil_k, iterations=2)
        motion *= valid
        v = motion[valid > 0]
        if v.size < 100:
            self._g2, self._g1 = self._g1, g
            return []
        thr = float(v.mean() + self.k * v.std())
        mask = (motion > max(thr, 6.0)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_k)
        mask = cv2.dilate(mask, self.dil_k)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (self.min_area <= area <= self.max_area):
                continue
            x, y, bw, bh = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                            int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            if max(bw, bh) > 120:            # too big to be a small drone blob
                continue
            score = float(motion[lab == i].mean())
            out.append(Detection(float(x), float(y), float(x + bw), float(y + bh), score, "motion"))
        self._g2, self._g1 = self._g1, g
        return out


class MCHybrid(BaseMethod):
    def __init__(self, name, weights=None, conf=0.02,
                 drone_classes: list[int] | None = None, device=0,
                 tile: int = 640, overlap: float = 0.25, crop_half: int = 48,
                 zoom: int = 4, max_crops: int = 24, confirm_dist: float = 16.0,
                 verify_only: bool = False, motion_kw: dict | None = None):
        from ultralytics import YOLO
        self.name = name
        self.model = None if weights is None else YOLO(weights)
        self.conf, self.classes, self.device = conf, drone_classes, device
        self.tile, self.overlap = tile, overlap
        self.crop_half, self.zoom, self.max_crops = crop_half, zoom, max_crops
        self.confirm_dist = confirm_dist
        self.verify_only = verify_only          # True => no full-frame pass (mc-motion+verify)
        self.motion = _MCMotion(**(motion_kw or {}))

    # -- full-frame SAHI-tiled appearance pass --------------------------------
    def _full_pass(self, frame):
        if self.model is None:
            return []
        h, w = frame.shape[:2]
        xs, ys = tile_origins(self.tile, w, self.overlap), tile_origins(self.tile, h, self.overlap)
        tiles, offs = [], []
        for y0 in ys:
            for x0 in xs:
                tiles.append(np.ascontiguousarray(frame[y0:y0 + self.tile, x0:x0 + self.tile]))
                offs.append((x0, y0))
        res = self.model(tiles, imgsz=self.tile, conf=self.conf, classes=self.classes,
                         device=self.device, verbose=False)
        out = []
        for r, (dx, dy) in zip(res, offs):
            for b in r.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                out.append(Detection(x1 + dx, y1 + dy, x2 + dx, y2 + dy,
                                     float(b.conf[0]), r.names[int(b.cls[0])]))
        return out

    def process(self, idx, frame_bgr, m_stab):
        h, w = frame_bgr.shape[:2]
        cands = sorted(self.motion.process(frame_bgr), key=lambda d: -d.score)[: self.max_crops]

        if self.model is None:                  # mc-motion: proposals only
            return nms(cands, dist_thresh=10.0)

        # zoomed YOLO verification of each motion candidate
        side = 2 * self.crop_half
        crops, origins = [], []
        for c in cands:
            x0 = int(np.clip(c.cx - self.crop_half, 0, max(w - side, 0)))
            y0 = int(np.clip(c.cy - self.crop_half, 0, max(h - side, 0)))
            crops.append(cv2.resize(frame_bgr[y0:y0 + side, x0:x0 + side], None,
                                    fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_CUBIC))
            origins.append((x0, y0))
        vconf = [0.0] * len(cands)
        if crops:
            for i, (res, (x0, y0), cand) in enumerate(
                    zip(self.model(crops, imgsz=side * self.zoom, conf=self.conf,
                                   classes=self.classes, device=self.device, verbose=False),
                        origins, cands)):
                for b in res.boxes:
                    bc = float(b.conf[0])
                    if bc < 0.20:            # strict: weak zoom-crop hits are clutter
                        continue             # hallucinations, not confirmations
                    bx = float(b.xyxy[0][0] + b.xyxy[0][2]) / 2 / self.zoom + x0
                    by = float(b.xyxy[0][1] + b.xyxy[0][3]) / 2 / self.zoom + y0
                    if (bx - cand.cx) ** 2 + (by - cand.cy) ** 2 <= self.confirm_dist ** 2:
                        vconf[i] = max(vconf[i], bc)
        # ADAPTIVE motion trust: unverified motion is reliable evidence only when
        # the scene is clean (few motion blobs -> near-static cam / uniform sky, as
        # in 10_06). When the frame is cluttered (many blobs -> aggressive ego-motion
        # over textured ground, as in NPS), motion is mostly parallax FP, so we
        # suppress unverified motion and let the appearance model rule.
        clean = max(0.0, 1.0 - max(0, len(cands) - 5) / 14.0)     # 5 blobs->1.0, >=19->0
        dets = []
        for cand, vc in zip(cands, vconf):
            s_m = min(1.0, cand.score / 25.0)
            if vc > 0:                      # motion AND appearance agree -> strongest, always trusted
                dets.append(Detection(cand.x1, cand.y1, cand.x2, cand.y2,
                                      0.6 + 0.4 * vc, "drone+motion"))
            elif not self.verify_only and clean > 0:
                dets.append(Detection(cand.x1, cand.y1, cand.x2, cand.y2,
                                      (0.28 + 0.42 * s_m) * clean, "motion"))
        if not self.verify_only:
            dets.extend(self._full_pass(frame_bgr))
        return nms(dets, dist_thresh=10.0)
