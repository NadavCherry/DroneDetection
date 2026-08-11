"""Round-2 hybrid: dual motion proposals + 2-class expert verification.

Proposals: union of the slow-mover median detector (lagged background) and
MOG2 -- MOG2 sees almost everything (R~0.86 on the real drone) at terrible
precision; the verifier provides the precision.

Verifier: a 2-class (drone/bird) tiny specialist on 640 crops at 1:1
scale. Two modes:
  * single-frame (ft6): crop from the current color frame;
  * temporal (ft7): crop channels are STABILIZED grays at t-12/t-6/t --
    static scenery cancels to gray, movers leave color-fringed trails
    (the user's manual-labeling technique, formalized).

Full-frame pass: the near/big-object expert (ft1). Fusion produces
'drone+motion' (verified), 'bird' (classified away, low score),
'motion' (unverified) and full-frame detections, merged by center-NMS.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..detections import Detection, nms
from ..motion import MotionDetector
from ..stabilize import warp_to_reference
from .base import BaseMethod
from .motion_only import map_back

DT = 6  # temporal channel spacing (must match make_dataset_ft7)


class Hybrid2(BaseMethod):
    def __init__(
        self,
        name: str,
        weights: str,                  # 2-class tiny specialist (verifier)
        full_weights: str | None = None,  # near/big expert for full frames
        temporal: bool = False,
        conf: float = 0.02,
        device: int | str = 0,
        crop_half: int = 320,
        max_crops: int = 20,
        confirm_dist: float = 16.0,
        full_imgsz: int = 1280,
        slow_kw: dict | None = None,
    ):
        from ultralytics import YOLO

        self.name = name
        self.model = YOLO(weights)
        self.full_model = YOLO(full_weights) if full_weights else None
        self.temporal = temporal
        self.conf = conf
        self.device = device
        self.crop_half = crop_half
        self.max_crops = max_crops
        self.confirm_dist = confirm_dist
        self.full_imgsz = full_imgsz
        slow = dict(backend="median", lag=90, window=240, sample_stride=10,
                    warmup=30, flicker_alpha=0.004, flicker_rate0=0.30,
                    flicker_boost=3.0, max_dets=80)
        slow.update(slow_kw or {})
        self.motion_slow = MotionDetector(**slow)
        self.motion_mog2 = MotionDetector(backend="mog2", max_dets=120)
        self.grays: deque[tuple[int, np.ndarray]] = deque(maxlen=2 * DT + 1)

    def process(self, idx, frame_bgr, m_stab):
        h, w = frame_bgr.shape[:2]
        gray_stab = warp_to_reference(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
                                      m_stab)
        self.grays.append((idx, gray_stab))

        cands_stab = self.motion_slow.process(gray_stab) + \
            self.motion_mog2.process(gray_stab)
        cands_stab = nms(sorted(cands_stab, key=lambda d: -min(d.score / 20, 1)),
                         dist_thresh=10.0)
        # candidates in both coordinate frames (verification crops for the
        # temporal mode are cut from stabilized grays)
        cands = map_back(cands_stab, m_stab)
        pairs = sorted(zip(cands, cands_stab), key=lambda p: -p[0].score)[: self.max_crops]

        crops, metas = [], []
        side = 2 * self.crop_half
        mid = max(0, len(self.grays) - 1 - DT)
        for cand, cand_s in pairs:
            if self.temporal:
                # early frames clamp to the oldest available gray (matches
                # the v3 warmup training images)
                x0 = int(np.clip(cand_s.cx - self.crop_half, 0, w - side))
                y0 = int(np.clip(cand_s.cy - self.crop_half, 0, h - side))
                chans = [self.grays[0][1], self.grays[mid][1], self.grays[-1][1]]
                crop = np.dstack([c[y0:y0 + side, x0:x0 + side] for c in chans])
                metas.append((cand, cand_s, x0, y0, True))
            else:
                x0 = int(np.clip(cand.cx - self.crop_half, 0, w - side))
                y0 = int(np.clip(cand.cy - self.crop_half, 0, h - side))
                crop = frame_bgr[y0:y0 + side, x0:x0 + side]
                metas.append((cand, cand_s, x0, y0, False))
            crops.append(np.ascontiguousarray(crop))

        dets: list[Detection] = []
        verdicts = {}
        if crops:
            results = self.model(crops, imgsz=side, conf=self.conf,
                                 device=self.device, verbose=False)
            for res, (cand, cand_s, x0, y0, stab_coords) in zip(results, metas):
                best = {"drone": 0.0, "bird": 0.0}
                names = res.names
                ref = cand_s if stab_coords else cand
                for b in res.boxes:
                    bx = float(b.xyxy[0][0] + b.xyxy[0][2]) / 2 + x0
                    by = float(b.xyxy[0][1] + b.xyxy[0][3]) / 2 + y0
                    if (bx - ref.cx) ** 2 + (by - ref.cy) ** 2 <= self.confirm_dist ** 2:
                        cl = names[int(b.cls[0])]
                        best[cl] = max(best.get(cl, 0.0), float(b.conf[0]))
                verdicts[id(cand)] = best

        for cand, cand_s in pairs:
            best = verdicts.get(id(cand), {"drone": 0.0, "bird": 0.0})
            s_m = min(1.0, cand.score / 20.0)
            if best["drone"] >= max(0.5 * best["bird"], 0.10):
                dets.append(Detection(cand.x1, cand.y1, cand.x2, cand.y2,
                                      0.5 + 0.5 * max(best["drone"], s_m),
                                      "drone+motion"))
            elif best["bird"] >= 0.10:
                dets.append(Detection(cand.x1, cand.y1, cand.x2, cand.y2,
                                      0.12 + 0.08 * min(best["bird"], 1.0), "bird"))
            else:
                dets.append(Detection(cand.x1, cand.y1, cand.x2, cand.y2,
                                      0.5 * s_m, "motion"))

        if self.full_model is not None:
            res = self.full_model(frame_bgr, imgsz=self.full_imgsz, conf=self.conf,
                                  device=self.device, verbose=False)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                dets.append(Detection(x1, y1, x2, y2, float(b.conf[0]),
                                      res.names[int(b.cls[0])]))
        return nms(dets, dist_thresh=10.0)
