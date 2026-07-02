"""Hybrid: stabilized motion proposals verified by zoomed YOLO, unioned
with a full-frame YOLO pass.

The motion detector supplies high-recall candidate locations (it sees
2-pixel movers). Each candidate region is cropped at native resolution,
bicubically zoomed x4 (a 12 px drone becomes ~48 px -- comfortably inside
YOLO's working regime) and verified by the detector. The full-frame pass
covers static targets that motion cannot see (e.g. a parked drone).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..detections import Detection, nms
from ..motion import MotionDetector
from ..stabilize import warp_to_reference
from .base import BaseMethod
from .motion_only import map_back
from .yolo import COCO_DRONE_LIKE, _YoloBase


class HybridMethod(_YoloBase):
    def __init__(
        self,
        name: str,
        weights: str = "yolo11s.pt",
        conf: float = 0.02,
        drone_classes: list[int] | None = COCO_DRONE_LIKE,
        device: int | str = 0,
        full_imgsz: int = 1280,
        crop_half: int = 48,
        zoom: int = 4,
        max_crops: int = 16,
        confirm_dist: float = 14.0,
        motion_kw: dict | None = None,
        full_weights: str | None = None,
        full_classes: list[int] | None = COCO_DRONE_LIKE,
        sr_model_path: str | None = None,
    ):
        super().__init__(weights=weights, conf=conf, drone_classes=drone_classes,
                         device=device)
        # mixture of experts: an optional separate model for the full-frame
        # pass (e.g. tiny-specialist verifier + near/big-object expert)
        if full_weights is not None:
            from ultralytics import YOLO

            self.full_model = YOLO(full_weights)
            self.full_classes = full_classes
        else:
            self.full_model = self.model
            self.full_classes = drone_classes
        self.name = name
        self.full_imgsz = full_imgsz
        self.crop_half = crop_half
        self.zoom = zoom
        self.max_crops = max_crops
        self.confirm_dist = confirm_dist
        self.motion = MotionDetector(**(motion_kw or {}))
        # learned super-resolution for the verification crops (ablation
        # against plain bicubic; the research predicts marginal-at-best)
        self.sr = None
        if sr_model_path is not None:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(sr_model_path)
            self.sr.setModel("fsrcnn", int(zoom))

    def process(self, idx, frame_bgr, m_stab):
        h, w = frame_bgr.shape[:2]

        gray_stab = warp_to_reference(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), m_stab)
        cands = map_back(self.motion.process(gray_stab), m_stab)
        cands = sorted(cands, key=lambda d: -d.score)[: self.max_crops]

        # zoomed verification crops around each motion candidate
        crops, origins = [], []
        side = 2 * self.crop_half
        for c in cands:
            x0 = int(np.clip(c.cx - self.crop_half, 0, w - side))
            y0 = int(np.clip(c.cy - self.crop_half, 0, h - side))
            crop = frame_bgr[y0:y0 + side, x0:x0 + side]
            if self.sr is not None:
                crops.append(self.sr.upsample(crop))
            else:
                crops.append(cv2.resize(crop, None, fx=self.zoom, fy=self.zoom,
                                        interpolation=cv2.INTER_CUBIC))
            origins.append((x0, y0))

        confirmed_conf = [0.0] * len(cands)
        if crops:
            results = self._predict(crops, side * self.zoom)
            for i, (res, (x0, y0), cand) in enumerate(zip(results, origins, cands)):
                for b in res.boxes:
                    bx = float(b.xyxy[0][0] + b.xyxy[0][2]) / 2 / self.zoom + x0
                    by = float(b.xyxy[0][1] + b.xyxy[0][3]) / 2 / self.zoom + y0
                    if (bx - cand.cx) ** 2 + (by - cand.cy) ** 2 <= self.confirm_dist ** 2:
                        confirmed_conf[i] = max(confirmed_conf[i], float(b.conf[0]))

        dets: list[Detection] = []
        for cand, vconf in zip(cands, confirmed_conf):
            s_m = min(1.0, cand.score / 20.0)
            if vconf > 0:
                score = 0.5 + 0.5 * max(vconf, s_m)
                label = "drone+motion"
            else:
                score = 0.5 * s_m
                label = "motion"
            dets.append(Detection(cand.x1, cand.y1, cand.x2, cand.y2, score, label))

        # full-frame pass: static targets, plus anything motion missed
        res = self.full_model(
            frame_bgr, imgsz=self.full_imgsz, conf=self.conf,
            classes=self.full_classes, device=self.device, verbose=False,
        )[0]
        dets.extend(self._to_dets(res))
        return nms(dets, dist_thresh=10.0)
