"""Pretrained/fine-tuned YOLO methods: full-frame and SAHI-style tiling.

COCO has no ``drone`` class; a small quadcopter is typically fired as
``airplane`` (4), ``bird`` (14) or ``kite`` (33) at low confidence, so
pretrained variants restrict to those classes with a very low threshold.
Fine-tuned weights (single ``drone`` class) pass ``drone_classes=None``.
"""

from __future__ import annotations

import numpy as np

from ..detections import Detection, nms
from .base import BaseMethod

COCO_DRONE_LIKE = [4, 14, 33]  # airplane, bird, kite


class _YoloBase(BaseMethod):
    def __init__(self, weights: str = "yolo11s.pt", conf: float = 0.02,
                 drone_classes: list[int] | None = COCO_DRONE_LIKE,
                 device: int | str = 0):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.classes = drone_classes
        self.device = device

    def _predict(self, imgs, imgsz: int):
        return self.model(
            imgs, imgsz=imgsz, conf=self.conf, classes=self.classes,
            device=self.device, verbose=False,
        )

    @staticmethod
    def _to_dets(result, dx: float = 0.0, dy: float = 0.0) -> list[Detection]:
        out = []
        names = result.names
        for b in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append(Detection(x1 + dx, y1 + dy, x2 + dx, y2 + dy,
                                 float(b.conf[0]), names[int(b.cls[0])]))
        return out


class YoloFullFrame(_YoloBase):
    def __init__(self, name: str, imgsz: int = 640, **kw):
        super().__init__(**kw)
        self.name = name
        self.imgsz = imgsz

    def process(self, idx, frame_bgr, m_stab):
        res = self._predict(frame_bgr, self.imgsz)[0]
        return self._to_dets(res)


def tile_origins(size: int, total: int, overlap: float) -> list[int]:
    """Start offsets covering [0, total] with ``size`` tiles and >= overlap."""
    if total <= size:
        return [0]
    stride = int(size * (1 - overlap))
    xs = list(range(0, total - size, stride))
    xs.append(total - size)
    return sorted(set(xs))


class YoloSahi(_YoloBase):
    """Overlapping fixed-size tiles at native resolution, batched in one
    forward pass, merged with center-distance NMS (SAHI's slicing strategy;
    implemented directly for batching speed and control)."""

    def __init__(self, name: str, tile: int = 640, overlap: float = 0.25, **kw):
        super().__init__(**kw)
        self.name = name
        self.tile = tile
        self.overlap = overlap

    def process(self, idx, frame_bgr, m_stab):
        h, w = frame_bgr.shape[:2]
        xs = tile_origins(self.tile, w, self.overlap)
        ys = tile_origins(self.tile, h, self.overlap)
        tiles, offs = [], []
        for y0 in ys:
            for x0 in xs:
                tiles.append(np.ascontiguousarray(
                    frame_bgr[y0:y0 + self.tile, x0:x0 + self.tile]))
                offs.append((x0, y0))
        results = self._predict(tiles, self.tile)
        dets: list[Detection] = []
        for res, (dx, dy) in zip(results, offs):
            dets.extend(self._to_dets(res, dx, dy))
        return nms(dets, dist_thresh=10.0)
