"""Classical stabilized motion detection as a standalone method."""

from __future__ import annotations

import cv2

from ..motion import MotionDetector
from ..stabilize import apply_to_points, invert, warp_to_reference
from ..detections import Detection
from .base import BaseMethod


class MotionMethod(BaseMethod):
    def __init__(self, name: str, backend: str = "median", **kw):
        self.name = name
        self.det = MotionDetector(backend=backend, **kw)

    def process(self, idx, frame_bgr, m_stab):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_stab = warp_to_reference(gray, m_stab)
        dets_stab = self.det.process(gray_stab)
        return map_back(dets_stab, m_stab)


def map_back(dets_stab: list[Detection], m_stab) -> list[Detection]:
    """Stabilized (reference) coords -> original frame coords."""
    minv = invert(m_stab)
    out = []
    for d in dets_stab:
        (x1, y1), (x2, y2) = apply_to_points(minv, [[d.x1, d.y1], [d.x2, d.y2]])
        out.append(Detection(x1, y1, x2, y2, d.score, d.label))
    return out
