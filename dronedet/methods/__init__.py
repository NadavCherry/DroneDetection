"""Competing detector implementations behind a single interface.

Registry names:
    yolo-640        pretrained YOLO, full frame letterboxed to 640 (baseline)
    yolo-1280       pretrained YOLO, full frame at native 1280 width
    yolo-sahi       pretrained YOLO on overlapping 640 tiles (SAHI-style)
    motion-median   stabilized median-background + MAD-sigma motion detector
    motion-mog2     stabilized MOG2 motion detector
    hybrid          motion candidates + zoomed YOLO confirmation, union with
                    full-frame YOLO (classical proposal + deep verification)
    yolo-ft         fine-tuned YOLO (weights required), full frame at 1280
    yolo-ft-hybrid  motion candidates + zoomed fine-tuned-YOLO confirmation
"""

from __future__ import annotations

from .base import BaseMethod, run_method
from .motion_only import MotionMethod
from .yolo import YoloFullFrame, YoloSahi
from .hybrid import HybridMethod


def build_method(name: str, **kw) -> BaseMethod:
    if name == "yolo-640":
        return YoloFullFrame("yolo-640", imgsz=640, **kw)
    if name == "yolo-1280":
        return YoloFullFrame("yolo-1280", imgsz=1280, **kw)
    if name == "yolo-sahi":
        return YoloSahi("yolo-sahi", **kw)
    if name == "motion-median":
        return MotionMethod("motion-median", backend="median", **kw)
    if name == "motion-slow":
        slow = dict(backend="median", lag=90, window=240, sample_stride=10,
                    warmup=30, flicker_alpha=0.004, flicker_rate0=0.30,
                    flicker_boost=3.0, max_dets=80)
        slow.update(kw)
        return MotionMethod("motion-slow", **slow)
    if name == "motion-mog2":
        return MotionMethod("motion-mog2", backend="mog2", **kw)
    if name == "hybrid":
        return HybridMethod("hybrid", **kw)
    if name == "yolo-ft":
        return YoloFullFrame("yolo-ft", imgsz=1280, drone_classes=None, **kw)
    if name == "yolo-ft-hybrid":
        return HybridMethod("yolo-ft-hybrid", drone_classes=None, **kw)
    if name == "yolo-ft-sahi":
        return YoloSahi("yolo-ft-sahi", drone_classes=None, **kw)
    if name == "sr-hybrid":
        return HybridMethod("sr-hybrid", **kw)
    if name == "moe2-hybrid":
        from .hybrid2 import Hybrid2

        return Hybrid2("moe2-hybrid", temporal=False, **kw)
    if name == "moe3-stacked":
        from .hybrid2 import Hybrid2

        return Hybrid2("moe3-stacked", temporal=True, **kw)
    if name in ("mc-motion", "mc-hybrid", "mc-verify"):
        from .mc_hybrid import MCHybrid

        opts = dict(drone_classes=None)
        if name == "mc-motion":
            opts["weights"] = None                 # motion proposals only
        if name == "mc-verify":
            opts["verify_only"] = True             # motion -> YOLO verify, no full-frame pass
        opts.update(kw)
        return MCHybrid(name, **opts)
    raise ValueError(f"unknown method: {name}")
