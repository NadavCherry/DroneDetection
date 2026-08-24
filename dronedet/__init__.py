"""Tiny-drone detection and tracking pipeline.

Modules:
    video      -- sequential video reading
    stabilize  -- global camera-motion estimation (translation / affine)
    motion     -- stabilized background-model motion detector for tiny targets
    detections -- detection data model + JSON serialization
    methods    -- competing detector implementations behind one interface
    gt         -- ground-truth store
    evaluate   -- center-distance based per-frame evaluation
    metrics    -- benchmark-grade scoring: centre-distance AND IoU/COCO AP, AI-TOD
                  size bins, distractor (bird) accounting, bootstrap intervals
    track      -- camera-motion-compensated Kalman tracker
    viz        -- overlays and video writing
"""

# 2.0.0: the SOTA campaign. Every benchmark number in the README now comes from this
# repository's own runs -- ours AND the competitor's -- under one evaluator, with paired
# statistics. The 0.x claims this replaces were retired as disputed in the 2026-08 audit.
__version__ = "2.0.0"
