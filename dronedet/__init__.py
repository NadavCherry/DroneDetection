"""Tiny-drone detection and tracking pipeline.

Modules:
    video      -- sequential video reading
    stabilize  -- global camera-motion estimation (translation / affine)
    motion     -- stabilized background-model motion detector for tiny targets
    detections -- detection data model + JSON serialization
    methods    -- competing detector implementations behind one interface
    gt         -- ground-truth store
    evaluate   -- center-distance based per-frame evaluation
    track      -- camera-motion-compensated Kalman tracker
    viz        -- overlays and video writing
"""

__version__ = "0.1.0"
