"""Edge / real-time variants of the HiveLab tiny-drone pipeline.

Target class: Jetson Orin Nano or weaker. Everything here is designed
around the PC pipeline's measured cost profile (76% = verifier crops,
9% = full-frame expert, 7% = full-res stabilization):

    rt_stabilize  -- phase correlation on a downscaled gray (~1 ms)
    rt_motion     -- O(1)/frame lagged-EMA background (no periodic median)
    rt_models     -- TRT / torch wrappers for the nano detectors
    pipelines     -- RT-A .. RT-F pipeline definitions
    runner        -- timed video runner producing detections + stage stats

Training uses data/videos/07_05.mp4 ONLY (time-split). data/videos/10_06.mp4 is a pure test set.
"""

__version__ = "0.1.0"
