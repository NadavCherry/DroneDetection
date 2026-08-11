"""Closed-loop air-to-air pursuit: detect a drone, track it, and fly into it.

The detection half of this repository answers "is there a drone in this video".
This package answers the next question -- "can the aircraft carrying the camera
*catch* it" -- by closing the loop: the chaser's own camera is the only sensor,
its output steers the chaser, and the chaser's motion changes what the camera
sees on the next frame. Nothing here is scored offline; a run either ends with
the two aircraft touching or it does not.

The pieces, in the order a frame passes through them:

:mod:`~pursuit.perception`
    Frame -> one :class:`~pursuit.perception.TargetEstimate` (a bearing, a pixel
    span, a flag). A YOLO plus a single-target image-plane tracker that coasts
    through the frames the detector misses. Swap in
    :class:`~pursuit.perception.OracleDetector` to fly the same guidance against
    a perfect sensor -- which is how a guidance bug is told from a vision one.
:mod:`~pursuit.guidance`
    Estimate -> body velocity. Proportional navigation rather than the
    reference stack's centre-and-advance visual servo, because the target moves
    (see that module's docstring for why that changes the law and not just the
    gains), wrapped in the search / acquire / pursue / terminal mode logic.
:mod:`~pursuit.dynamics`
    Body velocity -> pose, under speed, acceleration and yaw-rate limits, so the
    guidance law is tested against an aircraft rather than a teleport.
:mod:`~pursuit.evader`
    What the fleeing drone does -- a ladder of policies from a straight line to
    a break turn, each of which breaks a different guidance law.
:mod:`~pursuit.episode`
    The loop itself, and the scoring: closest point of approach between ticks,
    not the sampled range.

The simulator runs in the Isaac Sim container as a long-lived render server; see
:mod:`simulators.pegasus.pursuit_proto` for why the two halves are separate
processes and how they talk.
"""

from .dynamics import Airframe, BodyCommand, Limits
from .episode import Episode, EpisodeResult, ScenarioConfig
from .evader import Evader, EvaderConfig, make_evader
from .geometry import Intrinsics
from .guidance import GuidanceConfig, PursuitGuidance
from .perception import (
    Box,
    OracleDetector,
    Perception,
    TargetEstimate,
    YoloDetector,
    build_detector,
)

__all__ = [
    "Airframe", "BodyCommand", "Limits",
    "Episode", "EpisodeResult", "ScenarioConfig",
    "Evader", "EvaderConfig", "make_evader",
    "Intrinsics",
    "GuidanceConfig", "PursuitGuidance",
    "Box", "OracleDetector", "Perception", "TargetEstimate", "YoloDetector",
    "build_detector",
]
