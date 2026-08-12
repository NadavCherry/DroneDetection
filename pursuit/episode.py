"""One pursuit, start to finish: see the drone, close on it, hit it.

The loop here is the whole mission, and it is short on purpose -- every part of
it that could be interesting lives in :mod:`~pursuit.perception`,
:mod:`~pursuit.guidance`, :mod:`~pursuit.dynamics` or :mod:`~pursuit.evader`,
and what remains is the order they run in and how the result is scored.

## The order, and why it is this one

Each tick, the chaser acts on the frame it has *already* been given, and the
simulator renders the next one only after both aircraft have moved::

    estimate = perception(frame)     # what the camera showed
    command  = guidance(estimate)    # what to do about it
    chaser.step(command)             # fly it
    evader.step(evader_policy)       # the target flies too
    frame    = sim.step(both poses)  # what the camera shows now

The alternative -- render, then decide, then move within the same tick -- is one
tick of clairvoyance: the chaser would be reacting to the frame produced by the
positions it is about to fly to. On a 20 Hz loop that is 50 ms of free
information, which is enough to hide a real lag problem.

## Scoring, and why the sampled range is the wrong number

At 20 Hz with a 15 m/s closing speed the two aircraft jump three quarters of a
metre per tick, so a clean hit and a clean miss can produce the same *sampled*
minimum range -- the moment they were closest simply fell between two ticks.
:func:`segment_cpa` computes the true closest point of approach analytically
over each tick's straight-line segment, which is what ``miss_distance_m``
reports. A pursuit is scored on the geometry that happened, not on the geometry
that got sampled.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .dynamics import Airframe, BodyCommand, Limits
from .evader import Evader, EvaderConfig, evader_limits, make_evader
from .geometry import Intrinsics
from .guidance import HIT, GuidanceConfig, PursuitGuidance
from .perception import Perception, TargetEstimate


def segment_cpa(p0, p1, q0, q1) -> tuple:
    """Closest approach between two points moving in straight lines over one tick.

    ``p`` is the chaser and ``q`` the target; each moves from its ``0`` position
    to its ``1`` position at constant velocity over the interval. Returns
    ``(distance, u)`` with ``u`` in ``[0, 1]`` the fraction of the tick at which
    it happened.

    This is the difference between measuring a pursuit and measuring a sampling
    rate. Two aircraft closing at 15 m/s move 0.75 m between 20 Hz ticks; a pass
    that came within 20 cm can easily show 0.8 m at both surrounding samples.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    d0 = q0 - p0
    dv = (q1 - p1) - d0
    denom = float(dv @ dv)
    if denom <= 1e-12:
        return float(np.linalg.norm(d0)), 0.0
    u = float(-(d0 @ dv) / denom)
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return float(np.linalg.norm(d0 + dv * u)), u


def pass_geometry(p, vp, q, vq, yaw: float) -> dict:
    """Where the chaser actually passes the target, in the chaser's own axes.

    ``miss_distance_m`` is not this number, and the difference is large enough to
    mislead. The loop stops the instant a tick's segment CPA drops under the hit
    radius, so it scores the geometry at the *end of that tick* -- with the two
    aircraft still closing at 15 m/s and up to 50 ms of un-flown approach left.
    Measured across 223 intercepts, 89 percent of the reported miss was nothing
    but that un-flown forward distance: mean 0.615 m scored, against a true
    cross-track error near 0.10 m.

    That matters beyond bookkeeping. A scored miss that is dominated by
    along-track residue cannot tell you whether the aircraft is passing high,
    low or to one side -- the very thing you need in order to fix an aim bias.
    So this projects both aircraft forward at their current velocities, solves
    for the closest approach of the two straight lines, and decomposes the
    separation there into ``along`` (+ ahead), ``lateral`` (+ right) and
    ``vertical`` (+ chaser passes ABOVE the target).

    Constant velocity over the last few tens of milliseconds is a good model at
    these speeds; checked against a constant-acceleration extrapolation, the
    vertical component agreed to within 4 mm.
    """
    d = [float(q[i]) - float(p[i]) for i in range(3)]
    dv = [float(vq[i]) - float(vp[i]) for i in range(3)]
    denom = sum(c * c for c in dv)
    tstar = 0.0 if denom <= 1e-12 else max(0.0, -sum(d[i] * dv[i] for i in range(3)) / denom)
    m = [d[i] + dv[i] * tstar for i in range(3)]

    fwd = (math.cos(yaw), math.sin(yaw), 0.0)
    right = (math.sin(yaw), -math.cos(yaw), 0.0)
    return {
        "cpa_m": math.sqrt(sum(c * c for c in m)),
        "along_m": sum(m[i] * fwd[i] for i in range(3)),
        "lateral_m": sum(m[i] * right[i] for i in range(3)),
        # + means the chaser went over the top of the target.
        "vertical_m": -m[2],
        "dt_s": tstar,
    }


VERTICAL_AUTHORITY_RATIO = 2.57
"""Horizontal speed over vertical speed, for the interceptor's airframe.

Not a free parameter: 14.4 / 5.6 is the ratio the guidance law's vertical lead
was tuned against, and ``vertical_lead_scale`` exists precisely to compensate
for it. Letting it drift silently re-tunes the law.

And it *did* drift, because the chaser's climb rate was derived from the
**evader's**. A 9 m/s evader with a 1.6x advantage gives 14.4 over 5.6 -- the
tuned 2.57. A 12 m/s evader with 1.5x gives 18 over 5.25, a ratio of 3.43: an
interceptor a third weaker in the vertical channel than the one the lead was
computed for, flown against an intruder that spends the whole engagement
descending onto a rooftop. Measured over 24 rendered intercepts, it passed
**3.45 cm low, t = -8.76** -- small, and the most clearly biased thing in the
system.

Restoring the ratio is a no-op wherever it had not drifted, which is every suite
tuned before the city one.
"""


def _closure_rate(xyz, vel, target_xyz) -> float:
    """Speed made good from ``xyz`` toward ``target_xyz`` -- the component of
    velocity along the line to it, not the speed, which would over-state the
    margin for anything not flying straight at its objective."""
    d = [float(target_xyz[i]) - float(xyz[i]) for i in range(3)]
    n = math.sqrt(sum(c * c for c in d))
    if n <= 1e-9:
        return 0.0
    return sum(float(vel[i]) * d[i] / n for i in range(3))


@dataclass
class ScenarioConfig:
    """One pursuit to fly.

    Attributes:
        name: Label for the report.
        policy: Evader policy (see :mod:`~pursuit.evader`).
        seed: Seeds the evader's randomness and any sensor noise, so a scenario
            is a repeatable flight rather than a different one each run.
        start_range_m: Initial separation.
        start_bearing_deg: Where the target starts relative to the chaser's nose.
            Non-zero is the honest case: a mission that always begins with the
            target already centred never exercises acquisition.
        start_elevation_deg: Initial elevation of the target.
        altitude_m: Chaser's starting height above the scene's ground.
        evader_speed: Cruise speed of the target (m/s).
        speed_advantage: Chaser limits as a multiple of the evader's. Below 1.0
            a tail chase cannot close, which is kinematics and not a bug.
        max_seconds: Wall of the episode.
        hit_radius_m: Separation counted as a collision.
    """

    name: str = "default"
    policy: str = "flee"
    seed: int = 0
    start_range_m: float = 40.0
    start_bearing_deg: float = 0.0
    start_elevation_deg: float = 0.0
    altitude_m: float = 25.0
    evader_speed: float = 9.0
    speed_advantage: float = 1.6
    max_seconds: float = 45.0
    hit_radius_m: float = 1.0

    # -- where the two aircraft actually start ------------------------------
    chaser_offset_xy: Tuple[float, float] = (0.0, 0.0)
    """Chaser start, relative to the scene origin.

    Every scenario used to begin with the chaser at the origin facing world +x,
    which quietly made a whole class of results untrustworthy: the same patch of
    scenery sat behind the target in every run, so a detector that had learned
    that patch would look good for the wrong reason.
    """

    chaser_yaw_deg: float = 0.0
    """Chaser's initial heading. ``start_bearing_deg`` is measured from it, so
    existing scenarios are unchanged at the default of zero."""

    ingress: bool = False
    """Start the target outside the camera's field of view and fly it in.

    This is the realistic counter-UAS engagement and the one the earlier suites
    never posed: an intruder appears from somewhere, crosses the field of view,
    and only begins evading once it has been acquired. It exercises search,
    acquisition, and lock-onto-a-mover, none of which a target pre-placed at the
    centre of frame tests at all.
    """

    transit_miss_m: float = 0.0
    """Horizontal distance the ingress course would pass the chaser by. Zero is
    an inbound intruder flying straight at it; large values are a crossing pass."""

    transit_vertical_m: float = 0.0
    """Signed height offset of the ingress aim point. Negative with a high start
    is the "comes in from above and descends through the frame" case."""

    transit_ahead_m: float = 0.0
    """How far *in front of* the chaser the ingress course is aimed.

    Needed for any intruder that starts beyond the beam. A target beginning at
    115 degrees is behind the chaser, and a course aimed at a point beside it
    stays in the rear hemisphere for the whole run -- it arrives without ever
    having been in front of the camera. Aiming ahead forces the path to cross
    the forward field of view, which is the event the scenario exists to create.
    """

    transit_speed: Optional[float] = None
    """Speed on the ingress leg; defaults to ``evader_speed``."""

    reveal_range_m: float = 0.0
    """Also start evading when the chaser gets this close, lock or no lock. An
    intruder that lets a pursuer close to 20 m without reacting is a fiction; it
    would see it. Zero disables the range trigger."""

    entry: str = ""
    """Human label for how the intruder arrives (``left``/``right``/``high``/...)."""

    # -- point defence -------------------------------------------------------
    defend_xy: Optional[Tuple[float, float]] = None
    """XY of the asset the chaser is protecting, **relative to the scene origin**.

    Relative, like ``chaser_offset_xy``, and it was absolute until a real bug
    made the distinction unmissable: Rivermark's usable origin is (60, 60), not
    (0, 0), so a suite that built its assets around the origin and handed them
    over as world coordinates put every defended building 85 m from where the
    scenario said it was -- in the renderer only. The sandbox, whose origin *is*
    (0, 0), agreed with the scenario perfectly and reported nothing wrong.

    This turns the engagement into the mission the system is actually for. In
    every other scenario the intruder's business is with the chaser -- it flies
    at it, past it, or away from it -- which quietly makes the chaser the centre
    of the world and gives it unlimited time. A real intruder is not interested
    in the interceptor at all: it is going somewhere, and the interceptor's job
    is to reach it *first*. Setting this makes the intruder fly at a building
    and puts a clock on the whole engagement.
    """

    defend_height_m: float = 12.0
    """Height above ground of the aim point on the defended structure."""

    strike_radius_m: float = 10.0
    """How close the intruder must get to the defended asset to have hit it.

    Generous on purpose. The intruder is aiming at a building, not a point, and
    scoring a near miss as a save would flatter the interceptor for arriving
    late.
    """

    strike_commit: bool = False
    """The intruder keeps flying at the building after it has been detected.

    Off by default, which is the reconnaissance case: an intruder that notices
    an interceptor and breaks off into an evasion policy. On, it is a one-way
    strike aircraft -- it does not care about the interceptor, it is going
    somewhere, and the only way to stop it is to reach it first. That is the
    only setting in which "the building was hit" is a possible outcome, which
    makes it the only setting in which the mission is actually being scored.
    """

    strike_evade: float = 0.0
    """How hard a committed intruder jinks about its inbound course, as a
    fraction of its speed. Superimposed on the course, not replacing it."""

    defend_label: str = ""
    """Which structure this scenario defends, for the per-building breakdown."""

    calibrate_frames: int = 0
    """Frames of quiet observation before the engagement starts.

    The interceptor on overwatch has been sitting there. Its cameras have had
    time to learn what the city looks like when nothing is happening -- which is
    the entire premise of detecting a 3 px mover against a static background,
    and which every episode was previously denying it by starting the clock at
    the same instant the model was created.

    That denial was expensive and measurable. A background model needs its
    per-pixel noise estimate to converge (~20 frames) *and* its chronic-pixel
    map to converge (~100), and until the second one has, every genuinely noisy
    pixel in a rendered city is reported as a mover: a stationary camera on a
    static scene was returning **40-55 contacts a frame** at the moment the
    intruder appeared, and the drone was not reliably among them.

    So the model is warmed with the target parked out of sight first. Nothing
    about the engagement changes -- no clock runs, no command is issued, the
    tracker is cleared afterwards -- except that the sensor arrives at t=0
    having already seen the place it is looking at.
    """


def place_engagement(sc: "ScenarioConfig", origin_xy: Sequence[float],
                     ground_z: float, altitude_floor_m: float = 8.0
                     ) -> Tuple[Tuple[float, float, float], float,
                                Tuple[float, float, float], float,
                                Optional[Tuple[float, float, float]]]:
    """Where the two aircraft start, and where an intruder is heading.

    Returns ``(chaser_xyz, chaser_yaw, target_xyz, target_yaw, aim_xyz)``, with
    ``aim_xyz`` set only for an ingress scenario.

    Shared by the simulator (:class:`Episode`) and the headless
    :mod:`~pursuit.sandbox` deliberately. These were two copies of the same
    trigonometry, which is exactly the kind of duplication that lets a sandbox
    result stop describing the thing it is supposed to predict.

    ``start_bearing_deg`` is measured from the chaser's own heading rather than
    from world +x, so a scenario means the same engagement wherever in the scene
    it is flown and whichever way the chaser happens to be facing.
    """
    cx = float(origin_xy[0]) + float(sc.chaser_offset_xy[0])
    cy = float(origin_xy[1]) + float(sc.chaser_offset_xy[1])
    cz = ground_z + sc.altitude_m
    yaw = math.radians(sc.chaser_yaw_deg)

    b = yaw + math.radians(sc.start_bearing_deg)
    e = math.radians(sc.start_elevation_deg)
    r = sc.start_range_m
    tx = cx + r * math.cos(e) * math.cos(b)
    ty = cy + r * math.cos(e) * math.sin(b)
    tz = max(cz + r * math.sin(e), ground_z + altitude_floor_m)

    aim = None
    if sc.defend_xy is not None:
        # The intruder is going for the building, so the aim point is the
        # building -- fixed in the scene, not relative to the chaser. The
        # transit machinery is unchanged; only what it is aimed at differs,
        # which is the whole point of having expressed it as an aim point.
        aim = (float(origin_xy[0]) + float(sc.defend_xy[0]),
               float(origin_xy[1]) + float(sc.defend_xy[1]),
               ground_z + sc.defend_height_m)
    elif sc.ingress:
        # The aim point is offset from the chaser *along its own boresight's
        # right*, not perpendicular to the inbound course, and the difference is
        # not cosmetic. A target flying straight at an observer holds a constant
        # bearing -- that is what a collision course *is* -- so it would sit
        # frozen at the frame edge and never enter. What sweeps a target across
        # the frame is passing to one side, and to cross the boresight it has to
        # pass on the *opposite* side from where it started. Defining the offset
        # in the chaser's frame makes that directly expressible: positive
        # ``transit_miss_m`` means "passes this far to the right of where I am
        # looking", so an intruder entering from the left is given a positive
        # offset and is guaranteed to traverse the field of view.
        fx_, fy_ = math.cos(yaw), math.sin(yaw)     # along the boresight
        rx, ry = math.sin(yaw), -math.cos(yaw)      # right of the boresight
        aim = (cx + fx_ * sc.transit_ahead_m + rx * sc.transit_miss_m,
               cy + fy_ * sc.transit_ahead_m + ry * sc.transit_miss_m,
               max(cz + sc.transit_vertical_m, ground_z + 4.0))

    # A revealed evader flies away from the chaser; an intruder is inbound, so
    # point it at its aim point instead of over its shoulder.
    tyaw = math.atan2(aim[1] - ty, aim[0] - tx) if aim else b
    return (cx, cy, cz), yaw, (tx, ty, tz), tyaw, aim


@dataclass
class EpisodeResult:
    """What one pursuit did."""

    name: str
    policy: str
    seed: int
    success: bool = False
    outcome: str = "timeout"
    time_to_intercept_s: Optional[float] = None
    miss_distance_m: float = float("inf")
    min_sampled_range_m: float = float("inf")
    frames: int = 0
    detect_rate: float = 0.0
    """Fraction of frames the *detector* produced a usable box on, over the
    frames where the target was actually visible in the image."""
    track_rate: float = 0.0
    """Fraction of frames guidance had a valid estimate for, measured or coasted."""
    offtarget_rate: float = 0.0
    """Fraction of *tracked* frames whose belief was more than 1.5 degrees from
    the drone -- time spent confidently steering at something else.

    Reported next to ``track_rate`` because the two are easy to confuse and
    only their difference is informative. A run that tracked something on 92
    percent of frames and was on the drone for 1 percent of them is the
    single-camera failure mode this project has met more than once, and no
    other number in this record distinguishes it from a good run."""
    acquire_frame: Optional[int] = None
    acquire_time_s: Optional[float] = None
    max_bearing_err_deg: float = 0.0
    mean_range_err_m: Optional[float] = None
    """Monocular range error against truth -- the number that says whether the
    speed schedule and terminal trigger were being fed anything real."""
    pass_cpa_m: Optional[float] = None
    """True closest approach, extrapolated past the tick the hit was declared on.
    See :func:`pass_geometry` -- ``miss_distance_m`` is mostly un-flown forward
    distance and cannot tell you which way an aim bias points."""
    pass_along_m: Optional[float] = None
    pass_lateral_m: Optional[float] = None
    pass_vertical_m: Optional[float] = None
    """+ means the chaser passed ABOVE the target."""
    struck_asset: bool = False
    """The intruder reached what it was aiming at. A loss, however good the
    tracking looked."""
    strike_margin_s: Optional[float] = None
    """Seconds between the intercept and the intruder's projected arrival at the
    defended asset. This is the number the mission is actually judged on -- an
    intercept with 0.2 s to spare and one with 8 s to spare are not the same
    result, and the hit/miss column cannot tell them apart."""
    min_asset_range_m: Optional[float] = None
    """Closest the intruder ever got to the thing being defended."""
    reveal_time_s: Optional[float] = None
    """When an ingress target stopped transiting and began evading. Read it
    against ``acquire_time_s``: the gap is how long the intruder flew unopposed,
    and it is the acquisition cost measured in the only currency that matters."""
    stage_ms: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    note: str = ""


class Episode:
    """Fly one scenario against a live simulator.

    Args:
        client: Connected :class:`~simulators.pegasus.pursuit_proto.SimClient`.
        info: The simulator's ``info`` reply (intrinsics, ground height, dt).
        perception: The chaser's eye.
        guidance_cfg: Tuning for the intercept law.
        evader_cfg: Tuning for the target's flight.
        recorder: Optional callable invoked per frame with
            ``(frame, gt, estimate, guidance_state, perception)`` -- used to
            paint an annotated video without the loop knowing about drawing.
    """

    def __init__(self, client, info: dict, perception: Perception,
                 guidance_cfg: Optional[GuidanceConfig] = None,
                 evader_cfg: Optional[EvaderConfig] = None,
                 recorder: Optional[Callable] = None) -> None:
        self.client = client
        self.info = info
        self.perception = perception
        self.guidance_cfg = guidance_cfg or GuidanceConfig()
        self.evader_cfg = evader_cfg or EvaderConfig()
        self.recorder = recorder
        self.intr = Intrinsics.from_dict(info["intrinsics"])
        # A ring changes what "the frame" is -- four of them, in mount order --
        # so the observation handed to perception and to the recorder becomes a
        # dict rather than an array. Everything else about the loop is
        # unchanged, which is the point of having put a dataclass between the
        # sensor and the guidance law.
        self.ring = None
        if info.get("ring"):
            from .ring import Ring
            self.ring = Ring.from_info(info)
        self.dt = float(info["dt"])
        self.ground_z = float(info["ground_z"])
        self.origin_xy = tuple(info["origin_xy"])
        self.target_span_m = float(info["target_span_m"])
        self.telemetry: List[dict] = []

    # -- what the sensor was actually looking at -----------------------------

    OFF_TARGET_RAD = math.radians(1.5)
    """How far a belief may sit from the truth and still count as the drone.

    Generous next to a tracking error of one or two pixels, and deliberately so:
    this is not an accuracy metric, it is the difference between *the target*
    and *something else entirely*. A lock on a roofline is tens of degrees out.
    """

    def _truth_los(self, gt: dict):
        """Body-frame direction of the target, from the simulator's own geometry.

        The analytic projection rather than the rendered box, for the same
        reason ``visible`` uses it: at the ranges this mission is decided at
        there is no rendered box.
        """
        if self.ring is None:
            return None
        cam = self.ring.get(gt.get("camera") or "")
        uv = gt.get("analytic_uv") or gt.get("uv")
        if cam is None or uv is None:
            return None
        return cam.to_body(uv[0], uv[1])

    def _on_target(self, est: TargetEstimate, truth) -> bool:
        """Is the belief on the drone, or on something else?

        Answering this is what turns "tracked 92 percent of frames" from a
        compliment into a measurement. The single-camera runs learned the hard
        way that a seeker can hold a confident lock on a fixed feature for forty
        seconds, and every rate in the report looked healthy throughout.
        """
        if truth is None or est.los_body is None:
            return True
        from .geometry import angle_between
        return angle_between(est.los_body, truth) <= self.OFF_TARGET_RAD

    # -- setup --------------------------------------------------------------

    def _initial_poses(self, sc: ScenarioConfig):
        """Place both aircraft for the start of ``sc``.

        The chaser sits at the scene's origin looking along ``+x``; the target is
        put at the requested range, bearing and elevation *relative to that
        heading*, so ``start_bearing_deg`` means what it says regardless of where
        in the world the scene happens to put things.
        """
        cxyz, cyaw, txyz, tyaw, aim = place_engagement(
            sc, self.origin_xy, self.ground_z,
            self.evader_cfg.altitude_band[0])
        chaser = Airframe(xyz=cxyz, yaw=cyaw, ground_z=self.ground_z)
        target = Airframe(xyz=txyz, yaw=tyaw, ground_z=self.ground_z)
        return chaser, target, aim

    # -- the loop -----------------------------------------------------------

    def run(self, sc: ScenarioConfig) -> EpisodeResult:
        cfg_g = self.guidance_cfg
        cfg_e = EvaderConfig(**{**asdict(self.evader_cfg), "speed": sc.evader_speed})

        chaser, target, aim = self._initial_poses(sc)
        v_xy = cfg_e.speed * sc.speed_advantage
        chaser.limits = Limits(
            max_speed_xy=v_xy,
            max_speed_z=max(4.0, cfg_e.climb_mps * sc.speed_advantage,
                            v_xy / VERTICAL_AUTHORITY_RATIO),
            max_accel_xy=14.0 * sc.speed_advantage,
            max_accel_z=7.0,
            max_yaw_rate=2.5, max_yaw_accel=10.0, min_agl=2.0)
        target.limits = evader_limits(cfg_e)

        # Head the evader away from the chaser so `straight` is an escape rather
        # than a head-on merge that would intercept itself.
        heading0 = math.atan2(target.xyz[1] - chaser.xyz[1],
                              target.xyz[0] - chaser.xyz[0])
        evader = make_evader(sc.policy, sc.seed, self.ground_z, cfg_e,
                             heading0=heading0, centre_xy=self.origin_xy)
        if aim is not None:
            evader.arm_ingress(aim, sc.transit_speed or cfg_e.speed,
                               commit=sc.strike_commit, evade=sc.strike_evade)
        defend_xyz = None
        if sc.defend_xy is not None:
            defend_xyz = (self.origin_xy[0] + float(sc.defend_xy[0]),
                          self.origin_xy[1] + float(sc.defend_xy[1]),
                          self.ground_z + sc.defend_height_m)

        guidance = PursuitGuidance(self.intr, chaser.limits, self.target_span_m,
                                   cfg_g)
        self.perception.reset()
        self.telemetry = []

        obs_call = (self.client.reset_all if self.ring is not None
                    else self.client.reset)
        header, obs = obs_call(chaser.pose(), target.pose())
        gt = header["gt"]
        # Only a sensor with a background model has anything to calibrate. An
        # oracle run would otherwise spend two minutes an episode warming a
        # model it never consults.
        if sc.calibrate_frames > 0 and getattr(
                self.perception, "motion", None) is not None:
            self._calibrate(chaser, target, sc.calibrate_frames)
            header, obs = obs_call(chaser.pose(), target.pose())
            gt = header["gt"]

        res = EpisodeResult(name=sc.name, policy=sc.policy, seed=sc.seed,
                            config=asdict(sc))
        n_visible = n_detected = n_tracked = n_offtarget = 0
        range_errs: List[float] = []
        t = 0.0
        t0 = time.perf_counter()
        guidance_ms: List[float] = []
        max_frames = int(sc.max_seconds / self.dt)

        for idx in range(max_frames):
            est = self.perception.step(obs, idx, t, gt, ego_yaw=chaser.yaw,
                                       ego_speed=chaser.speed)
            _g0 = time.perf_counter()
            gs = guidance.step(t, self.dt, chaser.xyz, chaser.yaw, chaser.vel, est)
            guidance_ms.append((time.perf_counter() - _g0) * 1000.0)

            truth = self._truth_los(gt)
            # "Visible" means the target is geometrically inside a frame, not
            # that the renderer's box annotator managed to draw a box round it.
            # Those differ exactly where this mission lives: past about 120 m
            # the annotator returns nothing for a 3 px object, so a detection
            # rate scored on rendered boxes reads zero over the whole
            # acquisition phase -- the phase the mission is decided in.
            visible = bool(gt.get("visible")) or bool(gt.get("analytic_in_frame"))
            if visible:
                n_visible += 1
                if est.valid and est.source == "detector" and self._on_target(
                        est, truth):
                    n_detected += 1
            if est.valid:
                n_tracked += 1
                if truth is not None and not self._on_target(est, truth):
                    n_offtarget += 1
                if res.acquire_frame is None and guidance.confirmed:
                    res.acquire_frame, res.acquire_time_s = idx, round(t, 3)
                if gt.get("uv") and est.u is not None:
                    err = math.degrees(math.atan2(
                        math.hypot(est.u - gt["uv"][0], est.v - gt["uv"][1]),
                        self.intr.fx))
                    res.max_bearing_err_deg = max(res.max_bearing_err_deg, err)
                if est.source == "detector" and gs.range_est is not None:
                    range_errs.append(abs(gs.range_est - gt["range_m"]))

            self._log(idx, t, gt, est, gs, chaser, target, truth)
            if self.recorder is not None:
                self.recorder(obs, gt, est, gs, self.perception, chaser, target)

            if gs.mode == HIT:
                res.note = "guidance declared intercept"

            # An intruder starts evading when it has reason to: the chaser has
            # committed to a confirmed track, or it has closed near enough that
            # the intruder would plainly see it. Tying this to the *chaser's*
            # lock rather than to a stopwatch is what makes acquisition part of
            # the engagement -- a slow acquisition buys the target free
            # closing distance, and a fast one denies it.
            if not evader.revealed:
                if guidance.confirmed or (
                        sc.reveal_range_m > 0.0
                        and (gt.get("range_m") or 1e9) <= sc.reveal_range_m):
                    evader.reveal(t)
                    if res.reveal_time_s is None:
                        res.reveal_time_s = round(t, 3)

            if defend_xyz is not None:
                d_asset = math.dist(target.xyz, defend_xyz)
                res.min_asset_range_m = (d_asset if res.min_asset_range_m is None
                                         else min(res.min_asset_range_m, d_asset))
                if d_asset <= sc.strike_radius_m:
                    res.struck_asset = True
                    res.success = False
                    res.outcome = "target_struck"
                    res.note = (f"intruder reached the defended asset at "
                                f"t={t:.2f}s")
                    break

            # -- fly both aircraft -------------------------------------------
            p_before, q_before = chaser.xyz, target.xyz
            chaser.step(gs.command, self.dt)
            target.step(evader.command(t, target, chaser.xyz), self.dt)

            cpa, _u = segment_cpa(p_before, chaser.xyz, q_before, target.xyz)
            res.miss_distance_m = min(res.miss_distance_m, cpa)
            t += self.dt
            res.frames = idx + 1

            if cpa <= sc.hit_radius_m:
                res.success = True
                res.outcome = "intercept"
                res.time_to_intercept_s = round(t, 3)
                g = pass_geometry(chaser.xyz, chaser.vel, target.xyz,
                                  target.vel, chaser.yaw)
                res.pass_cpa_m = round(g["cpa_m"], 4)
                res.pass_along_m = round(g["along_m"], 4)
                res.pass_lateral_m = round(g["lateral_m"], 4)
                res.pass_vertical_m = round(g["vertical_m"], 4)
                if defend_xyz is not None:
                    # Time the intruder still needed to reach its objective,
                    # at the speed it was making good toward it.
                    remaining = math.dist(target.xyz, defend_xyz) - sc.strike_radius_m
                    closing = _closure_rate(target.xyz, target.vel, defend_xyz)
                    if closing > 0.1:
                        res.strike_margin_s = round(max(0.0, remaining) / closing, 2)
                break

            step = (self.client.step_all if self.ring is not None
                    else self.client.step)
            header, obs = step(chaser.pose(), target.pose())
            gt = header["gt"]
            res.min_sampled_range_m = min(res.min_sampled_range_m,
                                          float(gt["range_m"]))
        else:
            res.outcome = "timeout"

        res.detect_rate = round(n_detected / max(1, n_visible), 4)
        res.track_rate = round(n_tracked / max(1, res.frames), 4)
        res.offtarget_rate = round(n_offtarget / max(1, n_tracked), 4)
        res.mean_range_err_m = (round(float(np.mean(range_errs)), 3)
                                if range_errs else None)
        res.miss_distance_m = round(res.miss_distance_m, 4)
        res.min_sampled_range_m = round(res.min_sampled_range_m, 4)
        res.max_bearing_err_deg = round(res.max_bearing_err_deg, 3)
        res.stage_ms = self.perception.stage_report()
        gn = max(1, len(guidance_ms))
        res.stage_ms["guidance_ms"] = round(sum(guidance_ms) / gn, 3)
        # The number that matters for hardware: what rate the brain alone can
        # hold. `wall_fps` cannot answer that -- it is dominated by Isaac
        # rendering five frames per control tick, which is a property of the
        # test rig and will not exist on the aircraft.
        loop_ms = res.stage_ms["perception_ms"] + res.stage_ms["guidance_ms"]
        res.stage_ms["pipeline_ms"] = round(loop_ms, 2)
        res.stage_ms["pipeline_fps"] = round(1000.0 / loop_ms, 1) if loop_ms > 0 else 0.0
        if guidance_ms:
            g = sorted(guidance_ms)
            res.stage_ms["guidance_p95_ms"] = round(g[min(len(g) - 1, int(0.95 * len(g)))], 3)
            worst = (res.stage_ms.get("detect_p95_ms", 0.0)
                     + res.stage_ms.get("motion_p95_ms", 0.0)
                     + res.stage_ms.get("track_p95_ms", 0.0)
                     + res.stage_ms["guidance_p95_ms"])
            res.stage_ms["pipeline_p95_ms"] = round(worst, 2)
            res.stage_ms["pipeline_fps_p95"] = round(1000.0 / worst, 1) if worst > 0 else 0.0
        res.stage_ms["wall_fps"] = round(res.frames / max(1e-6, time.perf_counter() - t0), 2)
        if not res.success and res.outcome == "timeout" and n_tracked == 0:
            res.outcome = "never_acquired"
        return res

    def _calibrate(self, chaser, target, frames: int) -> None:
        """Let the sensor watch an empty sky before the engagement begins.

        The target is parked three kilometres away -- out of every camera and
        still inside the far clip plane -- so what the model learns is the city,
        not the drone. The tracker is cleared afterwards; the background model
        and its chronic-pixel map are exactly what is kept.
        """
        step = (self.client.step_all if self.ring is not None else self.client.step)
        far = {"xyz": [chaser.xyz[0] + 3000.0, chaser.xyz[1], target.xyz[2]],
               "yaw": 0.0}
        for i in range(int(frames)):
            header, obs = step(chaser.pose(), far)
            self.perception.step(obs, i, -1.0 - (frames - i) * self.dt,
                                 header["gt"], ego_yaw=chaser.yaw, ego_speed=0.0)
        reset_tracker = getattr(self.perception, "tracker", None)
        if reset_tracker is not None:
            reset_tracker.reset()

    def _log(self, idx, t, gt, est: TargetEstimate, gs, chaser, target,
             truth=None) -> None:
        self.telemetry.append({
            "frame": idx, "t": round(t, 3),
            "mode": gs.mode,
            "chaser": chaser.snapshot(),
            "target": target.snapshot(),
            "range_true": gt.get("range_m"),
            "range_est": None if gs.range_est is None else round(gs.range_est, 3),
            "gt_uv": gt.get("uv"),
            "gt_span": gt.get("span_px"),
            # A valid estimate does not guarantee a pixel address: a bearing is
            # a direction, and a direction can be outside every frame.
            "est_uv": (None if not (est.valid and est.u is not None)
                       else [round(est.u, 2), round(est.v, 2)]),
            "est_span": None if est.span_px is None else round(est.span_px, 2),
            "est_source": est.source,
            # Which of the four cameras had it, and whether the evidence was a
            # silhouette or a mover. With a ring these two are the first things
            # you want when a track dies, and neither is recoverable afterwards.
            "est_camera": est.camera or None,
            "est_kind": est.kind,
            "gt_camera": gt.get("camera"),
            "seen_by": gt.get("seen_by"),
            "score": round(est.score, 3),
            "pool": getattr(self.perception, "pool_state", None),
            "truth_cand": (self.perception.truth_report(truth, t)
                           if hasattr(self.perception, "truth_report")
                           else None),
            "los_rate": gs.los_rate,
            "age_s": gs.age_s,
            "lateral": gs.lateral_speed,
            "closing": gs.closing_speed,
            "boresight_deg": gs.boresight_deg,
            "cmd": [round(v, 3) for v in gs.command.as_tuple()],
            "note": gs.note,
        })

    def save_telemetry(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.telemetry), encoding="utf-8")
