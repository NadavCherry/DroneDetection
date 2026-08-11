"""Four cameras, one target: seeing all the way round without turning round.

The single forward camera made *pointing* part of the mission. A target outside
its 76 degree cone did not exist, so the interceptor had to search by yawing --
a step-and-stare pattern that takes seconds to sweep a circle, during which an
intruder that is already inbound keeps coming. Every acquisition failure in
``pursuit/README.md`` is some version of that sentence.

A ring of four wide cameras removes the mechanism rather than tuning it. There
is no direction the aircraft cannot see, so SEARCH stops being a manoeuvre and
becomes a state of attention, and the aircraft's heading is free to do something
useful instead. What it costs, and what has to be built to pay for it, is what
this module is:

**A common frame.** Four cameras produce four pixel coordinates that mean
nothing to each other. Everything here converts immediately to a unit direction
in *body* axes and stays there. The tracker gates on the great-circle angle
between directions, so a target crossing a seam is one track the whole way
across, not two tracks and a gap.

**Overlap.** The seams overlap by 6 degrees on purpose -- a hole would be a
direction an intruder arrives from unseen -- which means a target in a seam is
genuinely detected twice. Those two detections are the *same drone*, and feeding
both to a tracker is how a single object becomes two tracks and an oscillating
lock. :func:`fuse_detections` merges them by angle before anything downstream
ever sees them.

**A detector that works at 3 pixels.** The ring trades angular resolution for
coverage (see ``simulators.pegasus.camera.RING_RESOLUTION``), so an intruder at
100 m is 3.4 px and no appearance model will find it. But an interceptor holding
station has four *stationary* cameras looking at a city that is not going
anywhere, and a small mover against a static background is the problem this
whole repository was built to solve. :class:`RingMotionDetector` is that half;
the YOLO is aimed by it rather than sweeping for it.

So the division of labour is: **motion proposes, appearance disposes, and only
appearance is allowed to say how far away something is.** A dilated blob's width
is a property of the morphology kernel, not of the drone, and letting it set the
monocular range would drive the speed schedule off a threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import (Intrinsics, angle_between, bearing_to_body, body_bearing,
                       body_to_pixel, offaxis_scale, pixel_to_body, wrap_pi,
                       yaw_homography)
from .perception import Box, TargetEstimate


# --------------------------------------------------------------------- layout

@dataclass(frozen=True)
class RingCamera:
    """One camera on the ring: where it points and what it sees through."""

    name: str
    mount_yaw: float          # radians CCW from the nose
    intr: Intrinsics

    def to_body(self, u: float, v: float) -> Tuple[float, float, float]:
        return pixel_to_body(self.intr, u, v, self.mount_yaw)

    def to_pixel(self, los_body) -> Optional[Tuple[float, float]]:
        return body_to_pixel(self.intr, los_body, self.mount_yaw)

    def sees(self, los_body, margin_px: float = 0.0) -> bool:
        uv = self.to_pixel(los_body)
        if uv is None:
            return False
        return (margin_px <= uv[0] < self.intr.width - margin_px
                and margin_px <= uv[1] < self.intr.height - margin_px)


@dataclass
class Ring:
    """The camera set, in the simulator's mount order.

    Order is part of the wire format -- the frames arrive concatenated in it --
    so it is carried rather than re-derived. A ring one camera out of step
    steers 90 degrees away from its target and looks completely healthy doing
    it.
    """

    cameras: Tuple[RingCamera, ...]

    @classmethod
    def from_info(cls, info: dict) -> "Ring":
        """Build from the simulator's ``info`` reply."""
        intr = Intrinsics.from_dict(info["intrinsics"])
        mounts = info.get("cameras") or [{"name": "nose", "yaw_deg": 0.0}]
        return cls(tuple(RingCamera(m["name"], math.radians(m["yaw_deg"]), intr)
                         for m in mounts))

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(c.name for c in self.cameras)

    def get(self, name: str) -> Optional[RingCamera]:
        for c in self.cameras:
            if c.name == name:
                return c
        return None

    def seeing(self, los_body, margin_px: float = 0.0) -> List[RingCamera]:
        """Every camera with this direction inside its frame -- one or two."""
        return [c for c in self.cameras if c.sees(los_body, margin_px)]

    def owner(self, los_body) -> Optional[RingCamera]:
        """The camera whose *centre* this direction is nearest.

        The same tie-break the simulator uses to pick which camera's ground
        truth is reported, and it has to be the same one or the label and the
        detection describe different images. Most-centred, so ownership changes
        hands exactly once per seam crossing, at the bisector.
        """
        best, best_k = None, -float("inf")
        for c in self.cameras:
            uv = c.to_pixel(los_body)
            if uv is None:
                continue
            du = abs(uv[0] - c.intr.cx) / (0.5 * c.intr.width)
            dv = abs(uv[1] - c.intr.cy) / (0.5 * c.intr.height)
            k = -max(du, dv)
            if k > best_k:
                best, best_k = c, k
        # The least-bad camera, not "none of them". A direction can be outside
        # every frame -- the ring covers the whole circle but only +/-20.9
        # degrees of elevation, and a coasted prediction is free to drift above
        # that. "Which camera would be looking at this" still has an answer
        # there, and returning None instead made a coast report a valid
        # estimate with no pixel coordinates, which the telemetry writer then
        # tried to round.
        return best

    def coverage_deg(self) -> Tuple[float, float]:
        n = len(self.cameras)
        hfov = self.cameras[0].intr.hfov_deg if n else 0.0
        return (n * hfov, hfov - 360.0 / max(1, n))


def to_world(los_body, yaw: float) -> Tuple[float, float, float]:
    """Rotate a body direction into world axes for a level, yawed airframe."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (los_body[0] * c - los_body[1] * s,
            los_body[0] * s + los_body[1] * c, los_body[2])


RING_MOUNT_NAMES = ("fwd", "left", "aft", "right")
"""Body-relative names, matching the simulator's mount order exactly."""


def default_ring(intr: Intrinsics, n: int = 4) -> Ring:
    """``n`` copies of ``intr`` spaced evenly, nose first.

    Mirrors ``simulators.pegasus.camera.ring_mounts`` so the fast loop and the
    renderer describe the same aircraft. ``test_ring.py`` asserts they agree
    rather than trusting that two lists stay in step.
    """
    step = 2.0 * math.pi / n
    return Ring(tuple(RingCamera(RING_MOUNT_NAMES[i] if n == 4 else f"cam{i}",
                                 i * step, intr) for i in range(n)))


# ----------------------------------------------------------------- detections

@dataclass
class RingDetection:
    """One candidate, already in the frame everything downstream reasons in."""

    los: Tuple[float, float, float]
    """Unit direction, body axes."""
    score: float
    camera: str
    u: float
    v: float
    span_px: Optional[float] = None
    """None for a motion blob. A dilated connected component's width is a
    property of the morphology kernel, so it is not allowed to imply a range."""
    kind: str = "appearance"
    """``appearance`` or ``motion``."""
    span_rad: float = 0.0
    """Angular size, which *is* comparable across cameras and off-axis
    positions -- unlike the pixel span, which a 96 degree lens stretches by
    ``sec^2`` toward the edge of frame."""
    confirmed: bool = True
    """Something was here on a recent frame too.

    A renderer's temporal noise is *uncorrelated* between frames and a drone is
    not: at long range the target moves a fifth of a pixel a tick, so it lands
    in almost exactly the same place twice running, while a noise blob does not
    land anywhere twice. Measured on an empty Rivermark sky, that one test is
    the difference between eight false contacts a frame and effectively none.

    It is a *tag*, not a filter, and the distinction is what makes it safe:
    an unconfirmed detection can still feed a track that already exists -- where
    the tracker's own gate is doing the work -- but only a confirmed one may
    create a new candidate. Recall where it is cheap, precision where it
    matters."""


def fuse_detections(dets: Sequence[RingDetection], ring: Ring,
                    merge_rad: float = math.radians(0.8)) -> List[RingDetection]:
    """Collapse detections of the same object seen by more than one camera.

    Necessary because the seams overlap on purpose. A drone sitting in the
    6 degree band between two cameras is detected twice, at two pixel addresses
    in two images, and both are correct -- but a tracker handed both sees a pair
    of objects a fraction of a degree apart and spends the engagement
    alternating between them.

    Merging is by angle rather than by pixel distance because the two detections
    are not in the same image and there is no pixel distance between them. The
    gate is the sum of the merge tolerance and the pair's angular sizes, so a
    large close target -- whose two boxes legitimately disagree by more -- still
    merges.

    Within a cluster the survivor is the detection from the camera that has the
    object most centred: least lens stretch, least chance of a clipped box, and
    the same choice the simulator's ground truth makes.
    """
    out: List[RingDetection] = []
    for d in sorted(dets, key=lambda x: -x.score):
        for i, keep in enumerate(out):
            gate = merge_rad + 0.5 * (d.span_rad + keep.span_rad)
            if angle_between(d.los, keep.los) <= gate:
                out[i] = _prefer(keep, d, ring)
                break
        else:
            out.append(d)
    return out


def _prefer(a: RingDetection, b: RingDetection, ring: Ring) -> RingDetection:
    """Pick the better of two views of one object, keeping the useful parts.

    An appearance detection wins outright over a motion blob whatever the
    centring, because only it carries a span worth believing. Between two of a
    kind, the more centred camera wins -- and either way the pair keeps whatever
    span either of them managed to measure, since a motion blob that merged with
    a YOLO box is a *confirmed* motion blob and should not throw the box away.
    """
    def centrality(d: RingDetection) -> float:
        cam = ring.get(d.camera)
        if cam is None:
            return -1.0
        du = abs(d.u - cam.intr.cx) / (0.5 * cam.intr.width)
        dv = abs(d.v - cam.intr.cy) / (0.5 * cam.intr.height)
        return -max(du, dv)

    if (a.kind == "appearance") != (b.kind == "appearance"):
        win, lose = (a, b) if a.kind == "appearance" else (b, a)
    else:
        win, lose = (a, b) if centrality(a) >= centrality(b) else (b, a)
    return RingDetection(
        los=win.los, score=max(a.score, b.score), camera=win.camera,
        u=win.u, v=win.v,
        span_px=win.span_px if win.span_px is not None else lose.span_px,
        kind=win.kind, span_rad=max(a.span_rad, b.span_rad))


# ------------------------------------------------------------ motion detector

@dataclass
class MotionConfig:
    """Tuning for :class:`RingMotionDetector`.

    Two of these defaults are the difference between a 46 m defended radius and
    nothing at all, and both were inherited from ``dronedet``'s moving-camera
    detector, where they are right. Measured here on identical Rivermark frames,
    reliable detection range against a 12 m/s inbound intruder:

    | scale | opening | reliable to | first seen |
    |---|---|---|---|
    | **1.0** | **none** | **100 m** | **167 m** |
    | 1.0 | 3x3 | 80 m | 159 m |
    | 0.5 | none | 80 m | 137 m |
    | 0.5 | 3x3 | never 50% | 99 m |

    A morphological **opening deletes anything smaller than its kernel**, and at
    100 m the target *is* smaller than a 3x3 kernel -- so the step that exists to
    remove speckle was removing the drone. And halving the frame turns a 4 px
    target into 2 px and averages its contrast down with it. Both are sensible
    economies against a 20 px drone from a moving camera; against a 3 px drone
    from a stationary one they are the whole problem.

    Attributes:
        scale: Work on a downscaled greyscale. 1.0 -- see above. Kept as a knob
            because a moving aircraft has a bigger, closer target and can afford
            the economy, but the default is the one the acquisition phase needs.
        k_static: Threshold in robust sigmas above the background when the
            aircraft is holding station. Low, because a stationary camera
            cancels a stationary city *exactly* and anything left is a mover.
        k_moving: The same when it is flying. Higher, because translation
            introduces parallax that no single homography can remove and every
            near roofline lights up.
        min_area / max_area: Blob size, in scaled pixels. The lower bound is 3
            at full resolution: an intruder at 140 m is 3.3 px across and covers
            about eight pixels above threshold, so a floor much higher would
            quietly define the acquisition range instead of measuring it.
        max_span_px: Full-resolution span above which a blob is not a distant
            drone -- it is a cloud edge, a passing car, or the whole skyline
            shifting.
        border_px: Ignore this margin, scaled. Warping a past frame leaves an
            invalid wedge at the edge whose width is the rotation, and a
            difference against nothing is the largest difference there is.
        open_ksize: Morphological opening kernel, or 0 to skip it -- which is
            the default, see above. The area filter rejects single-pixel noise
            without also rejecting the target.
        refine_lk: Estimate a residual homography with grid LK on top of the
            analytic yaw rotation. Off by default and measured that way: the
            rotation is *known exactly* from the aircraft's own heading, which
            is the part that matters, and LK on an empty sky contributes noise
            and 40 ms.
        score_base / score_per_hit / score_max: How a motion track earns
            confidence. See :class:`RingTracker`.
    """

    background: bool = True
    """Model the background while the aircraft holds station, instead of
    differencing consecutive frames.

    The single most valuable line in this class, and it is the detection half of
    this repository's own thesis applied where it finally fits. Frame
    differencing sees the *change between two frames*: for an intruder on a
    near-constant bearing -- which is what an inbound one is -- that is a
    fraction of a pixel of drift and a hair of growth, so a target with 60 grey
    levels of contrast against the sky produces a difference of five. A
    background model sees the *whole* contrast, every frame, because the drone
    is not in the background.

    It only works from a camera that is not moving, which is exactly the
    situation this mission puts the interceptor in and exactly the situation the
    forward-camera pursuit never had. When the aircraft flies, the model is
    thrown away and the ego-compensated differencing below takes over -- by then
    the target is close and the appearance detector is carrying it anyway.
    """

    bg_step: float = 1.0
    """Sigma-delta background update, grey levels per frame.

    An approximate running median: step the estimate one level toward each new
    sample. O(1) in memory and one pass in time, where a true rolling median
    over 30 frames of four 1.4 Mpx cameras is 700 MB and half a second of
    ``np.median`` -- which is the honest reason this is not simply
    ``dronedet.motion.MotionDetector``, whose statistics are better and whose
    cost is not a 20 Hz control loop's to pay.
    """

    bg_step_fg: float = 0.04
    """The same, at pixels currently called foreground. Not zero and not one.

    Zero would be a selective update, and a selective update is a trap: a false
    positive freezes its own pixels forever and becomes permanent. One would
    let the background learn the target -- and it would, because an intruder
    holding a constant bearing sits on the same pixels for seconds, which is
    precisely the geometry this detector exists for. A slow update expires a
    stuck artefact in a few seconds and takes minutes to absorb a drone.
    """

    fg_freeze_max: int = 30
    """Frames a pixel may stay foreground before its noise estimate resumes.

    Thirty is a second and a half: three times longer than a target spends
    crossing any one pixel at long range, and infinitely shorter than for ever,
    which is how long a stuck pixel would otherwise hold its exemption.
    """

    bg_warmup: int = 25
    """Frames of watching before the model is allowed to report anything.

    Sized from the noise estimate's own time constant, not from impatience.
    ``mad`` starts at the floor and converges at ``mad_alpha``, so for the first
    ~20 frames every pixel's threshold is the floor and the model reports the
    renderer's temporal antialiasing as a sky full of movers. That is how the
    first live engagement was lost: a track seeded on a warm-up artefact 0.5 s
    into the episode and the interceptor flew at it.

    A second and a quarter of deliberate blindness costs 15 m of the intruder's
    run, which the detection range can afford and a false lock cannot.
    """

    mad_alpha: float = 0.05
    """Rate the per-pixel noise estimate adapts at, **on background pixels only**.

    The gating is the whole thing, and leaving it out produced the most
    instructive failure in this file: a detector that got *worse* the closer the
    target came. Measured, detection rate by range with the noise estimate
    updated everywhere:

    | range | 20-40 m | 40-60 | 60-80 | 100-120 |
    |---|---|---|---|---|
    | ungated | 0.00 | 0.06 | 0.18 | 0.23 |
    | gated | see the module's measurements |

    A target that lingers on the same pixels -- which is exactly what an
    inbound one on a constant bearing does, and increasingly so as it closes --
    feeds its own contrast into those pixels' noise estimate. Sigma climbs to
    the size of the target's own signal, the threshold climbs with it, and the
    drone quietly threshold-suppresses itself. It is invisible in every unit
    test because it needs a hundred frames of dwell to develop.
    """
    sigma_floor: float = 1.6
    min_diff: float = 7.0
    k_sigma_bg: float = 5.0

    flicker_alpha: float = 0.01
    flicker_rate0: float = 0.25
    flicker_boost: float = 3.0
    """Suppress pixels that are *chronically* different from the background.

    Two things need this and they arrive from opposite directions. A renderer's
    temporal antialiasing jitters high-contrast edges every frame -- measured
    here at around ten blobs a frame across the ring with an empty sky. And the
    background model is seeded from the first frame, which already contains the
    intruder, so the target leaves a permanent imprint at its starting position
    that the model then reports forever as a difference.

    Both are pixels that fire *most of the time*, and a transiting drone is not:
    it crosses a given pixel for a handful of frames out of a hundred. So the
    threshold is raised where the foreground rate is high, which turns the ghost
    off -- and then, no longer being foreground, it is absorbed by the ordinary
    background update and stops existing at all. Lifted from
    ``dronedet.motion.MotionDetector``, which needed it for wind-blown foliage.

    The time constant is deliberately **long** -- a hundred frames -- and that
    is what makes it safe. A drone creeping across the sky at a fifth of a pixel
    a frame still leaves any given pixel's neighbourhood within a dozen frames,
    so its contribution decays; a stuck pixel accumulates for ever. Shortening
    it to twenty-five frames does suppress the clutter faster, and it suppresses
    a slow target with it -- measured, and the reason this is not simply turned
    up. What buys the time instead is
    :attr:`~pursuit.episode.ScenarioConfig.calibrate_frames`.
    """
    """Threshold in per-pixel robust sigmas.

    Per *pixel*, which is what makes one threshold work over sky and city at
    once. A patch of calm sky has a sigma at the floor, so a couple of grey
    levels of change is significant there; a foliage edge that flickers every
    frame has a large sigma and is ignored. A single global threshold has to be
    set for the noisiest thing in the image and is then blind everywhere else --
    and the target is always in the quiet half.
    """

    scale: float = 1.0
    k_static: float = 5.0
    k_moving: float = 7.0
    min_area: int = 3
    max_area: int = 4000
    max_span_px: float = 120.0
    border_px: int = 8
    blur: int = 0
    open_ksize: int = 0
    dilate_ksize: int = 3
    max_blobs: int = 64
    """How many contacts a tick may report, across all four cameras.

    Twelve, inherited from a detector whose job was to hand a *few* proposals to
    an expensive verifier, and it silently threw the target away. The list is
    ranked by contrast, and a 3 px drone at 150 m has less of it than any of the
    dozen high-contrast edges a rendered city offers -- so the cap removed
    exactly the one contact that mattered, on every frame, and the truth-
    referenced diagnostic showed the drone absent from the candidate pool for
    the entire engagement while the pool itself sat full.

    Sixty-four costs nothing downstream: discrimination moved into
    :class:`_CandidatePool`, which holds 256 and rejects on *behaviour over
    seconds* rather than on being outshone in a single frame. A cap on a
    high-recall stage is a filter pretending to be a budget.
    """
    threads: int = 4
    """Cameras processed concurrently. Four independent images, four
    independent background models, and every heavy step releases the GIL."""
    confirm_px: float = 6.0
    confirm_frames: int = 3
    """A blob is *confirmed* if one appeared within ``confirm_px`` in any of
    the previous ``confirm_frames - 1`` frames. Six pixels is thirty times the
    target's own per-frame motion at long range and a thousandth of the frame,
    so a real contact always corroborates and noise essentially never does."""
    refine_lk: bool = False
    static_speed_mps: float = 0.75
    score_base: float = 0.30
    score_per_hit: float = 0.10
    score_max: float = 0.90


class RingMotionDetector:
    """Ego-compensated frame differencing on every camera at once.

    The ego compensation is **analytic, not estimated**, and that is the one
    idea in this class. ``dronedet``'s moving-camera detectors solve for the
    camera's motion with grid LK and RANSAC and fall back to *identity* when the
    solve fails -- which is a good design for footage whose camera motion nobody
    recorded. Here the camera is bolted to an aircraft whose heading the
    controller knows to machine precision, and the failure case is precisely a
    frame of empty sky, where LK has nothing to lock onto and identity is the
    worst possible answer: it declares the entire image to be moving the instant
    the aircraft turns, and buries a 3 px drone under it.

    So the rotation comes from the yaw history (see
    :func:`~pursuit.geometry.yaw_homography`), exactly, for free, and works on a
    blank sky. What it cannot remove is *translation* parallax, which is why the
    threshold is raised while the aircraft is flying -- and why the appearance
    detector, not this one, carries the closing phase.

    Three frames rather than two: a blob has to differ from *both* of the last
    two registered pasts at its current location. Differencing against one past
    reports the place a target *was* just as loudly as where it is, and a seeker
    handed both ends of that dumbbell splits the difference.
    """

    name = "ring-motion"

    def __init__(self, ring: Ring, cfg: Optional[MotionConfig] = None) -> None:
        self.ring = ring
        self.cfg = cfg or MotionConfig()
        self._pool_exec = None
        if self.cfg.threads > 1 and len(ring.cameras) > 1:
            from concurrent.futures import ThreadPoolExecutor
            # One worker per camera. The per-camera work is a colour convert, a
            # dozen full-frame numpy ops and a connected-components pass -- all
            # of which release the GIL -- over four completely independent
            # images and four independent background models. It is the one part
            # of this loop that parallelises for free, and it is CPU work, so it
            # takes nothing from the GPU that the renderer and the detector are
            # competing over.
            self._pool_exec = ThreadPoolExecutor(
                max_workers=min(int(self.cfg.threads), len(ring.cameras)),
                thread_name_prefix="ringmotion")
        self.reset()

    def reset(self) -> None:
        # (grey, yaw) for t-1 and t-2, per camera.
        self._hist: Dict[str, List[Tuple[np.ndarray, float]]] = {}
        self._scaled: Dict[str, Intrinsics] = {}
        # Static background model per camera: (bg, mad, frames_held).
        self._bg: Dict[str, List] = {}
        self._bg_yaw: Optional[float] = None
        self._blob_hist: List[List[Tuple[str, float, float]]] = []

    def _invalidate_background(self) -> None:
        """Throw the background away. Called the moment the aircraft moves.

        A background model is a statement about a camera that is not moving.
        Keeping one across a translation is not a stale model, it is a wrong
        one, and it fails in the worst possible way: every edge in the city
        exceeds threshold at once, the blob list saturates, and the real target
        is discarded by the ``max_blobs`` cap.
        """
        self._bg.clear()
        self._bg_yaw = None

    def _scaled_intr(self, cam: RingCamera) -> Intrinsics:
        i = self._scaled.get(cam.name)
        if i is None:
            s = self.cfg.scale
            i = Intrinsics(int(cam.intr.width * s), int(cam.intr.height * s),
                           cam.intr.fx * s, cam.intr.fy * s,
                           cam.intr.cx * s, cam.intr.cy * s)
            self._scaled[cam.name] = i
        return i

    def detect(self, frames: Dict[str, np.ndarray], ego_yaw: float,
               ego_speed: float = 0.0) -> List[RingDetection]:
        import cv2

        cfg = self.cfg
        moved = (float(ego_speed) > cfg.static_speed_mps
                 or (self._bg_yaw is not None
                     and abs(wrap_pi(float(ego_yaw) - self._bg_yaw)) > 2e-3))
        static = cfg.background and not moved
        if moved:
            self._invalidate_background()
        self._bg_yaw = float(ego_yaw)
        k = cfg.k_static if not moved else cfg.k_moving
        out: List[RingDetection] = []

        def one(cam):
            return self._one_camera(cv2, cam, frames.get(cam.name), static,
                                    float(ego_yaw), k)

        if self._pool_exec is not None:
            for got in self._pool_exec.map(one, self.ring.cameras):
                out.extend(got)
        else:
            for cam in self.ring.cameras:
                out.extend(one(cam))

        out = self._confirm(out)
        out.sort(key=lambda d: (not d.confirmed, -d.score))
        return out[:cfg.max_blobs]

    def _one_camera(self, cv2, cam: RingCamera, frame, static: bool,
                    ego_yaw: float, k: float) -> List[RingDetection]:
        """One camera's contacts. Runs on its own thread and its own state.

        Nothing here is shared between cameras -- separate images, separate
        background models, separate frame histories -- which is what makes the
        thread pool in :meth:`detect` safe as well as worthwhile.
        """
        cfg = self.cfg
        out: List[RingDetection] = []
        if frame is None:
            return out
        g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if cfg.scale != 1.0:
            g = cv2.resize(g, None, fx=cfg.scale, fy=cfg.scale,
                           interpolation=cv2.INTER_AREA)
        if cfg.blur >= 3:
            g = cv2.GaussianBlur(g, (cfg.blur, cfg.blur), 0)

        if static:
            out.extend(self._background_blobs(cv2, cam, g))
        else:
            hist = self._hist.setdefault(cam.name, [])
            if len(hist) >= 2:
                out.extend(self._blobs(cv2, cam, g, hist, ego_yaw, k))
        # The differencing history is kept up to date either way, so the first
        # frame after the aircraft launches has two pasts to work with instead
        # of two blind frames at the worst possible moment.
        hist = self._hist.setdefault(cam.name, [])
        hist.append((g, ego_yaw))
        if len(hist) > 2:
            hist.pop(0)
        return out

    def _confirm(self, dets: List[RingDetection]) -> List[RingDetection]:
        """Tag each blob with whether something was recently in the same place."""
        cfg = self.cfg
        hist = self._blob_hist
        for d in dets:
            d.confirmed = any(
                any(abs(d.u - u) <= cfg.confirm_px
                    and abs(d.v - v) <= cfg.confirm_px
                    for (cam, u, v) in frame if cam == d.camera)
                for frame in hist)
        hist.append([(d.camera, d.u, d.v) for d in dets])
        while len(hist) > max(1, cfg.confirm_frames - 1):
            hist.pop(0)
        return dets

    def _background_blobs(self, cv2, cam: RingCamera,
                          g: np.ndarray) -> List[RingDetection]:
        """Everything that is not part of the scene, from a camera holding still."""
        cfg = self.cfg
        state = self._bg.get(cam.name)
        if state is None:
            gf = g.astype(np.float32)
            # Every intermediate is preallocated once per camera and reused.
            # Four 1.4 Mpx cameras a tick is 18 full-frame array operations, and
            # letting numpy allocate a fresh 5.8 MB temporary for each of them
            # costs more than the arithmetic does -- measured at 120 ms a tick
            # before this, which is most of a 20 Hz budget spent on malloc.
            state = {
                "bg": gf.copy(),
                "mad": np.full_like(gf, cfg.sigma_floor),
                "flicker": np.zeros_like(gf),
                "age": np.zeros(gf.shape, np.uint16),
                "dev": np.empty_like(gf), "adev": np.empty_like(gf),
                "thr": np.empty_like(gf), "tmp": np.empty_like(gf),
                "n": 1,
            }
            self._bg[cam.name] = state
            return []

        bg, mad, flicker = state["bg"], state["mad"], state["flicker"]
        age = state["age"]
        dev, adev, thr, tmp = (state["dev"], state["adev"], state["thr"],
                               state["tmp"])
        n = state["n"]

        np.subtract(g, bg, out=dev, dtype=np.float32)
        np.abs(dev, out=adev)

        # thr = max(k * sigma * flicker_boost, min_diff), all in place.
        np.multiply(mad, 1.4826 * cfg.k_sigma_bg, out=thr)
        np.maximum(thr, cfg.sigma_floor * cfg.k_sigma_bg, out=thr)
        # Chronic-pixel accounting uses the *unboosted* bar. Feeding a
        # suppressed pixel's own suppression back into the statistic that
        # suppressed it is a limit cycle: the flicker estimate decays because
        # the pixel stopped firing, the bar drops, it fires again, and a
        # renderer artefact reappears every few seconds forever.
        raw = adev > thr
        np.multiply(flicker, cfg.flicker_boost / max(1e-6, cfg.flicker_rate0),
                    out=tmp)
        np.minimum(tmp, 3.0 * cfg.flicker_boost, out=tmp)
        np.add(tmp, 1.0, out=tmp)
        np.multiply(thr, tmp, out=thr)
        np.maximum(thr, cfg.min_diff, out=thr)
        fg = adev > thr

        # Sigma-delta update, slowed where the pixel is currently foreground so
        # the target is not learned; the noise estimate is *not* updated there
        # at all, so the target cannot inflate its own threshold.
        np.clip(dev, -cfg.bg_step, cfg.bg_step, out=dev)
        np.multiply(dev, np.where(fg, cfg.bg_step_fg / max(1e-6, cfg.bg_step),
                                  1.0), out=dev)
        bg += dev
        # Both running estimates use a rate of max(alpha, 1/n) while young, so
        # they are a plain mean of everything seen so far until the exponential
        # rate takes over. Starting straight in at alpha means the noise floor
        # needs 1/alpha frames to reach the truth and the flicker map needs 100,
        # during which every threshold in the image is wrong in the direction
        # that invents targets.
        a_mad = max(cfg.mad_alpha, 1.0 / max(1, n))
        a_flk = max(cfg.flicker_alpha, 1.0 / max(1, n))
        np.subtract(adev, mad, out=tmp)
        np.multiply(tmp, a_mad, out=tmp)
        if n >= cfg.bg_warmup:
            # Slowed on foreground pixels, not frozen, and the difference is
            # the difference between two opposite failures.
            #
            # Updating everywhere lets a target that lingers feed its own
            # contrast into its own threshold until it disappears -- measured,
            # 0.23 detection at 110 m falling to 0.00 at 30 m.
            #
            # Freezing entirely is the mirror image, and it is what a *static*
            # camera on a *static* scene was reporting fifty contacts a frame
            # for: a genuinely noisy pixel crosses the threshold once, becomes
            # foreground, stops updating its own noise estimate, and is
            # therefore foreground forever. The estimate that would have
            # explained it away is the one thing switched off.
            #
            # No single rate serves both: a tenth-rate update still suppresses
            # a target that dwells for three seconds, and a hundredth takes a
            # minute to expire a stuck pixel. The estimate therefore stays
            # frozen -- protecting the target -- and *chronic* pixels are dealt
            # with by the flicker map instead, which is the statistic that can
            # actually tell the two apart: a stuck pixel is foreground almost
            # always, a target crossing one is foreground for a handful of
            # frames out of a hundred.
            # Frozen only for pixels that have been foreground *briefly*.
            #
            # Neither extreme works and both were measured. Updating everywhere
            # lets a target that lingers feed its own contrast into its own
            # threshold until it vanishes -- 0.23 detection at 110 m falling to
            # 0.00 at 30 m. Freezing everywhere is the mirror image: a genuinely
            # noisy pixel crosses the threshold once, becomes foreground, stops
            # updating the estimate that would have explained it away, and is
            # foreground for ever -- a stationary camera on a static scene
            # returning fifty contacts a frame.
            #
            # What separates them is *duration*. A drone crossing a pixel at a
            # fifth of a pixel a frame is on it for a dozen frames; a stuck
            # pixel is on for ever. So the freeze expires: past `fg_freeze_max`
            # the pixel is treated as background again and its noise estimate
            # catches up, which turns it off.
            np.multiply(tmp, ~((age < cfg.fg_freeze_max) & fg), out=tmp)
        mad += tmp
        # The neighbourhood of a chronic mover counts too -- an aliasing edge
        # wanders a pixel or two and would otherwise dodge its own suppression.
        np.add(age, 1, out=age, where=fg)
        age[~fg] = 0
        spread = cv2.dilate(raw.view(np.uint8), np.ones((5, 5), np.uint8))
        np.subtract(spread, flicker, out=tmp, dtype=np.float32)
        np.multiply(tmp, a_flk, out=tmp)
        flicker += tmp
        state["n"] = n + 1
        if n < cfg.bg_warmup:
            return []

        return self._extract(cv2, cam, fg.view(np.uint8).copy(), adev, thr)

    def _blobs(self, cv2, cam: RingCamera, g: np.ndarray,
               hist: List[Tuple[np.ndarray, float]], now_yaw: float, k: float
               ) -> List[RingDetection]:
        cfg = self.cfg
        h, w = g.shape
        intr_s = self._scaled_intr(cam)
        diffs = []
        valid = np.ones((h, w), np.uint8)
        for past, past_yaw in hist:
            if past.shape != g.shape:
                continue
            # The history holds the heading each past frame was captured at, so
            # the rotation to undo is measured against *now* -- the frame being
            # differenced -- and not against the previous entry. Getting that
            # one index wrong leaves exactly one tick of uncompensated yaw,
            # which at a saturated 2.5 rad/s is 90 px of false motion across the
            # whole image.
            hom = yaw_homography(intr_s, wrap_pi(now_yaw - past_yaw))
            if cfg.refine_lk:
                hom = self._refine(cv2, past, g, hom) or hom
            warp = cv2.warpPerspective(past, hom, (w, h), flags=cv2.INTER_LINEAR,
                                       borderValue=0)
            vm = cv2.warpPerspective(np.ones((h, w), np.uint8), hom, (w, h))
            diffs.append(cv2.absdiff(g, warp).astype(np.float32))
            valid &= (vm > 0).astype(np.uint8)
        if len(diffs) < 2:
            return []

        motion = np.minimum(diffs[0], diffs[1])
        b = max(1, int(cfg.border_px))
        valid[:b, :] = 0
        valid[-b:, :] = 0
        valid[:, :b] = 0
        valid[:, -b:] = 0
        motion *= valid
        vals = motion[valid > 0]
        if vals.size < 100:
            return []

        # Median + MAD rather than mean + std. The background here is bimodal --
        # featureless sky next to a city whose every roofline moves under
        # translation -- and a mean is pulled up by the city until the sky half
        # is effectively switched off. The median is not.
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) * 1.4826
        thr = max(6.0, med + k * max(mad, 0.6))

        mask = (motion > thr).astype(np.uint8)
        return self._extract(cv2, cam, mask, motion,
                             np.full_like(motion, max(1.0, thr)))

    def _extract(self, cv2, cam: RingCamera, mask: np.ndarray,
                 response: np.ndarray, scale: np.ndarray) -> List[RingDetection]:
        """Mask to detections, shared by both paths.

        ``response`` is the per-pixel magnitude and ``scale`` the per-pixel
        threshold it had to beat, so a confidence means the same thing whether
        it came from a background model's per-pixel sigma or from a frame
        difference's global threshold. It has to: the tracker downstream cannot
        tell the two apart and should not have to.
        """
        cfg = self.cfg
        # An opening is an erosion followed by a dilation, so a 3x3 ellipse
        # *deletes* anything smaller than 3x3 -- which at 100 m is the target.
        # Zero means skip it, and zero is the default: the area filter below
        # rejects single-pixel noise without also rejecting the drone.
        if cfg.open_ksize >= 3:
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (cfg.open_ksize, cfg.open_ksize)))
        b = max(1, int(cfg.border_px))
        mask[:b, :] = 0
        mask[-b:, :] = 0
        mask[:, :b] = 0
        mask[:, -b:] = 0
        if cfg.dilate_ksize >= 3:
            mask = cv2.dilate(mask, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (cfg.dilate_ksize, cfg.dilate_ksize)))
        n, lab, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)

        inv = 1.0 / cfg.scale
        found: List[RingDetection] = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (cfg.min_area <= area <= cfg.max_area):
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            span_full = max(bw, bh) * inv
            if span_full > cfg.max_span_px:
                continue
            sel = lab == i
            peak = float(response[sel].max())
            ref = float(scale[sel].min())
            # Contrast above the threshold that admitted it, squashed into
            # [0, 1]. A per-detection number, deliberately modest: what makes a
            # motion contact trustworthy is that it keeps happening in the same
            # place, and that judgement belongs to the tracker, not here.
            snr = (peak - ref) / max(1.0, ref)
            score = min(0.85, 0.25 + 0.30 * math.log2(1.0 + max(0.0, snr)))
            u = (x + bw * 0.5) * inv
            v = (y + bh * 0.5) * inv
            los = cam.to_body(u, v)
            found.append(RingDetection(
                los=los, score=score, camera=cam.name, u=u, v=v,
                span_px=None, kind="motion",
                span_rad=span_full / (cam.intr.fx
                                      * offaxis_scale(cam.intr, u, v))))
        return found

    def _refine(self, cv2, past: np.ndarray, now: np.ndarray, seed):
        """Residual homography on top of the analytic rotation, or None.

        Seeded rather than solved from scratch: the rotation is already known,
        so LK only has to explain the leftover translation parallax, which keeps
        it inside its convergence radius even at a saturated yaw rate.
        """
        h, w = now.shape
        warped = cv2.warpPerspective(past, seed, (w, h))
        xs = np.linspace(w * 0.06, w * 0.94, 24)
        ys = np.linspace(h * 0.06, h * 0.94, 14)
        pts = np.array([[x, y] for y in ys for x in xs],
                       np.float32).reshape(-1, 1, 2)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            warped, now, pts, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        good = st.ravel() == 1
        if good.sum() < 16:
            return None
        hh, inl = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 2.5)
        if hh is None or inl is None or int(inl.sum()) < 12:
            return None
        return hh @ seed


# --------------------------------------------------------------------- tracker

@dataclass
class RingTrackerConfig:
    """Tuning for :class:`RingTracker`.

    Every gate is an **angle**, which is the change from
    :class:`~pursuit.perception.TrackerConfig` and the reason this is a separate
    class rather than a parameter. Pixels are not comparable across a ring: the
    same drone moving the same way subtends different pixel distances in
    different parts of a 96 degree lens, and no pixel distance at all between
    two cameras. Angles are comparable everywhere, which is what makes a seam
    crossing an ordinary frame.

    The numbers are the pixel gates of the single-camera tracker converted
    through its own ``fx`` and then re-checked against the physics: a 9 m/s
    crossing target at 40 m moves 11 mrad per frame at 20 Hz, so a 60 mrad base
    gate is five frames of slack and a 25 mrad corroboration gate is two.

    Attributes:
        gate_base_rad: Association radius for a fresh track.
        gate_per_miss_rad: Extra radius per consecutive miss.
        gate_max_rad: Ceiling on that growth.
        gate_span_scale: Extra radius proportional to the target's angular size,
            so a target filling the frame at impact -- whose centre legitimately
            moves degrees per frame -- is still associated.
        init_hits / init_window_s / init_gate_rad: Corroborated seeding, exactly
            as the single-camera tracker does it and for the same measured
            reason. The ego-yaw correction that made it work there is *free*
            here: these bearings are already body-frame, so a heading change
            between the pair is subtracted rather than approximated by
            ``fx * dpsi``.
        init_score: Confidence needed to start a track.
        init_max_span_rad: Largest angular size that may *start* one. A first
            sight is not a 20 degree object.
        max_coast_frames: Frames of prediction with no detection before the
            track is declared lost.
        reseed_after_misses: Misses after which the gate is abandoned and the
            best corroborated candidate is taken wherever it is.
        confirm_hits: Detections before the track is reported at all.
        span_tau: EMA time constant on the angular span (s).
        motion_score_base / motion_score_per_hit / motion_score_max: What a
            motion-only track is worth. A single blob is worth very little --
            0.30, below anything that would steer the aircraft -- and it climbs
            with each *consecutive gated* hit, reaching 0.70 on the fourth.

            That is the whole confidence model for the long-range half of this
            sensor, and it is an argument about statistics rather than about
            appearance. A motion detector cannot tell a drone from a bird from a
            thresholding artefact in one frame. What it can do is notice that
            something produced a detection in very nearly the same direction
            four frames running, which scattered clutter does not: at the
            measured false-blob rate this detector runs at against a static
            city, the chance of four consecutive hits inside a 60 mrad gate is
            negligible, while a real inbound drone produces them every time.
            The same reasoning as ``TrackerConfig.init_gate_px``, extended from
            a pair to a run.
    """

    gate_base_rad: float = 0.025
    """25 mrad, not the 60 the single camera's 60 px gate converts to.

    A gate is a statement about how far the target can have moved, and a ring
    watching a cluttered city is also making a statement about how much clutter
    it is prepared to swallow. An inbound intruder moves a quarter of a
    milliradian per frame; 25 mrad is a hundred times that and still tight
    enough to matter -- measured against ten to thirty fixed artefacts, dropping
    from 60 to 25 mrad took the fraction of time locked on something that was
    not the drone from 1.4 percent to **zero**, and cost the drone nothing.

    Close range is covered by ``gate_span_scale`` rather than by a loose base:
    at ten metres the target is 23 mrad across and legitimately moves degrees a
    frame, and the span term opens the gate exactly then and not before.
    """
    gate_per_miss_rad: float = 0.0125
    gate_max_rad: float = 0.125
    gate_span_scale: float = 1.2
    init_hits: int = 2
    init_window_s: float = 0.25
    """Twice the single camera's, and affordable for a reason that is specific
    to this sensor. There, the window had to be about one frame because the
    corroboration gate was in pixels and a slewing camera swept 65 px between
    frames -- so a longer window meant a wider gate meant more clutter pairing
    up. Here the gate is an angle and the aircraft's rotation is subtracted
    exactly, and during acquisition it is not rotating at all. The gate can stay
    tight while the window grows, which is what turns a detector firing on a
    third of frames at 120 m into a lock instead of a scatter of glimpses."""
    init_gate_rad: float = 0.025
    init_score: float = 0.20
    init_max_span_rad: float = 0.30
    max_coast_frames: int = 12
    reseed_after_misses: int = 4
    confirm_hits: int = 2
    span_tau: float = 0.15
    motion_score_base: float = 0.30
    motion_score_per_hit: float = 0.10
    motion_score_max: float = 0.90

    motion_min_travel_rad: float = 0.010
    motion_travel_window_s: float = 3.0
    """Three seconds, not one, and the number comes from the target.

    An inbound intruder is nearly *radial*, so its bearing rate is small exactly
    where it needs to be found: measured on the city approach, 4.7 mrad/s at
    160 m rising to 12 mrad/s at 100 m. A one-second window therefore fails the
    drone at the range the mission is decided at, and the fix is a longer window
    rather than a lower bar -- a fixed object accumulates nothing over three
    seconds either, while the drone accumulates 14 mrad. The cost is that the
    first promotion cannot come before three seconds of tracking, which puts it
    at ~124 m against the 77-106 m the mission needs.
    """
    motion_unproven_score: float = 0.45

    cand_gate_rad: float = 0.015
    cand_ttl_s: float = 0.7
    cand_min_hits: int = 12
    cand_ambiguity: int = 1
    """Contacts above which a track has to prove it is flying before it is
    granted. One means: with a single thing in the sky, believe it."""
    cand_min_rate_rad_s: float = 0.0015
    """Angular rate a contact must show before it counts as flying, rad/s.

    Set from both sides of the measurement rather than derived from the travel
    bar, which put it at 3.3 mrad/s and was **above the target**. An inbound
    intruder is nearly radial: 2.0 mrad/s at 150 m, 4.7 at 120, 12 at 100. A bar
    of 3.3 therefore refused to believe the drone until it was inside 130 m and
    the fit had another 1.8 s of history to gather -- acquisition landed at 84 m
    with 2.4 s left on the clock.

    The floor underneath is the noise: a *fixed* object's fitted rate is its
    jitter divided by the window, measured on the rig at 0.0-0.6 mrad/s. 1.5
    mrad/s sits between the two with a factor of 2.5 either way, which is what a
    threshold is supposed to look like.
    """
    cand_min_span_s: float = 1.8
    cand_max_residual_rad: float = 0.003
    cand_max: int = 256
    """Room for everything the ring reports for as long as the TTL keeps it.

    Rivermark returns a dozen motion blobs a frame and the TTL holds a contact
    for fourteen, so a pool of 48 was over-subscribed four times over and spent
    the engagement thrashing. A candidate is a few kilobytes; being generous
    here costs nothing and being mean costs the target.
    """
    """The candidate pool: how contacts are watched before one is chosen.

    ``cand_gate_rad`` is deliberately tighter than the tracking gate. A
    candidate is not being *followed*, it is being *characterised*, and a loose
    gate lets two neighbouring artefacts merge into one contact that appears to
    wander -- which is the one way a fixed object can fake having flown.
    ``cand_max_residual_rad`` catches whatever gets through, and it is the test
    that actually works: fit a constant angular rate and ask how well the
    bearings sit on it. Three milliradians is twice the measurement jitter and a
    third of the gate, so a drone passes and a candidate flipping between two
    artefacts -- which produces a perfectly healthy average *rate* -- does not.
    """
    """A motion track must **move across the sky** before it may steer.

    Persistence alone is not enough, and the first live run said so in the
    clearest possible way: the interceptor seeded on a renderer artefact within
    half a second of the episode starting, watched it produce a gated detection
    on every single frame -- which is what a chronically flickering edge does --
    climbed its confidence to 0.9, promoted it, and flew 145 m due north while
    the intruder came in from the east and hit the building. Nothing in the
    confidence model could tell the two apart, because a fixed artefact is
    *more* persistent than a real target, not less.

    What separates them is physics, and from an interceptor holding station it
    separates them with no ambiguity at all. A fixed object at a fixed camera
    has a bearing rate of exactly zero. Anything flying has one: an intruder at
    150 m with four metres a second of cross-range moves 27 mrad in a second,
    and 10 mrad -- nine pixels at this focal length -- is a bar no artefact can
    clear and no aircraft can fail.

    Until it clears that bar a motion track is capped below the promotion floor:
    it exists, it is tracked, it is refined -- it simply does not get to point
    the aircraft. Which is the same separation ``GuidanceConfig.promote_hits``
    makes for appearance detections, applied to the evidence a motion detector
    actually produces.
    """


class _Candidate:
    """One thing the ring keeps half an eye on, and how it has behaved.

    Cheap on purpose -- a smoothed direction, a hit count, and how far it has
    travelled -- because there are a dozen of these and only one of them is
    going to be worth a Kalman filter.
    """

    __slots__ = ("los", "first", "t_first", "t_last", "hits", "path", "score",
                 "samples", "_fit")

    def __init__(self, los, t: float, score: float):
        self.los = los
        self.first = los
        self.t_first = t
        self.t_last = t
        self.hits = 1
        self.path = 0.0
        self.score = score
        az, el = body_bearing(los)
        self.samples: List[Tuple[float, float, float]] = [(t, az, el)]
        self._fit: Optional[Tuple[int, float, float]] = None

    def update(self, los, t: float, score: float, smooth: float = 0.45) -> None:
        blend = tuple(self.los[i] * (1.0 - smooth) + los[i] * smooth
                      for i in range(3))
        n = math.sqrt(sum(c * c for c in blend)) or 1.0
        blend = tuple(c / n for c in blend)
        self.path += angle_between(self.los, blend)
        self.los = blend
        self.t_last = t
        self.hits += 1
        self.score = max(self.score, score)
        az, el = body_bearing(los)
        # The *raw* bearing, not the smoothed one: the fit below is a test of
        # how well the measurements lie on a line, and pre-smoothing them would
        # be testing the smoother.
        self.samples.append((t, az, el))
        if len(self.samples) > 80:
            self.samples.pop(0)
        self._fit = None

    @property
    def net(self) -> float:
        return angle_between(self.first, self.los)

    @property
    def rate(self) -> float:
        """Fitted angular rate, rad/s -- 0 until there is enough to fit."""
        return self.fit()[0]

    def fit(self) -> Tuple[float, float]:
        """``(angular rate, RMS residual)`` of a constant-rate fit, rad and rad/s.

        The discriminator that survives everything cheaper. A drone flies a
        *line* in bearing: over a couple of seconds its azimuth and elevation
        are linear in time to within the jitter. A fixed artefact has zero rate.
        And -- this is the case net displacement could not catch -- a candidate
        accidentally stitched out of two neighbouring artefacts flips between
        them, which produces a healthy-looking rate and a residual half their
        separation wide.

        Two independent least-squares fits, one per angle, sharing the residual.
        Azimuths are taken relative to the first sample so a candidate sitting
        on the +/-180 seam does not fit a line through a discontinuity.
        """
        if self._fit is not None and self._fit[0] == len(self.samples):
            return self._fit[1], self._fit[2]
        s = self.samples
        if len(s) < 4:
            self._fit = (len(s), 0.0, 0.0)
            return 0.0, 0.0
        t0, az0, el0 = s[0]
        n = float(len(s))
        ts = [p[0] - t0 for p in s]
        mt = sum(ts) / n
        stt = sum((x - mt) ** 2 for x in ts)
        if stt <= 1e-9:
            return 0.0, 0.0
        rate2 = 0.0
        resid2 = 0.0
        for idx, base in ((1, az0), (2, el0)):
            ys = [wrap_pi(p[idx] - base) for p in s]
            my = sum(ys) / n
            b = sum((ts[i] - mt) * (ys[i] - my) for i in range(len(s))) / stt
            a = my - b * mt
            rate2 += b * b
            resid2 += sum((ys[i] - (a + b * ts[i])) ** 2 for i in range(len(s)))
        out = (math.sqrt(rate2), math.sqrt(resid2 / n))
        self._fit = (len(s), out[0], out[1])
        return out


class _CandidatePool:
    """Watch everything cheaply; commit to the one that is flying.

    The stage that was missing, and the live run said so unmistakably: against
    a Rivermark sky the ring's motion detector returns about *fifty* blobs a
    frame, nearly all of them renderer artefacts, and exactly one drone. (An
    earlier figure of ten was itself an artefact of a since-removed twelve-blob
    cap: the stage was reporting up to its ceiling and the ceiling was what got
    measured.) A single-target tracker seeded on the first corroborated pair
    therefore locks onto clutter almost every time and, being single-target, it
    never looks at anything else again.

    No confidence threshold fixes that, because the artefacts are *more*
    persistent and often more contrasty than a 3 px aircraft. What separates
    them is behaviour over seconds: an artefact sits still and a drone flies.
    So every contact gets a cheap running record, and the expensive machinery
    downstream is only ever handed one that has demonstrably moved --

    * **net travel** past ``min_travel_rad``: a fixed object accumulates none;
    * **straightness**, net over path: a real track walks in one direction while
      a candidate stitched out of two neighbouring artefacts wanders;
    * enough hits that neither number is an accident.

    This is the same argument the detection half of this repository makes with
    its track classifier -- do not announce a drone from appearance, announce it
    from a track that behaved like one -- and it arrives here for the same
    reason.
    """

    def __init__(self, cfg: "RingTrackerConfig") -> None:
        self.cfg = cfg
        self.items: List[_Candidate] = []

    def clear(self) -> None:
        self.items = []

    def update(self, dets: Sequence[RingDetection], t: float,
               ego_yaw: float) -> None:
        """Record this frame's contacts, in **world** directions.

        World, not body, and the distinction is the same one the rest of this
        project is organised around: in body axes an aircraft yawing at 2.5
        rad/s makes every fixed object in the town sweep past at 2.5 rad/s --
        smoothly, in a straight line, fitting a constant rate beautifully. The
        test that is supposed to prove a contact is flying would certify the
        whole skyline.
        """
        cfg = self.cfg
        self.items = [c for c in self.items if t - c.t_last <= cfg.cand_ttl_s]
        for d in dets:
            w = to_world(d.los, ego_yaw)
            best, best_a = None, 0.0
            for c in self.items:
                # The gate widens by however fast this contact is already known
                # to be moving. A fixed gate is a statement that nothing moves
                # faster than it, and a crossing target at 40 m moves 11 mrad a
                # frame -- so a fixed 15 mrad gate spawns a fresh candidate
                # every frame, none of which ever accumulates enough history to
                # prove anything, and the tracker locks onto nothing at all.
                gate = cfg.cand_gate_rad + c.rate * max(0.0, t - c.t_last)
                a = angle_between(c.los, w)
                if a <= gate and (best is None or a < best_a):
                    best, best_a = c, a
            if best is None:
                if not d.confirmed:
                    # Only a corroborated contact may open a new candidate.
                    continue
                if len(self.items) >= cfg.cand_max:
                    # Mature contacts only. A brand-new candidate has fewer
                    # than four samples and therefore a fitted rate of exactly
                    # zero, so ranking by rate alone evicts *the newest thing in
                    # the pool* -- which is the drone, on the frame it first
                    # appears, every frame, for the whole engagement.
                    # Full. Drop the most *static* contact rather than the new
                    # one, because the pool is a shortlist of things that might
                    # be flying and something that has not moved is the least
                    # likely candidate in it.
                    #
                    # Measured, and it is the whole ballgame: Rivermark returns
                    # a dozen motion blobs a frame, the pool filled with static
                    # clutter within four frames of the episode starting, and
                    # every subsequent contact -- including the drone, every
                    # frame, for ten seconds -- was silently discarded at this
                    # line. The telemetry showed 48 candidates, 117 hits on the
                    # top one, and a fitted rate of 0.1 mrad/s.
                    mature = [c for c in self.items if len(c.samples) >= 4]
                    worst = (min(mature, key=lambda c: c.rate) if mature
                             else min(self.items, key=lambda c: c.t_first))
                    self.items.remove(worst)
                self.items.append(_Candidate(w, t, d.score))
            else:
                best.update(w, t, d.score)

    def proven(self, t: float) -> Optional[_Candidate]:
        """The best contact that has earned a real track, or None."""
        cfg = self.cfg
        best, best_key = None, 0.0
        need_rate = cfg.cand_min_rate_rad_s
        for c in self.items:
            if c.hits < cfg.cand_min_hits:
                continue
            if t - c.t_first < cfg.cand_min_span_s:
                continue
            rate, resid = c.fit()
            if rate < need_rate or resid > cfg.cand_max_residual_rad:
                continue
            # One interceptor can chase one drone, so this is a *choice*, not a
            # filter, and it goes to the contact there is most reason to
            # believe in. Three things, multiplied because each can veto:
            #
            #   how cleanly it flies   rate / residual -- a real track is fast
            #                          relative to its own scatter, and the
            #                          fastest thing in a cluttered frame is
            #                          usually the worst-behaved one;
            #   how long it has been   hits, saturating -- something seen forty
            #   there                  times is not twice as real as one seen
            #                          twenty times, but it is more real than
            #                          one seen twelve;
            #   what it looks like     the best detector score it has drawn.
            key = ((rate / max(resid, 1e-4))
                   * min(2.0, c.hits / max(1, cfg.cand_min_hits))
                   * max(0.25, c.score))
            if key > best_key:
                best, best_key = c, key
        return best


class _BearingKF:
    """Constant-rate ``[az, el, az_rate, el_rate]`` in body axes, radians.

    Azimuth is kept **unwrapped**. A target crossing behind the aircraft passes
    through the +/-180 seam, and a filter whose state jumps by 2 pi there
    reports a line-of-sight rate of 125 rad/s for one frame -- which is exactly
    the sort of number the guidance law's rate clamp exists to survive but
    should never have to see.
    """

    def __init__(self, az: float, el: float, q: float = 0.35, r: float = 0.004):
        self.x = np.array([az, el, 0.0, 0.0], dtype=float)
        self.P = np.diag([1e-3, 1e-3, 0.5, 0.5])
        self.q = float(q)
        self.r = float(r)

    def predict(self, dt: float, dpsi: float = 0.0) -> None:
        """Advance, then subtract the aircraft's own rotation.

        ``dpsi`` is not noise and is not estimated: the airframe yawed by a
        known amount, so every body-frame bearing moved by exactly ``-dpsi``.
        Applying it here is what lets the gate stay tight while the aircraft is
        manoeuvring -- the single-camera tracker had to approximate the same
        correction as ``fx * dpsi`` pixels because it had nowhere else to put
        it.
        """
        f = np.eye(4)
        f[0, 2] = f[1, 3] = dt
        self.x = f @ self.x
        self.x[0] -= float(dpsi)
        d2, d3, d4 = dt * dt, dt ** 3, dt ** 4
        qm = np.array([[d4 / 4, 0, d3 / 2, 0],
                       [0, d4 / 4, 0, d3 / 2],
                       [d3 / 2, 0, d2, 0],
                       [0, d3 / 2, 0, d2]]) * self.q
        self.P = f @ self.P @ f.T + qm

    def update(self, az: float, el: float) -> None:
        h = np.zeros((2, 4))
        h[0, 0] = h[1, 1] = 1.0
        rr = np.eye(2) * self.r * self.r
        y = np.array([wrap_pi(az - self.x[0]), el - self.x[1]])
        s = h @ self.P @ h.T + rr
        kg = self.P @ h.T @ np.linalg.inv(s)
        self.x = self.x + kg @ y
        self.P = (np.eye(4) - kg @ h) @ self.P

    @property
    def los(self) -> Tuple[float, float, float]:
        return bearing_to_body(float(self.x[0]), float(self.x[1]))


class RingTracker:
    """One target, tracked as a direction rather than as a pixel.

    Same three jobs as :class:`~pursuit.perception.SingleTargetTracker` --
    associate, bridge dropped frames, decide what to report -- and the same two
    rules that were each learned from a bug:

    * a detection's own bearing is reported directly and the filter's is used
      only while coasting, because de-rotating a filtered blend of past frames
      manufactures line-of-sight rate out of the aircraft's own yaw;
    * a track may only be *started* by two detections that corroborate each
      other in place and time, because one is evidence of a number.

    What is new is that "the same place" now means the same place *in the
    world*, across a set of cameras, which is the thing a ring needs and the
    thing pixel coordinates cannot express.
    """

    def __init__(self, ring: Ring, cfg: Optional[RingTrackerConfig] = None) -> None:
        self.ring = ring
        self.cfg = cfg or RingTrackerConfig()
        self.reset()

    def reset(self) -> None:
        self.kf: Optional[_BearingKF] = None
        self.span_px: Optional[float] = None
        self.span_rad: Optional[float] = None
        self.hits = 0
        self.misses = 0
        self.age = 0
        self.score = 0.0
        self.motion_run = 0
        self.last: Optional[RingDetection] = None
        self.t_last: Optional[float] = None
        self.yaw_last: Optional[float] = None
        self._pending: List[Tuple[float, RingDetection, float]] = []
        # (t, world-frame direction) at each detection, for the travel test.
        # World rather than body, so the aircraft's own rotation is not read as
        # the target moving -- the same distinction the whole guidance file is
        # organised around.
        self._travel: List[Tuple[float, Tuple[float, float, float]]] = []
        self.travel_rad = 0.0
        self.flight_proven = False
        self._pool = _CandidatePool(self.cfg)

    @property
    def alive(self) -> bool:
        return self.kf is not None and self.misses <= self.cfg.max_coast_frames

    @property
    def confirmed(self) -> bool:
        return self.kf is not None and self.hits >= self.cfg.confirm_hits

    def predicted_los(self) -> Optional[Tuple[float, float, float]]:
        return None if self.kf is None else self.kf.los

    def _gate(self) -> float:
        c = self.cfg
        g = c.gate_base_rad + c.gate_per_miss_rad * self.misses
        g += c.gate_span_scale * (self.span_rad or 0.0)
        return min(g, c.gate_max_rad + c.gate_span_scale * (self.span_rad or 0.0))

    def step(self, dets: Sequence[RingDetection], t: float,
             ego_yaw: float, ego_static: bool = True) -> Optional[RingDetection]:
        cfg = self.cfg
        dt = 0.05 if self.t_last is None else max(1e-3, float(t) - self.t_last)
        dpsi = 0.0 if self.yaw_last is None else wrap_pi(float(ego_yaw) - self.yaw_last)
        self.t_last, self.yaw_last = float(t), float(ego_yaw)

        if self.kf is not None:
            self.kf.predict(dt, dpsi)

        # Everything is watched, every tick, whether or not there is a track.
        self._static = bool(ego_static)
        if ego_static:
            self._pool.update(dets, self.t_last, float(ego_yaw))
        else:
            self._pool.clear()

        # An unproven lock does not get to keep the tracker. If something else
        # in the sky has meanwhile proved it is flying, take that instead --
        # otherwise the first blob of the episode occupies the single-target
        # tracker for the whole engagement and the drone is never looked at.
        if (self.kf is not None and ego_static and not self.flight_proven
                and self.travel_rad < cfg.motion_min_travel_rad):
            best = self._pool.proven(self.t_last)
            if best is not None and angle_between(
                    to_world(self.kf.los, float(ego_yaw)),
                    best.los) > cfg.cand_gate_rad * 2.0:
                keep = self._pool          # the pool is what found the winner
                self.reset()
                self._pool = keep
                self.t_last, self.yaw_last = float(t), float(ego_yaw)
                dpsi = 0.0

        pick = self._associate(dets, self.t_last, float(ego_yaw))

        if pick is not None:
            az, el = body_bearing(pick.los)
            reseed = self.kf is not None and self.misses >= cfg.reseed_after_misses
            if self.kf is None or reseed:
                self.kf = _BearingKF(az, el)
                self.span_rad = pick.span_rad
                # Not `pick.span_px` unconditionally: seeding is the one path
                # where a motion blob could hand its dilated width straight to
                # the range filter, and a range seeded 2x short latches the
                # terminal phase from 80 m out.
                self.span_px = (pick.span_px if pick.kind == "appearance"
                                else None)
                self.motion_run = 0
                # A re-seed is a different object. Its predecessor's travel is
                # not evidence about it -- unless this very reseed came from a
                # proven candidate, which `_seed` will have just recorded.
                self._travel = []
                if not self.flight_proven:
                    self.travel_rad = 0.0
            else:
                self.kf.update(az, el)
                a = 1.0 - math.exp(-dt / max(1e-6, cfg.span_tau))
                if pick.span_rad > 0.0:
                    base = self.span_rad if self.span_rad is not None else pick.span_rad
                    self.span_rad = base + a * (pick.span_rad - base)
            # Only an appearance box may move the pixel span, and therefore the
            # range. A motion blob has been through a dilation kernel; believing
            # its width would put the target nearer than it is and latch the
            # terminal phase from 90 m away.
            if pick.kind == "appearance" and pick.span_px:
                base = self.span_px if self.span_px is not None else pick.span_px
                a = 1.0 - math.exp(-dt / max(1e-6, cfg.span_tau))
                self.span_px = base + a * (pick.span_px - base)
            self.hits += 1
            self.misses = 0
            self.last = pick
            if ego_static:
                self._note_travel(az, el, float(ego_yaw), self.t_last, cfg)
            if pick.kind == "appearance":
                self.motion_run = 0
                self.score = pick.score
            else:
                self.motion_run += 1
                self.score = min(cfg.motion_score_max,
                                 cfg.motion_score_base
                                 + cfg.motion_score_per_hit * self.motion_run)
            if (pick.kind != "appearance" and not self.flight_proven
                    and self.travel_rad < cfg.motion_min_travel_rad):
                # A motion contact that has not moved has produced no evidence
                # at all -- a blob is a blob. An *appearance* detection is
                # exempt, and the exemption is load-bearing rather than
                # generous: a target fleeing straight away has a bearing rate of
                # zero by construction, which is the easiest case there is for
                # proportional navigation and an impossible one for a test built
                # on bearing motion. Capping it too locked the aircraft on the
                # ground for every tail chase in the `full` suite.
                #
                # What protects the city case is not this cap but
                # _CandidatePool: while holding station among several contacts,
                # nothing gets a track until it has been watched flying, so a
                # confident box on a rooftop never reaches this line.
                self.score = min(self.score, cfg.motion_unproven_score)
        else:
            self.misses += 1
            if self.kf is not None and not self.alive:
                self.reset()
                return None
        self.age += 1
        return pick

    def _note_travel(self, az: float, el: float, ego_yaw: float, t: float,
                     cfg: RingTrackerConfig) -> None:
        """How far this contact has moved across the *world*, recently.

        Recorded only on detections -- a coasted prediction is generated by the
        constant-velocity model itself, so letting it contribute would let a
        track certify its own motion.
        """
        world = bearing_to_body(az + ego_yaw, el)
        self._travel.append((t, world))
        cut = t - cfg.motion_travel_window_s
        while len(self._travel) > 2 and self._travel[0][0] < cut:
            self._travel.pop(0)
        if len(self._travel) < 2:
            self.travel_rad = 0.0
            return
        # **Net** displacement over the window, not path length. Summing the
        # per-frame steps would defeat the whole test: a renderer artefact
        # jitters about a pixel a frame, which is 1.1 mrad, and twenty frames of
        # that adds up to 21 mrad of "travel" without the thing having gone
        # anywhere. Endpoint separation cancels jitter and keeps real motion.
        self.travel_rad = angle_between(self._travel[0][1], self._travel[-1][1])
        if self.travel_rad >= cfg.motion_min_travel_rad:
            self.flight_proven = True

    def _associate(self, dets: Sequence[RingDetection], t: float,
                   ego_yaw: float) -> Optional[RingDetection]:
        if not dets:
            return None
        if self.kf is None:
            return self._seed(dets, t, ego_yaw)
        pred = self.kf.los
        gate = self._gate()
        best, best_d = None, float("inf")
        for d in dets:
            ang = angle_between(d.los, pred)
            if ang <= gate and ang < best_d:
                best, best_d = d, ang
        if best is None and self.misses >= self.cfg.reseed_after_misses:
            return self._seed(dets, t, ego_yaw)
        return best

    def _seed(self, dets: Sequence[RingDetection], t: float,
              ego_yaw: float) -> Optional[RingDetection]:
        cfg = self.cfg
        if not getattr(self, "_static", True) and len(dets) > cfg.cand_ambiguity + 1:
            # Flying, in clutter. Bearing cannot separate a fixed object from a
            # moving one from a moving observer -- everything sweeps -- so there
            # is no honest way to choose here and the corroborated seed would
            # simply take whichever rooftop is nearest the last prediction.
            # Refusing means the lock stays lost, which stops the aircraft
            # (ACQUIRE holds), which restores the stationary observer the
            # candidate pool needs. The loop is self-correcting; guessing is not.
            return None
        if getattr(self, "_static", True) and len(self._pool.items) > cfg.cand_ambiguity:
            # Holding station with more than one thing in the sky: only a
            # contact that has proved it flies gets a real track. See
            # _CandidatePool -- against ten fixed artefacts, corroboration alone
            # picks the wrong one nine times in ten.
            #
            # With *one* contact there is nothing to disambiguate and the proof
            # is pure delay, so the ordinary corroborated seed runs instead.
            # That is not a loophole: a single unambiguous contact is what an
            # empty sky looks like, and making it wait three seconds to prove
            # it is flying costs acquisitions and buys nothing.
            best = self._pool.proven(t)
            if best is None:
                return None
            near, near_a = None, cfg.cand_gate_rad * 2.0
            for d in dets:
                a = angle_between(to_world(d.los, ego_yaw), best.los)
                if a <= near_a:
                    near, near_a = d, a
            if near is not None:
                # The candidate did the proving; the track inherits the verdict.
                # Recording it as a *flag* rather than as a travel figure is not
                # bookkeeping: `_note_travel` recomputes `travel_rad` from the
                # new track's own (empty) history on the very next line, which
                # silently threw the proof away and capped a perfectly good lock
                # at 0.45 for the rest of the engagement.
                self.travel_rad = best.net
                self.flight_proven = True
            return near
        strong = [d for d in dets
                  if d.score >= cfg.init_score and d.span_rad <= cfg.init_max_span_rad]
        self._pending = [(ts, d, y) for ts, d, y in self._pending
                         if t - ts <= cfg.init_window_s]
        if not strong:
            return None
        best = max(strong, key=lambda d: d.score)
        if cfg.init_hits <= 1:
            return best
        gate = cfg.init_gate_rad + cfg.gate_span_scale * best.span_rad
        for _ts, prev, prev_yaw in self._pending:
            # Rotate the earlier bearing into the heading held now. Exact, not
            # a small-angle pixel approximation: a body-frame azimuth moves by
            # precisely minus the heading change.
            paz, pel = body_bearing(prev.los)
            shifted = bearing_to_body(paz - wrap_pi(ego_yaw - prev_yaw), pel)
            if angle_between(shifted, best.los) <= gate:
                self._pending = []
                return best
        self._pending.append((t, best, ego_yaw))
        return None

    def estimate(self) -> TargetEstimate:
        """The current belief, whether or not this frame had a detection."""
        if self.kf is None or not self.confirmed:
            return TargetEstimate(valid=False, source="none")
        measured = self.misses == 0 and self.last is not None
        if measured:
            los = self.last.los
            u, v, cam_name = self.last.u, self.last.v, self.last.camera
        else:
            los = self.kf.los
            cam = self.ring.owner(los)
            uv = cam.to_pixel(los) if cam else None
            u, v = (uv if uv else (None, None))
            cam_name = cam.name if cam else ""
        az, el = body_bearing(los)
        return TargetEstimate(
            valid=True, u=u, v=v, span_px=self.span_px, az=az, el=el,
            score=self.score if measured else max(0.0, self.score * 0.5),
            bbox=None, source="detector" if measured else "coast",
            age_frames=self.age, los_body=los, camera=cam_name,
            kind=(self.last.kind if measured and self.last else "coast"))


# ------------------------------------------------------------------ front end

class RingPerception:
    """Four frames in, one :class:`~pursuit.perception.TargetEstimate` out.

    Interface-compatible with :class:`~pursuit.perception.Perception` on
    purpose: the same ``reset`` / ``step`` / ``stage_report`` triple, and the
    same dataclass on the way out. Guidance is not told which sensor it has, and
    the whole attribution argument this project rests on -- oracle against real,
    same scenarios -- keeps working with a ring in place of a camera.

    **Scheduling is the design.** Running the appearance model on four
    1600x640 frames is 520 ms a tick, which is not a control loop. So:

    * the motion detector runs on **every** camera, every tick, because it is
      cheap (a warp and a difference at half resolution) and because it is the
      only thing that finds a 3 px contact at 100 m;
    * the appearance model runs on **one or two**, aimed by whatever is already
      known -- the camera holding the track, plus its neighbour while the target
      is in a seam, or failing that the camera with the strongest motion blob;
    * with neither a track nor a blob it walks one camera per tick, so a target
      that is somehow not moving in the image is still found eventually.

    That is the ``mc-hybrid`` pattern from the detection half of this
    repository -- motion proposes, appearance verifies -- and it lands here for
    the same reason: the cheap wide-recall stage decides where the expensive
    precise one looks.
    """

    def __init__(self, ring: Ring, detector=None,
                 motion: Optional[RingMotionDetector] = None,
                 tracker_cfg: Optional[RingTrackerConfig] = None,
                 min_score: float = 0.0,
                 max_appearance_cams: int = 2,
                 crop_px: int = 640,
                 oracle=None) -> None:
        self.ring = ring
        self.intr = ring.cameras[0].intr
        self.detector = detector
        self.motion = motion
        self.tracker = RingTracker(ring, tracker_cfg)
        self.min_score = float(min_score)
        self.max_appearance_cams = int(max_appearance_cams)
        self.crop_px = int(crop_px)
        self.oracle = oracle
        self.last_boxes: List[Box] = []
        self.last_by_camera: Dict[str, List[Box]] = {}
        self.last_dets: List[RingDetection] = []
        self.pool_state: Dict[str, float] = {}
        self.timings = {"detect_ms": 0.0, "motion_ms": 0.0, "track_ms": 0.0}
        self.samples: Dict[str, List[float]] = {"detect_ms": [], "motion_ms": [],
                                                "track_ms": []}
        self.n = 0
        self._rr = 0

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        self.tracker.reset()
        if self.motion is not None:
            self.motion.reset()
        for d in (self.detector, self.oracle):
            fn = getattr(d, "reset", None)
            if callable(fn):
                fn()
        self.last_boxes = []
        self.last_by_camera = {}
        self.last_dets = []
        self.timings = {"detect_ms": 0.0, "motion_ms": 0.0, "track_ms": 0.0}
        self.samples = {"detect_ms": [], "motion_ms": [], "track_ms": []}
        self.n = 0
        self._rr = 0

    # -- one tick -----------------------------------------------------------

    def truth_report(self, truth_los, t: float) -> dict:
        """What the pool makes of the contact that *is* the drone.

        Diagnostics only, and it needs the answer in order to ask the question
        -- which is exactly why it is worth having. "Sixty candidates and none
        of them proved" has two completely different causes: the drone's contact
        is not in the pool at all, or it is in there and one of four thresholds
        is refusing it. Nothing observable from outside separates those, and
        they are fixed in opposite directions.
        """
        if truth_los is None:
            return {}
        best, best_a = None, math.radians(2.0)
        for c in self.tracker._pool.items:
            a = angle_between(c.los, truth_los)
            if a < best_a:
                best, best_a = c, a
        if best is None:
            return {"has": False}
        rate, resid = best.fit()
        cfg = self.tracker.cfg
        return {"has": True, "hits": best.hits,
                "age": round(t - best.t_first, 2),
                "rate": round(1000.0 * rate, 2),
                "resid": round(1000.0 * resid, 2),
                "why": ("hits" if best.hits < cfg.cand_min_hits else
                        "age" if t - best.t_first < cfg.cand_min_span_s else
                        "rate" if rate < cfg.cand_min_rate_rad_s else
                        "resid" if resid > cfg.cand_max_residual_rad else "ok")}

    def step(self, frames: Optional[Dict[str, np.ndarray]], idx: int, t: float,
             gt: Optional[dict] = None, ego_yaw: float = 0.0,
             ego_speed: float = 0.0) -> TargetEstimate:
        import time as _time

        frames = frames or {}
        dets: List[RingDetection] = []

        t0 = _time.perf_counter()
        if self.motion is not None and frames:
            dets.extend(self.motion.detect(frames, ego_yaw, ego_speed))
        t1 = _time.perf_counter()

        by_cam: Dict[str, List[Box]] = {}
        if self.oracle is not None:
            dets.extend(self._oracle_dets(gt, by_cam))
        elif self.detector is not None and frames:
            dets.extend(self._appearance_dets(frames, idx, dets, by_cam))
        t2 = _time.perf_counter()

        dets = [d for d in dets if d.score >= self.min_score]
        dets = fuse_detections(dets, self.ring)
        self.tracker.step(dets, t, ego_yaw,
                          ego_static=float(ego_speed) <= 0.75)
        t3 = _time.perf_counter()

        pool = self.tracker._pool.items
        top = max(pool, key=lambda c: c.hits, default=None)
        rate, resid = (top.fit() if top is not None else (0.0, 0.0))
        # A ring that reports nothing and a ring that sees nothing look
        # identical from the outside and have opposite fixes.
        self.pool_state = {
            "dets": len(dets), "cands": len(pool),
            "top_hits": (top.hits if top is not None else 0),
            "rate_mrad": round(1000.0 * rate, 2),
            "resid_mrad": round(1000.0 * resid, 2),
            "proven": bool(self.tracker.flight_proven),
        }
        self.last_dets = dets
        self.last_by_camera = by_cam
        self.last_boxes = [b for bs in by_cam.values() for b in bs]
        for key, ms in (("motion_ms", (t1 - t0) * 1000.0),
                        ("detect_ms", (t2 - t1) * 1000.0),
                        ("track_ms", (t3 - t2) * 1000.0)):
            self.timings[key] += ms
            self.samples[key].append(ms)
        self.n += 1
        return self.tracker.estimate()

    # -- detector scheduling -------------------------------------------------

    def _appearance_cams(self, motion_dets: Sequence[RingDetection]) -> List[RingCamera]:
        pred = self.tracker.predicted_los()
        if pred is not None and self.tracker.alive:
            cams = self.ring.seeing(pred)
            if cams:
                # In a seam both cameras see it; run both, so the handover is
                # covered from both sides rather than betting on the tie-break.
                return cams[:self.max_appearance_cams]
        if motion_dets:
            order, seen = [], set()
            for d in sorted(motion_dets, key=lambda x: -x.score):
                if d.camera in seen:
                    continue
                seen.add(d.camera)
                cam = self.ring.get(d.camera)
                if cam is not None:
                    order.append(cam)
                if len(order) >= self.max_appearance_cams:
                    break
            if order:
                return order
        # Nothing to aim at. Walk the ring so a target that is somehow not
        # moving in the image -- a hovering intruder, a detector that dropped
        # every blob -- is still swept eventually, at one camera per tick
        # instead of four.
        cam = self.ring.cameras[self._rr % len(self.ring.cameras)]
        self._rr += 1
        return [cam]

    def _appearance_dets(self, frames: Dict[str, np.ndarray], idx: int,
                         motion_dets: Sequence[RingDetection],
                         by_cam: Dict[str, List[Box]]) -> List[RingDetection]:
        """Run the appearance model where something has already been seen.

        A crop, not the frame, whenever there is anything to aim at -- which is
        this repository's own ``hybrid`` pattern arriving where it belongs.
        Two reasons, and the second is the bigger one:

        * a 640 px window is a twentieth of a 2048x704 frame, so the expensive
          stage stops setting the loop rate;
        * and it runs at **native scale**. A full-frame pass has to fit 2048 px
          into the network's input, which shrinks a 9 px drone to 8; the crop
          does not shrink it at all. The detector was always the thing that ran
          out of pixels first.

        With nothing to aim at it falls back to a whole frame, walking one
        camera per tick, so a target the motion stage never saw is still swept
        eventually.
        """
        out: List[RingDetection] = []
        aim = self._aim_points(motion_dets)
        for cam in self._appearance_cams(motion_dets):
            frame = frames.get(cam.name)
            if frame is None:
                continue
            for box, (ox, oy) in self._passes(cam, frame, aim.get(cam.name), idx):
                b = Box(box.x1 + ox, box.y1 + oy, box.x2 + ox, box.y2 + oy,
                        box.score, box.label)
                by_cam.setdefault(cam.name, []).append(b)
                out.append(self._to_ring(cam, b, "appearance"))
        return out

    def _aim_points(self, motion_dets: Sequence[RingDetection]) -> Dict[str, tuple]:
        """Where to point the crop on each camera: the track, else the best blob."""
        aim: Dict[str, tuple] = {}
        pred = self.tracker.predicted_los()
        if pred is not None and self.tracker.alive:
            for cam in self.ring.seeing(pred):
                uv = cam.to_pixel(pred)
                if uv is not None:
                    aim[cam.name] = uv
        for d in sorted(motion_dets, key=lambda x: -x.score):
            aim.setdefault(d.camera, (d.u, d.v))
        return aim

    def _passes(self, cam: RingCamera, frame, uv, idx: int):
        """``(boxes, offset)`` for each region of this camera worth looking at."""
        if uv is None or not self.crop_px:
            return [(b, (0.0, 0.0)) for b in self.detector.detect(frame, idx, None)]
        h, w = frame.shape[:2]
        half = int(self.crop_px) // 2
        x0 = max(0, min(w - 2 * half, int(uv[0]) - half))
        y0 = max(0, min(h - 2 * half, int(uv[1]) - half))
        crop = np.ascontiguousarray(frame[y0:y0 + 2 * half, x0:x0 + 2 * half])
        if crop.size == 0:
            return []
        # At the crop's own size, which is the entire point. Left at the
        # full-frame `imgsz` the network would *upsample* a 640 px window by
        # 2.9x -- three times the cost of the whole-frame pass it replaced, on
        # an image with no more information in it.
        keep = getattr(self.detector, "imgsz", None)
        try:
            if keep is not None:
                self.detector.imgsz = int(2 * half)
            boxes = self.detector.detect(crop, idx, None)
        finally:
            if keep is not None:
                self.detector.imgsz = keep
        return [(b, (float(x0), float(y0))) for b in boxes]

    def _oracle_dets(self, gt: Optional[dict],  # noqa: C901 - flat by design
                     by_cam: Dict[str, List[Box]]) -> List[RingDetection]:
        """A perfect detector, per camera, from the simulator's own labels.

        The oracle sees the target in *every* camera that renders it, seams
        included, which is exactly right: it is a perfect detector, not a
        perfect fusion. The duplicate it produces in an overlap is the same
        duplicate a real pair of detectors produces, so the merge is exercised
        rather than bypassed -- and a merge bug shows up in the run that was
        supposed to isolate guidance, where it is cheapest to find.
        """
        out: List[RingDetection] = []
        per = (gt or {}).get("per_camera") or {}
        if not per and gt and gt.get("bbox"):
            per = {gt.get("camera") or self.ring.names[0]: gt}
        for name, view in per.items():
            cam = self.ring.get(name)
            if cam is None:
                continue
            boxes = self.oracle.detect(None, 0, _with_geometric_box(view))
            by_cam.setdefault(name, []).extend(boxes)
            for b in boxes:
                out.append(self._to_ring(cam, b, "appearance"))
        return out

    def _to_ring(self, cam: RingCamera, b: Box, kind: str) -> RingDetection:
        span_scale = cam.intr.fx * offaxis_scale(cam.intr, b.cx, b.cy)
        return RingDetection(los=cam.to_body(b.cx, b.cy), score=float(b.score),
                             camera=cam.name, u=b.cx, v=b.cy, span_px=b.span,
                             kind=kind, span_rad=b.span / max(1e-6, span_scale))

    # -- reporting -----------------------------------------------------------

    def stage_report(self) -> dict:
        n = max(1, self.n)
        out = {k: round(v / n, 2) for k, v in self.timings.items()}
        for key, xs in self.samples.items():
            if xs:
                ordered = sorted(xs)
                i = min(len(ordered) - 1, int(0.95 * len(ordered)))
                out[key.replace("_ms", "_p95_ms")] = round(ordered[i], 2)
        per_frame = sum(out.get(k, 0.0) for k in ("detect_ms", "motion_ms",
                                                  "track_ms"))
        out["perception_ms"] = round(per_frame, 2)
        out["perception_fps"] = round(1000.0 / per_frame, 1) if per_frame > 0 else 0.0
        return out


def _with_geometric_box(view: dict) -> dict:
    """Give a camera view a box even when the renderer refused to draw one.

    An oracle is supposed to be a *perfect detector*, and this one was not: it
    reads the simulator's ``bounding_box_2d_tight`` annotator, which returns
    nothing at all for a target a few pixels across. Measured on the city
    mission, that made the oracle acquire at **7.0 s** -- when the intruder had
    already flown two thirds of its run -- and lose both engagements, while the
    real motion detector was seeing the same target from 160 m. An oracle that
    is worse than the system it is supposed to bound is not a control, it is a
    second unexplained result.

    So when the annotator is empty but the geometry says the target is in frame,
    the box is built from the analytic projection and its predicted span. That
    is what "perfect detector" was always meant to mean; the annotator was only
    ever a convenient way to spell it at ranges where it works.
    """
    if view.get("bbox") or not view.get("analytic_in_frame"):
        return view
    uv = view.get("analytic_uv")
    span = view.get("analytic_span_px")
    if not uv or not span or span <= 0.0:
        return view
    out = dict(view)
    # Same aspect the rendered boxes have -- an Iris is wide and thin, and the
    # tracker's span is the larger side either way.
    out["bbox"] = [uv[0] - span / 2.0, uv[1] - span / 8.0,
                   uv[0] + span / 2.0, uv[1] + span / 8.0]
    return out


class RingOracle:
    """A perfect detector on every camera at once, with independent noise.

    One :class:`~pursuit.perception.OracleDetector` per camera rather than one
    shared between them, and the distinction is not pedantry. The degradations
    are the point of the oracle -- dropout, box noise, latency -- and four
    cameras sharing a single dropout draw or a single latency queue is not a
    degraded ring, it is a ring wired together wrongly: the queue would deal
    each camera the frame belonging to the one before it, and a target in a
    seam would blink in both views at once instead of in one, which is exactly
    the correlated failure a ring is supposed to be immune to.

    Args:
        ring: The camera set.
        seed: Base seed; each camera is offset from it so their noise is
            independent but the whole thing is still one reproducible flight.
    """

    name = "ring-oracle"
    needs_frame = False

    def __init__(self, ring: Ring, seed: int = 11, **oracle_kw) -> None:
        from .perception import OracleDetector
        self.per = {c.name: OracleDetector(seed=seed + 977 * i, **oracle_kw)
                    for i, c in enumerate(ring.cameras)}

    def reset(self) -> None:
        for d in self.per.values():
            d.reset()

    def detect(self, frame, idx: int, gt: Optional[dict] = None) -> List[Box]:
        d = self.per.get((gt or {}).get("camera") or "")
        if d is None:
            return []
        return d.detect(frame, idx, gt)
