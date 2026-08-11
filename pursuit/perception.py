"""Frame in, one target estimate out: the chaser's eye.

Stage one of the mission (*detect and track the other drone*) lives here, and it
is deliberately separated from stage two (*close on it*) by a single small
dataclass, :class:`TargetEstimate`. Guidance never sees a frame, a box or a
confidence; it sees a bearing, a span and a flag. That boundary is what lets the
hardest question in this project -- "was that miss the perception or the
control?" -- be answered instead of argued about, because the same guidance law
can be flown against a *perfect* sensor (:class:`OracleDetector`) and against
the real one, and the difference between the two runs is the perception's
contribution, measured.

Three things happen between the frame and the estimate:

**Detection.** A YOLO on the raw frame. The repo's flagship detectors are
temporal -- they stack stabilised greyscales from t-12/t-6/t, or fuse an
ego-motion difference channel -- and that is the right design for their problem,
a 4-pixel drone against field clutter from a nearly-static camera. It is the
wrong one here and measurably so: a chaser flying an intercept is *translating
and rotating hard*, so frame-to-frame background motion is large and full of
parallax, and the motion channel fills with clutter exactly when the aircraft is
manoeuvring hardest. See :class:`YoloDetector`.

**Tracking.** One target, tracked in the image plane with a constant-velocity
Kalman filter and nearest-neighbour association inside a gate that widens as
misses accumulate. Its job is not to improve a good detection -- it is to
*bridge* the frames that have none, which on a 20 Hz control loop is most of
what stands between a lock and a lost target.

**Selection.** With several detections, the one nearest the track's prediction
wins, not the most confident one. During a pursuit the thing being chased is by
construction the thing that was in the same place last frame; taking the highest
score instead is how a lock jumps to a bird.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import (Intrinsics, bearing_from_pixel, normalized_offset,
                        wrap_pi)


# ------------------------------------------------------------------ estimates

@dataclass
class Box:
    """One detection in image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = "drone"

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def span(self) -> float:
        """The larger side. An Iris seen edge-on is wide and thin, so the width
        tracks its true 0.47 m rotor span while the height tracks its body
        thickness; the max of the two is the one that means "how far away"."""
        return max(self.w, self.h)


@dataclass
class TargetEstimate:
    """What guidance is allowed to know about the target this frame."""

    valid: bool = False
    u: Optional[float] = None
    v: Optional[float] = None
    span_px: Optional[float] = None
    az: Optional[float] = None
    el: Optional[float] = None
    score: float = 0.0
    bbox: Optional[Tuple[float, float, float, float]] = None
    source: str = "none"
    """``detector`` (measured this frame), ``coast`` (the tracker's prediction
    through a dropped frame), or ``none``."""
    age_frames: int = 0

    los_body: Optional[Tuple[float, float, float]] = None
    """Unit direction to the target in body axes, when the sensor can express
    one.

    A single forward camera cannot see past 90 degrees, so ``az``/``el`` -- the
    pinhole's own tangent-plane angles -- describe everything it could ever
    report and this stays None. A camera *ring* can put a target at 170 degrees,
    where a tangent-plane azimuth is not merely inaccurate but undefined, so it
    fills this in and guidance steers on it directly. See
    :mod:`pursuit.ring`.
    """

    camera: str = ""
    """Which camera of a ring the estimate came from. Empty for one camera."""

    kind: str = "appearance"
    """``appearance`` (a recognised silhouette, so its pixel span means a range)
    or ``motion`` (something moved there, bearing only)."""
    range_override: Optional[float] = None
    """A directly measured metric range, bypassing the monocular estimate.

    Nothing in this package sets it, deliberately -- including
    :class:`OracleDetector`, which is a perfect *detector*, not a perfect
    rangefinder. It reports a box, and the range that box implies is computed
    from its pixel span exactly as it would be for a YOLO, so an oracle run
    still exercises the real range-estimation error rather than papering over
    it. The hook exists for a platform that genuinely has depth (a stereo pair,
    a lidar), which would be a different sensor and a different experiment."""

    def offset(self, intr: Intrinsics):
        return normalized_offset(intr, self.u, self.v) if self.valid else (None, None)


# ------------------------------------------------------------------ detectors

class OracleDetector:
    """A perfect (or deliberately degraded) sensor, straight from the simulator.

    Built for two jobs. First, it decouples the halves of the mission: an
    intercept that fails with the oracle is a guidance bug, and one that works
    with the oracle and fails with YOLO is a perception bug -- and no amount of
    staring at a video distinguishes those. Second, its degradations are the
    cheapest robustness test there is: ``dropout`` makes frames vanish,
    ``noise_px`` shakes the box, ``span_bias`` makes every monocular range wrong
    by a fixed factor, and ``max_range_m`` makes the target simply not
    detectable beyond some distance -- which is what a real detector does.
    """

    name = "oracle"
    needs_frame = False

    def reset(self) -> None:
        """Clear the latency queue between episodes.

        The queue is genuine per-episode state and one ``Perception`` is shared
        across a whole scenario matrix, so without this the last frames of one
        pursuit are dealt out as the first frames of the next. Reproduced: with
        two frames of latency, a 257-pixel intercept box left over from the
        previous episode seeds the new track, the range filter takes it at face
        value and reports 2 m while the target is genuinely at 40, and the FSM
        latches TERMINAL before the target has been seen even once. The affected
        episode still *scored* an intercept, which is what makes it dangerous.
        """
        self._queue = []

    def __init__(self, dropout: float = 0.0, noise_px: float = 0.0,
                 span_noise: float = 0.0, span_bias: float = 1.0,
                 max_range_m: Optional[float] = None, seed: int = 0,
                 latency_frames: int = 0) -> None:
        self.dropout = float(dropout)
        self.noise_px = float(noise_px)
        self.span_noise = float(span_noise)
        self.span_bias = float(span_bias)
        self.max_range_m = max_range_m
        self.rng = np.random.default_rng(seed)
        self.latency_frames = int(latency_frames)
        self._queue: List[Optional[Box]] = []

    def detect(self, frame: Optional[np.ndarray], idx: int,
               gt: Optional[dict] = None) -> List[Box]:
        box = self._from_gt(gt)
        if self.latency_frames <= 0:
            return [] if box is None else [box]
        self._queue.append(box)
        if len(self._queue) <= self.latency_frames:
            return []
        out = self._queue.pop(0)
        return [] if out is None else [out]

    def _from_gt(self, gt: Optional[dict]) -> Optional[Box]:
        if not gt or not gt.get("bbox"):
            return None
        if self.max_range_m is not None and gt.get("range_m", 0.0) > self.max_range_m:
            return None
        if self.dropout > 0.0 and self.rng.random() < self.dropout:
            return None
        x1, y1, x2, y2 = (float(v) for v in gt["bbox"])
        if self.noise_px > 0.0:
            dx, dy = self.rng.normal(0.0, self.noise_px, 2)
            x1, x2 = x1 + dx, x2 + dx
            y1, y2 = y1 + dy, y2 + dy
        if self.span_bias != 1.0 or self.span_noise > 0.0:
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            k = self.span_bias * (1.0 + (self.rng.normal(0.0, self.span_noise)
                                         if self.span_noise > 0.0 else 0.0))
            k = max(0.2, k)
            x1, x2 = cx - (cx - x1) * k, cx + (x2 - cx) * k
            y1, y2 = cy - (cy - y1) * k, cy + (y2 - cy) * k
        return Box(x1, y1, x2, y2, 1.0, "drone")


class YoloDetector:
    """Ultralytics YOLO on the raw frame.

    Single-frame RGB rather than one of the repo's temporal stacks, and the
    choice is load-bearing. The temporal representation is what makes a 4-pixel
    drone findable against field clutter, and it works by cancelling a *static*
    background: stabilise, stack three moments, and anything that did not move
    goes grey. From a chaser flying an intercept nothing is static -- the camera
    translates at 14 m/s while yawing to hold the target -- so the background
    does not cancel, it smears, and the motion channel is brightest exactly
    during the hardest manoeuvre. Measured on this rig, the temporal edge model
    scores recall 0.50 at eleven thousand false positives.

    What replaces it is resolution: a P2 head at a large ``imgsz`` on the plain
    frame, trained on this domain, where the target is a crisp dark quadrotor
    silhouette rather than a smudge -- and the tracker behind it supplies the
    temporal continuity the input no longer carries.
    """

    name = "yolo"
    needs_frame = True

    def __init__(self, weights: str, imgsz: int = 1280, conf: float = 0.15,
                 device: int | str = 0, half: bool = True,
                 max_det: int = 16, input_is_rgb: bool = True) -> None:
        """Args:
            input_is_rgb: The simulator's camera hands back **RGB**, ultralytics
                expects BGR on a raw array, and the training images were written
                to disk through ``cv2.imwrite`` (BGR). Feeding RGB straight in
                swaps red and blue against everything the network was trained
                on -- which does not crash, does not look obviously wrong in a
                video, and quietly costs recall on a target whose whole signature
                is being darker than the sky behind it.
        """
        from ultralytics import YOLO

        self.model = YOLO(weights, task="detect")
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.device = device
        self.half = bool(half)
        self.max_det = int(max_det)
        self.weights = str(weights)
        self.input_is_rgb = bool(input_is_rgb)

    def detect(self, frame: np.ndarray, idx: int,
               gt: Optional[dict] = None) -> List[Box]:
        if self.input_is_rgb:
            frame = np.ascontiguousarray(frame[:, :, ::-1])
        r = self.model(frame, imgsz=self.imgsz, conf=self.conf, device=self.device,
                       half=self.half, max_det=self.max_det, verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append(Box(x1, y1, x2, y2, float(b.conf[0]),
                           r.names[int(b.cls[0])]))
        return out


# ------------------------------------------------------------------- tracking

class _Kalman2D:
    """Constant-velocity ``[cx, cy, vx, vy]`` in pixels, measuring position."""

    def __init__(self, cx: float, cy: float, q: float = 60.0, r: float = 3.0):
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.P = np.diag([9.0, 9.0, 400.0, 400.0])
        self.q = float(q)
        self.r = float(r)

    def predict(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        self.x = F @ self.x
        # Continuous white-noise-acceleration Q: the process noise has to scale
        # with dt, or a dropped frame is treated as being as certain as a fresh
        # one and the gate never opens when it is most needed.
        q = self.q
        d2, d3, d4 = dt * dt, dt ** 3, dt ** 4
        Q = np.array([[d4 / 4, 0, d3 / 2, 0],
                      [0, d4 / 4, 0, d3 / 2],
                      [d3 / 2, 0, d2, 0],
                      [0, d3 / 2, 0, d2]]) * q
        self.P = F @ self.P @ F.T + Q
        return self.x[:2].copy()

    def update(self, z) -> None:
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1.0
        R = np.eye(2) * self.r * self.r
        y = np.asarray(z, dtype=float) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


@dataclass
class TrackerConfig:
    """Tuning for :class:`SingleTargetTracker`.

    Attributes:
        gate_base_px: Association radius with a fresh track (px).
        gate_per_miss_px: Extra radius per consecutive miss -- the uncertainty
            after a dropout is genuinely larger, and a fixed gate is what makes a
            track that blinked unrecoverable.
        gate_max_px: Ceiling on that growth, so a long dropout does not end with
            a gate that would swallow the whole frame.
        gate_span_scale: Extra radius proportional to the target's pixel span.
            At the end of an intercept the target is hundreds of pixels across
            and its centre legitimately moves tens of pixels a frame; a gate
            sized for a 20 px target rejects every one of those.
        init_score: Minimum detector confidence to *start* a track, as opposed
            to maintaining one. Two thresholds rather than one, and the
            asymmetry is the point: starting a lock is a decision that commits
            the aircraft to flying at something, while continuing one is
            supported by everything already seen at that place. A single low
            threshold buys recall on the target and buys a false lock on a cloud
            with the same coin; a high bar to acquire and a low bar to keep is
            how a seeker gets both.
        init_hits: Confident detections required to start a track, and they must
            agree with each other in place and time (``init_window_s``,
            ``init_gate_px``). One is not enough. Measured in the closed loop:
            with a single-detection seed the chaser locked onto scene clutter and
            flew at it for 724 frames -- its belief a median 311 pixels from the
            drone, its range estimate collapsed to 0.1 m on an oversized spurious
            box -- and the run ended having never closed at all.

            The asymmetry that makes this work is that clutter false positives
            are *sparse and incoherent*: at this operating point they arrive at
            0.15 per frame, scattered. A real target produces repeated detections
            in the same place. Requiring agreement costs a real acquisition
            almost nothing and costs a false one almost everything.
        init_window_s: How long a confident detection stays eligible to pair up.
            Short, because the pair is meant to be *the same object seen twice*,
            not two things that happened during the same second.
        init_gate_px: How close the pair must be, on top of the target's own
            span. The window is deliberately about *one frame*, and that is what
            makes the whole scheme work.

            Pairing risk from clutter scales as (gate area) x (pairs the window
            admits); the real-target motion that must fit inside the gate scales
            with the window too. Widening the window therefore loses on both
            sides at once. Pairing only across *adjacent* frames gives the
            tightest gate a real target still satisfies: the search holds the
            camera still while it looks, so during a look the only image motion
            is the target's own -- about 10 px per frame for a 9 m/s crossing
            target at 40 m, comfortably inside 25 px.

            Because that collapses the clutter risk (two scattered false
            positives would have to land within 25 px on *consecutive* frames),
            the score bar can come back down to 0.20, which roughly doubles the
            per-frame chance of catching a real 12-pixel target. Measured
            progression against this detector's own clutter, over 15 s searches:
            55 px / 9 frames captured the seeker half the time; 30 px / 4 frames
            under a tenth; 25 px / 2 frames is negligible -- and it acquires
            faster, because the bar is lower.
        init_max_span_px: Largest box that may *start* a track. An acquisition is
            by definition the first sight of something, and the first sight of a
            drone is not a box filling a third of the frame -- to be that large
            it would have to be a few metres away, which cannot happen without
            having been seen further out first. The measured false lock came with
            an oversized spurious box that drove the monocular range straight to
            its 0.1 m floor and latched the terminal phase; this rejects that
            class of nonsense at the only moment it is cheap to reject. It does
            not constrain a track already running, which legitimately grows to
            fill the frame at impact.
        max_coast_frames: Frames the tracker will report a prediction with no
            detection behind it before declaring the target lost.
        reseed_after_misses: After this many consecutive frames with nothing in
            the gate, the best available detection is taken regardless of where
            it is. Without this the gate is a trap: once the prediction has drifted
            far enough from the truth, every real detection falls outside it, and
            the track coasts confidently into nothing while the target sits in
            plain view on the other side of the frame. Measured on the first
            closed-loop run, that failure alone cost half the detections.
        confirm_hits: Detections needed before the track is reported at all.
        span_tau: EMA time constant on the pixel span (s). The span drives the
            *range*, and a range that jumps drives a speed schedule that jumps,
            so it is smoothed harder than the position is.
    """

    gate_base_px: float = 60.0
    gate_per_miss_px: float = 25.0
    gate_max_px: float = 260.0
    gate_span_scale: float = 1.2
    init_score: float = 0.20
    """Detector confidence required to *start* a track.

    Deliberately permissive, and left that way after measuring the alternative.
    Raising it to 0.55 to keep urban clutter out does keep clutter out, and it
    also stops a genuine target at the far edge of the envelope from ever
    seeding -- a drone at 80 m scores below 0.55 far too often. Tried on the
    Rivermark ingress suite it converted a clean 0.38 m intercept into a run
    that never acquired anything.

    The confidence test that works is not per-detection but per-*lock*, and it
    lives in :class:`~pursuit.guidance.PursuitGuidance` as the score probation:
    seed cheaply, then require the track to prove itself with a couple of
    confident detections within its first second or die. That keeps 91 percent
    of true locks while removing 92 percent of false ones, which no single
    threshold on this axis can do.
    """
    init_hits: int = 2
    init_window_s: float = 0.12
    init_gate_px: float = 25.0
    init_max_span_px: float = 160.0
    max_coast_frames: int = 12
    reseed_after_misses: int = 4
    confirm_hits: int = 2
    span_tau: float = 0.15


class SingleTargetTracker:
    """One drone, tracked in the image plane, coasting through dropped frames.

    Deliberately single-target. The repo's :class:`dronedet.track.Tracker` is a
    multi-target Hungarian tracker built to survive a scene full of candidate
    blips and to be scored offline for identity switches; this loop has exactly
    one thing worth following and 50 ms to decide what to do about it, and the
    question it has to answer is not "which of these is which" but "is this the
    thing I was already chasing".
    """

    def __init__(self, cfg: Optional[TrackerConfig] = None,
                 fx: float = 921.8) -> None:
        self.cfg = cfg or TrackerConfig()
        # Only fx is needed: the aircraft is yaw-only, so ego motion moves the
        # image horizontally and the vertical channel is left alone.
        self.fx = float(fx)
        self.reset()

    def reset(self) -> None:
        self.kf: Optional[_Kalman2D] = None
        self.span: Optional[float] = None
        self._pending: List[Tuple[float, Box]] = []
        self.hits = 0
        self.misses = 0
        self.age = 0
        self.score = 0.0
        self.last_box: Optional[Box] = None
        self.t_last: Optional[float] = None

    @property
    def alive(self) -> bool:
        return self.kf is not None and self.misses <= self.cfg.max_coast_frames

    @property
    def confirmed(self) -> bool:
        return self.kf is not None and self.hits >= self.cfg.confirm_hits

    def _gate(self) -> float:
        c = self.cfg
        g = c.gate_base_px + c.gate_per_miss_px * self.misses
        if self.span:
            g += c.gate_span_scale * self.span
        return min(g, c.gate_max_px + c.gate_span_scale * (self.span or 0.0))

    def step(self, boxes: Sequence[Box], t: float,
             ego_yaw: Optional[float] = None) -> Optional[Box]:
        """Advance one frame; return the box now believed to be the target."""
        dt = 0.05 if self.t_last is None else max(1e-3, float(t) - self.t_last)
        self.t_last = float(t)

        if self.kf is not None:
            self.kf.predict(dt)

        pick = self._associate(boxes, self.t_last, ego_yaw)

        if pick is not None:
            reseed = self.kf is not None and self.misses >= self.cfg.reseed_after_misses
            if self.kf is None or reseed:
                self.kf = _Kalman2D(pick.cx, pick.cy)
                self.span = pick.span
            else:
                self.kf.update((pick.cx, pick.cy))
                a = 1.0 - math.exp(-dt / max(1e-6, self.cfg.span_tau))
                self.span = (self.span or pick.span) + a * (pick.span - (self.span or pick.span))
            self.hits += 1
            self.misses = 0
            self.score = pick.score
            self.last_box = pick
        else:
            self.misses += 1
            # Only a track that *existed* can die. With `kf is None` there is no
            # track, `alive` is False by definition, and resetting here wipes the
            # pending-seed buffer on every frame -- which makes corroborated
            # acquisition impossible, since no candidate ever survives to be
            # corroborated by the next one.
            if self.kf is not None and not self.alive:
                self.reset()
                return None
        self.age += 1
        return pick

    def _associate(self, boxes: Sequence[Box], t: float,
                   ego_yaw: Optional[float] = None) -> Optional[Box]:
        if not boxes:
            return None
        if self.kf is None:
            return self._seed_candidate(boxes, t, ego_yaw)
        px, py = self.kf.pos
        gate = self._gate()
        best, best_d = None, float("inf")
        for b in boxes:
            d = math.hypot(b.cx - px, b.cy - py)
            if d <= gate and d < best_d:
                best, best_d = b, d
        if best is None and self.misses >= self.cfg.reseed_after_misses:
            # The gate has stopped being a filter and started being a blindfold:
            # nothing has matched for several frames, so the prediction is the
            # thing that is wrong. Re-seed -- but through the same corroboration
            # the first lock needed, since moving a lock somewhere new is the
            # same commitment as making one.
            return self._seed_candidate(boxes, t, ego_yaw)
        return best

    def _seed_candidate(self, boxes: Sequence[Box], t: float,
                        ego_yaw: Optional[float] = None) -> Optional[Box]:
        """A detection may only start a track if a recent one corroborates it.

        With no track there is no prediction to gate against, so confidence is
        the only evidence -- and confidence alone is what let a piece of scene
        clutter capture the seeker. Two confident detections close together in
        time and place is evidence of a *thing*; one is evidence of a number.

        "Close together in place" has to mean *in the world*, not in the image,
        and that distinction was quietly fatal. The corroboration gate is about
        40 px wide, while a camera slewing at the search rate of 1.4 rad/s
        sweeps the image by ``fx * 1.4 * 0.05`` = 65 px between consecutive
        frames -- so a perfectly good pair of detections on a stationary drone
        landed outside the gate purely because the aircraft had turned, and **no
        track could be initiated at all while the camera was moving**. Since the
        search spends most of its time slewing, and since the chaser is also
        turning hard during a pursuit, that removed most of the opportunities to
        acquire anything.

        The aircraft is yaw-only and its camera is bolted on, so the correction
        is one term: a heading change of ``dpsi`` moves a fixed world point
        ``fx * dpsi`` across the image. The pending detection is reprojected
        through the yaw actually turned since it was taken, and only then
        compared. ``ego_yaw`` defaults to None, which restores the old raw-pixel
        behaviour exactly, so nothing that does not supply a heading changes.
        """
        cfg = self.cfg
        strong = [b for b in boxes
                  if b.score >= cfg.init_score and b.span <= cfg.init_max_span_px]
        self._pending = [(ts, b, y) for ts, b, y in self._pending
                         if t - ts <= cfg.init_window_s]
        if not strong:
            return None
        best = max(strong, key=lambda b: b.score)
        if cfg.init_hits <= 1:
            return best
        gate = cfg.init_gate_px + cfg.gate_span_scale * best.span
        for _ts, prev, prev_yaw in self._pending:
            du = 0.0
            if ego_yaw is not None and prev_yaw is not None:
                du = self.fx * wrap_pi(ego_yaw - prev_yaw)
            if math.hypot(prev.cx + du - best.cx, prev.cy - best.cy) <= gate:
                self._pending = []
                return best
        self._pending.append((t, best, ego_yaw))
        return None

    def estimate(self, intr: Intrinsics) -> TargetEstimate:
        """The current belief, whether or not this frame had a detection.

        When there *is* a detection, its box centre is reported directly and the
        filtered position is not used. That looks like throwing away a perfectly
        good filter, and it is the single most important line in this class.

        The bearing gets differentiated downstream, in the world frame, which
        means it is de-rotated by the chaser's current heading. A Kalman filter
        buys smoothness by reporting a blend of the last few frames -- so what
        gets de-rotated is a bearing from a moment when the aircraft was pointing
        somewhere else. While yawing at 2 rad/s to hold a target, that mismatch
        alone manufactures a world-frame line-of-sight rate of about 2 rad/s out
        of a geometry whose true rate is zero, the guidance law steers on it, the
        steering demands more yaw, and the loop diverges. Measured on the tail-
        chase scenario: apparent LOS rate 2.08 rad/s, intercept never closed.

        Bearing does not need the filter anyway -- a box centre is already
        accurate to a pixel or two, and the rate filter downstream does the
        smoothing where it can be done without a rotating frame underneath it.
        The filter earns its place on the frames that have no detection at all,
        which is exactly when it is used.
        """
        if self.kf is None or not self.confirmed:
            return TargetEstimate(valid=False, source="none")
        if self.misses == 0 and self.last_box is not None:
            u, v = self.last_box.cx, self.last_box.cy
        else:
            u, v = self.kf.pos
        az, el = bearing_from_pixel(intr, u, v)
        measured = self.misses == 0
        return TargetEstimate(
            valid=True, u=u, v=v, span_px=self.span, az=az, el=el,
            score=self.score if measured else max(0.0, self.score * 0.5),
            bbox=None if self.last_box is None else
            (self.last_box.x1, self.last_box.y1, self.last_box.x2, self.last_box.y2),
            source="detector" if measured else "coast",
            age_frames=self.age)


# ------------------------------------------------------------------ front end

class Perception:
    """Detector + tracker, wired into one call the control loop can make.

    Args:
        detector: Anything with ``detect(frame, idx, gt) -> list[Box]``.
        intr: Camera the bearings come out in.
        tracker_cfg: Tuning for the single-target tracker.
        min_score: Detections below this are dropped before association. Kept
            separate from the detector's own ``conf`` so the *operating point*
            can be swept without reloading the network.
    """

    def __init__(self, detector, intr: Intrinsics,
                 tracker_cfg: Optional[TrackerConfig] = None,
                 min_score: float = 0.0) -> None:
        self.detector = detector
        self.intr = intr
        self.tracker = SingleTargetTracker(tracker_cfg, fx=intr.fx)
        self.min_score = float(min_score)
        self.last_boxes: List[Box] = []
        self.timings = {"detect_ms": 0.0, "track_ms": 0.0}
        self.samples: dict = {"detect_ms": [], "track_ms": []}
        self.n = 0

    def reset(self) -> None:
        """Clear per-episode state, timings included.

        The counters have to go too: one :class:`Perception` is reused across a
        whole scenario matrix, so leaving them running makes every episode's
        reported stage timing an average over that episode *and every episode
        before it* -- which hides exactly the thing the number is for, a
        detector that slows down as the target fills the frame.
        """
        self.tracker.reset()
        self.last_boxes = []
        self.timings = {"detect_ms": 0.0, "track_ms": 0.0}
        self.samples = {"detect_ms": [], "track_ms": []}
        self.n = 0
        # The detector can hold per-episode state too (a latency queue, a frame
        # buffer). Anything that has state gets to say so by defining reset().
        reset = getattr(self.detector, "reset", None)
        if callable(reset):
            reset()

    def step(self, frame: Optional[np.ndarray], idx: int, t: float,
             gt: Optional[dict] = None,
             ego_yaw: Optional[float] = None,
             ego_speed: float = 0.0) -> TargetEstimate:
        """One frame in, one :class:`TargetEstimate` out.

        ``ego_yaw`` is the chaser's heading at the instant this frame was
        captured. It is used only by the track-*initiation* gate, to tell a pair
        of detections that moved because the aircraft turned from a pair that
        moved because they are different things. Omitting it is safe and
        restores the previous behaviour; supplying it is what lets a track start
        while the camera is slewing, which is most of the time the search
        spends looking.

        ``ego_speed`` is ignored here and accepted so that this and
        :meth:`pursuit.ring.RingPerception.step` are interchangeable at every
        call site. The ring's frame differencing genuinely needs to know whether
        the aircraft is holding station; a single appearance detector does not,
        and a caller should not have to know which it is talking to.
        """
        t0 = time.perf_counter()
        # A detector that reads pixels cannot be handed None; one that derives
        # its boxes from ground truth never needs them. The detector declares
        # which it is rather than the loop guessing from its type.
        if frame is None and getattr(self.detector, "needs_frame", True):
            boxes: List[Box] = []
        else:
            boxes = self.detector.detect(frame, idx, gt)
        boxes = [b for b in boxes if b.score >= self.min_score]
        t1 = time.perf_counter()
        self.tracker.step(boxes, t, ego_yaw)
        t2 = time.perf_counter()

        self.last_boxes = list(boxes)
        d_ms, k_ms = (t1 - t0) * 1000.0, (t2 - t1) * 1000.0
        self.timings["detect_ms"] += d_ms
        self.timings["track_ms"] += k_ms
        # Per-frame samples, not just a running mean. A mean frame time is the
        # wrong statistic for a control loop: the loop is late whenever a
        # *single* frame overruns, and detector cost here is strongly
        # bimodal -- a full-frame sweep with nothing to crop is cheap, a frame
        # with several candidate tiles is not. p95 is the number that decides
        # whether a rate is actually holdable on hardware.
        self.samples["detect_ms"].append(d_ms)
        self.samples["track_ms"].append(k_ms)
        self.n += 1
        return self.tracker.estimate(self.intr)

    def stage_report(self) -> dict:
        """Mean and p95 per stage, plus the rate the perception half can hold."""
        n = max(1, self.n)
        out = {k: round(v / n, 2) for k, v in self.timings.items()}
        for key, xs in self.samples.items():
            if xs:
                ordered = sorted(xs)
                idx = min(len(ordered) - 1, int(0.95 * len(ordered)))
                out[key.replace("_ms", "_p95_ms")] = round(ordered[idx], 2)
        per_frame = out.get("detect_ms", 0.0) + out.get("track_ms", 0.0)
        out["perception_ms"] = round(per_frame, 2)
        out["perception_fps"] = round(1000.0 / per_frame, 1) if per_frame > 0 else 0.0
        return out


class FusionDetector:
    """This repository's round-7 RGB+motion model, driven in the closed loop.

    The project's own flagship detector, and the right thing to try before
    training anything new: a 4-channel P2-YOLO whose fourth channel is an
    **ego-motion-compensated frame difference** -- grid-LK plus RANSAC
    homography registers t-3 and t-6 onto t, and the channel is the minimum of
    the two residuals. It was trained on ARD-MAV, NPS-Drones and this project's
    own footage, which is a far larger and wider dataset than a simulator can
    produce, and at 25 M parameters it is an order of magnitude larger than a
    nano.

    The ego-registration is why it is worth trying here despite the module
    docstring's warning about temporal methods. That warning applies to the
    *stabilised-stack* detectors, which cancel a static background and have
    nothing to cancel when the camera is translating at 14 m/s. This one does
    not assume a static background -- it explicitly solves for the camera's
    motion first and looks at what is left over, which is exactly the moving
    target. Whether that survives the domain gap to a renderer is a question
    for measurement, not argument, which is what ``tools/compare_detectors.py``
    is for.

    Args:
        weights: A ch=4 fusion checkpoint.
        tile: SAHI tile size the model was trained at.
        conf: Detector confidence floor.
        input_is_rgb: The simulator hands back RGB; the model wants BGR in its
            first three channels.
    """

    name = "fusion"
    needs_frame = True

    def __init__(self, weights: str, tile: int = 640, conf: float = 0.05,
                 device: int | str = 0, dt: int = 3,
                 input_is_rgb: bool = True) -> None:
        from dronedet.methods.fusion import FusionDetector as _Fusion

        self._impl = _Fusion("pursuit-fusion", weights=weights, tile=tile,
                             conf=conf, dt=dt, device=device)
        self.input_is_rgb = bool(input_is_rgb)
        self.weights = str(weights)

    def reset(self) -> None:
        """Drop the frame buffer between episodes.

        The motion channel is built from frames t-3 and t-6, so carrying a
        previous episode's frames across a teleport would difference two
        unrelated places and light the channel up everywhere.
        """
        self._impl._buf.clear()

    def detect(self, frame: np.ndarray, idx: int,
               gt: Optional[dict] = None) -> List[Box]:
        bgr = np.ascontiguousarray(frame[:, :, ::-1]) if self.input_is_rgb else frame
        dets = self._impl.process(idx, bgr, None)
        return [Box(d.x1, d.y1, d.x2, d.y2, d.score, "drone") for d in dets]


def build_detector(kind: str, **kw):
    """Make a detector by name (``oracle``, ``yolo`` or ``fusion``)."""
    if kind == "oracle":
        return OracleDetector(**kw)
    if kind == "yolo":
        return YoloDetector(**kw)
    if kind == "fusion":
        return FusionDetector(**kw)
    raise ValueError(f"unknown detector {kind!r}")
