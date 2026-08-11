"""Guidance: from a bearing in an image to a velocity that makes two drones meet.

A conventional reference stack's visual servo closes on
an object by centring it and driving forward -- yaw on the horizontal offset,
forward speed gated by how centred the target is, stop at a standoff. That is
the right law for the mission it was written for: walk up to a *stationary*
object and hold station in front of it.

It is the wrong law here, and the reason is geometric rather than a matter of
tuning. Aiming at where the target is right now is **pure pursuit**, whose path
always curves in behind a moving target and ends in a tail chase. Against an
evader with any speed at all a tail chase converges only as fast as the speed
difference, and against one that turns it does not converge at all -- the
pursuer spends its whole turn radius following the target *around* the turn
instead of cutting inside it.

So the closure here is **proportional navigation**, the law every real
interceptor uses, and it is chosen because of what a camera can and cannot
measure:

* A camera measures **bearing** essentially exactly -- a pixel is a ray.
* A camera measures **range** badly. Monocular range comes from the target's
  pixel span and its error grows with the square of the range.

PN needs only the *rate of rotation of the line of sight*, which is pure
bearing, and it does the one thing that matters: it steers to drive that
rotation to zero. A line of sight that does not rotate while the range shrinks
is a collision course -- that is the whole idea, and it is true whatever the
target does and whatever the range actually is. Range enters this file only as a
speed schedule and a terminal trigger, where being 20 percent wrong costs a
little time instead of the intercept.

PN commands an **acceleration** across the line of sight, ``N * Vc * ds/dt``,
and the distinction from commanding a *velocity* of similar-looking magnitude is
not cosmetic -- it was measured here. The first version of this file commanded
``N * r * ds/dt`` as a velocity directly; at 40 m range with ``N = 4`` and a
perfectly ordinary 60 mrad/s line-of-sight rate that asks for 10 m/s of instant
sideways crab, the aircraft slews, the slew rotates the line of sight the other
way, and the result is a half-second limit cycle that orbits the target at
constant range forever. The same geometry in acceleration form asks for
3.4 m/s^2, which is a lean, and it converges.

The chaser is commanded in velocity, so the acceleration is turned into one over
a short lookahead ``T`` and added to the across-LOS velocity the aircraft
already has::

    a_perp = N * Vc * ds/dt
    v_cmd  = v_close * s  +  (v_C_perp + a_perp * T)

Keeping ``v_C_perp`` rather than driving it to zero is the whole point of
parallel navigation: once the line of sight has stopped rotating, the course
being flown *is* the collision course and the correct command is to keep flying
it.

``ds/dt`` is taken in the **world** frame. The bearing rate seen in the image
also contains the chaser's own yaw rate, and steering on that is a feedback loop
into its own rotation -- the classic way to make a seeker chase its own tail.

``Vc`` is the *closing* speed, not the chaser's speed. In a tail chase those
differ by a factor of three (14 m/s of airspeed can be 5 m/s of closure), and
since ``Vc`` multiplies the gain, using the wrong one silently triples ``N`` --
straight back into the limit cycle it was set to avoid.

## Latency

A camera frame describes the past. Exposure, transfer and a YOLO forward pass
put 50-150 ms between the world and the bearing, and over that interval an
aircraft yawing at 2 rad/s has turned up to 17 degrees. Two separate errors
follow, and the second is far worse than the first:

* the bearing is **stale** -- it points where the target was, not where it is;
* the bearing is **de-rotated by the wrong heading** -- a body-frame bearing
  measured at t-Δ, resolved into the world using the yaw at t, picks up the
  chaser's own rotation as if it were the target's motion.

The second one is a feedback path from the aircraft's yaw into the signal its
yaw is steering on, which is how a seeker ends up chasing itself. Measured on
this rig, uncompensated: 100 percent of intercepts at zero latency, 91 percent
at one frame, and **6 percent at two frames** -- a cliff, not a slope.

The fix is to stop pretending measurements arrive instantly. Each bearing is
stamped with the time it was *captured*, de-rotated by the heading the aircraft
held at that instant (kept in a short history), and fed to the filters at that
timestamp; the line of sight and range are then propagated forward to now using
their own rates. What the guidance law steers on is an estimate of the present,
built out of an honest measurement of the past.

Everything else in this module exists to keep that one equation fed with
trustworthy numbers: an LOS-rate filter that survives dropped detections, a
range filter that knows range cannot change faster than the aircraft can fly, a
terminal commit for the last few metres where the target is too big for the
detector and too close to correct for, and a search/re-acquire behaviour for
when it is not in frame at all.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .dynamics import BodyCommand, Limits
from .geometry import (
    Intrinsics,
    bearing_from_pixel,
    body_to_world,
    range_from_span,
    world_to_body,
    wrap_pi,
)

SEARCH = "SEARCH"
ACQUIRE = "ACQUIRE"
PURSUE = "PURSUE"
TERMINAL = "TERMINAL"
REACQUIRE = "REACQUIRE"
HIT = "HIT"


# --------------------------------------------------------------------- filters

def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return (0.0, 0.0, 0.0) if n <= 1e-12 else tuple(c / n for c in v)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _solve3(a, b):
    """Gauss-Jordan with partial pivoting; None when the system is degenerate.

    Degeneracy is a real answer here, not an error: it is what a target on a
    collision course looks like -- every bearing parallel, meeting nowhere -- and
    the caller must treat it as "cannot tell" rather than as a verdict.
    """
    m = [[a[i][0], a[i][1], a[i][2], b[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-9:
            return None
        m[col], m[piv] = m[piv], m[col]
        d = m[col][col]
        m[col] = [v / d for v in m[col]]
        for r in range(3):
            if r != col and abs(m[r][col]) > 0.0:
                f = m[r][col]
                m[r] = [m[r][k] - f * m[col][k] for k in range(4)]
    return (m[0][3], m[1][3], m[2][3])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _lead(a, t: float, vertical_scale: float = 1.0):
    """PN lookahead, with the vertical component allowed its own horizon.

    ``a * t`` is the velocity change the across-LOS acceleration will have
    produced by the end of the lookahead -- which assumes the airframe can
    actually deliver ``a`` in every direction. It cannot: vertical acceleration
    authority here is about a third of horizontal, so the vertical part of the
    correction arrives late while the horizontal part arrives on time. Giving
    the vertical component a longer horizon compensates for the slower channel
    instead of pretending the aircraft is isotropic.
    """
    return (a[0] * t, a[1] * t, a[2] * t * vertical_scale)


def _scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


# A pinhole camera's bearings are bounded by its field of view; this cap only
# keeps the tangents finite if a caller ever passes something absurd.
_MAX_BEARING = math.radians(80.0)


def bearing_to_los(az: float, el: float) -> Tuple[float, float, float]:
    """Unit line-of-sight in body axes from azimuth (left +) and elevation (up +).

    This is the exact inverse of
    :func:`~pursuit.geometry.bearing_from_pixel`, and getting that right takes
    care. Those angles are **tangent-plane** angles -- the pinhole model gives
    ``tan(az) = left / forward`` and ``tan(el) = up / forward``, both measured
    against the forward axis independently. Rebuilding a direction as
    ``(cos el cos az, cos el sin az, sin el)`` instead treats them as
    *spherical* angles, where elevation is measured from the plane containing
    the already-rotated azimuth.

    The two agree on the boresight and nowhere else. At 40 degrees off in
    azimuth and 11 up -- an ordinary acquisition geometry for a 76-degree
    camera -- the spherical reconstruction tilts the line of sight by 2.5
    degrees in elevation, which the guidance law then faithfully flies.
    """
    a = max(-_MAX_BEARING, min(_MAX_BEARING, float(az)))
    e = max(-_MAX_BEARING, min(_MAX_BEARING, float(el)))
    return _unit((1.0, math.tan(a), math.tan(e)))


@dataclass
class LosRateFilter:
    """World-frame line-of-sight direction and its rate.

    The rate is the differentiated quantity the whole guidance law rests on, and
    differentiating a measurement is the most reliable way to turn a small error
    into a large one -- so the smoothing here is not incidental.

    Two things it must survive: a *dropped* detection (the tracker coasts, so
    the LOS is unchanged and a naive difference reports a rate of zero and steers
    the aircraft straight) and a *late* one (two frames of motion arriving as one
    sample, which a fixed-dt difference doubles). Both are handled by carrying
    the timestamp of the last accepted sample and differencing against real
    elapsed time, and by refusing to update at all when too little time has
    passed for the difference to mean anything.
    """

    tau: float = 0.12
    """EMA time constant on the rate (s). Long enough to reject per-frame box
    jitter -- at a 20 px target one pixel is ~1 mrad, and at 20 Hz that is 20
    mrad/s of pure noise -- and short enough to follow a real evasive turn,
    which develops over several tenths of a second."""

    max_gap_s: float = 0.5
    """Longer than this since the last sample and the rate is discarded rather
    than differenced: across a half-second gap the target may have reversed, and
    a stale rate would steer confidently in the wrong direction."""

    min_dt_s: float = 1e-3

    s: Optional[Tuple[float, float, float]] = None
    ds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    t_last: Optional[float] = None
    n: int = 0

    def reset(self) -> None:
        self.s = None
        self.ds = (0.0, 0.0, 0.0)
        self.t_last = None
        self.n = 0

    def update(self, s_world, t: float) -> Tuple[float, float, float]:
        """Feed a fresh LOS direction; return the filtered rate (1/s)."""
        s_world = _unit(s_world)
        if self.s is None or self.t_last is None:
            self.s, self.t_last, self.n = s_world, float(t), 1
            self.ds = (0.0, 0.0, 0.0)
            return self.ds
        dt = float(t) - self.t_last
        if dt < self.min_dt_s:
            return self.ds
        if dt > self.max_gap_s:
            self.s, self.t_last, self.n = s_world, float(t), 1
            self.ds = (0.0, 0.0, 0.0)
            return self.ds

        raw = _scale(_sub(s_world, self.s), 1.0 / dt)
        # Keep only the component across the LOS. |s| = 1 makes ds perpendicular
        # in exact arithmetic; after normalisation and float error it is not
        # quite, and the leftover radial part is a fake closing rate.
        raw = _sub(raw, _scale(s_world, _dot(raw, s_world)))
        a = 1.0 - math.exp(-dt / max(1e-6, self.tau))
        self.ds = _add(self.ds, _scale(_sub(raw, self.ds), a))
        self.s, self.t_last = s_world, float(t)
        self.n += 1
        return self.ds

    @property
    def rate_mag(self) -> float:
        return _norm(self.ds)


@dataclass
class RangeFilter:
    """Monocular range, smoothed and kept physically possible.

    Range comes from ``fx * span_m / span_px`` and inherits every wobble of the
    detector's box: a box one pixel wider is 4 percent nearer at 25 px and 25
    percent nearer at 4 px. Two guards, in this order:

    * **Innovation gate.** Between two frames the range cannot change by more
      than the two aircraft can close, plus slack. A sample outside that is a bad
      box, not a teleporting target, and it is rejected rather than smoothed --
      averaging an outlier in is still letting it in.
    * **EMA.** What survives the gate is blended, with the blend keyed to elapsed
      time so a dropped frame does not silently change the filter's bandwidth.

    Rejected samples are counted: a *persistent* disagreement is the signature of
    a wrong span assumption (the box is on something else), which is worth
    knowing rather than filtering away, so after ``max_reject`` in a row the
    filter re-seeds on the measurement instead of defending its own estimate.
    """

    max_closing_mps: float = 30.0
    tau: float = 0.25
    max_reject: int = 6

    value: Optional[float] = None
    rate: float = 0.0
    t_last: Optional[float] = None
    rejects: int = 0

    def reset(self) -> None:
        self.value = None
        self.rate = 0.0
        self.t_last = None
        self.rejects = 0

    def update(self, meas: Optional[float], t: float) -> Optional[float]:
        if meas is None or not math.isfinite(meas) or meas <= 0.0:
            return self.value
        t = float(t)
        if self.value is None or self.t_last is None:
            self.value, self.t_last, self.rejects = float(meas), t, 0
            return self.value
        dt = max(1e-3, t - self.t_last)
        slack = self.max_closing_mps * dt + 0.5 + 0.15 * self.value
        if abs(meas - self.value) > slack:
            self.rejects += 1
            if self.rejects < self.max_reject:
                self.t_last = t
                return self.value
            self.value, self.rate, self.t_last, self.rejects = float(meas), 0.0, t, 0
            return self.value
        self.rejects = 0
        a = 1.0 - math.exp(-dt / max(1e-6, self.tau))
        new = self.value + a * (meas - self.value)
        self.rate = (new - self.value) / dt
        self.value, self.t_last = new, t
        return self.value

    def predict(self, t: float) -> Optional[float]:
        """Range extrapolated to ``t`` (used while coasting through a dropout)."""
        if self.value is None or self.t_last is None:
            return None
        dt = float(t) - self.t_last
        if dt <= 0.0:
            return self.value
        return max(0.1, self.value + self.rate * min(dt, 1.0))


# ---------------------------------------------------------------------- config

@dataclass(frozen=True)
class GuidanceConfig:
    """Tuning for :class:`PursuitGuidance`.

    Attributes:
        nav_gain: PN gain ``N``. Below ~2 the loop removes heading error more
            slowly than a manoeuvring target creates it; 3-5 is the classical
            band; above ~6 it amplifies line-of-sight-rate noise faster than it
            removes error. Note the gain multiplies ``Vc``, so it is a gain on an
            *acceleration*, not on a velocity -- see the module docstring.
        lookahead_s: The ``T`` that turns the commanded acceleration into the
            velocity this platform actually accepts. Roughly the airframe's own
            velocity time constant (speed over acceleration limit): shorter and
            the command is smaller than the aircraft's own lag and nothing
            happens, longer and it overshoots what the acceleration was asking
            for.
        max_lateral_speed: Ceiling on the across-LOS component of the command
            (m/s), applied before the total is saturated. Without it a hard
            correction eats the entire speed budget and the closing component --
            the reason the aircraft is out here -- collapses to nothing, which
            reads as a chaser circling its target at constant range.
        approach_speed: Commanded closing speed in the cruise phase (m/s), before
            the airframe's own speed limit applies.
        terminal_range_m: Range at which guidance commits (see
            :meth:`PursuitGuidance.step`).
        terminal_speed: Closing speed once committed (m/s).
        hit_range_m: Range counted as a collision (m). The Iris is 0.47 m across,
            so two of them touching are ~0.5 m centre to centre; 1.0 m is a
            genuine strike with a little margin for the fact that both aircraft
            are being integrated at 20 Hz.
        capture_speed_scale: Fraction of full speed used while the target is far
            off-boresight. Flying flat out at something 40 degrees off to one
            side sweeps it straight out of the frame -- the aircraft outruns its
            own camera. Scales back in as the target centres.
        boresight_soft_deg: Bearing beyond which that scaling starts.
        boresight_hard_deg: Bearing at which closing speed is fully suppressed
            and the aircraft only turns.
        kp_yaw: Proportional gain from azimuth error to yaw rate (1/s).
        yaw_lead: Feed-forward of the LOS azimuth rate into the yaw command. Pure
            proportional yaw always trails a target that is *still* moving across
            the frame; feeding the rate forward is what keeps a crossing target
            near the centre instead of parked at a constant offset.
        yaw_deadband_rad: Azimuth below which no yaw is commanded (anti-jitter).
        vertical_lead_scale: Extra PN lookahead applied to the *vertical*
            component of the across-LOS correction, as a multiple of
            ``lookahead_s``.

            The vertical channel is not the horizontal one. A quadrotor here has
            22.4 m/s^2 of horizontal acceleration against 7.0 m/s^2 vertically
            and 14.4 m/s of horizontal speed against 5.6 m/s of climb, so an
            identical commanded correction takes about three times as long to
            fly out vertically. PN with a single lookahead therefore leads
            correctly across the frame and *late* up and down, which shows up
            exactly where you would predict: against a target that is itself
            changing height. Measured over the 120-scenario stress matrix with a
            perfect sensor, level policies pass within +/- 1 cm while ``barrel``
            -- a roll that climbs and descends -- passes 11.4 cm high.

            Scaling the vertical lead by the ratio of the two authorities is the
            fix that follows from the cause rather than from a curve fit.

            This replaces a ``vertical_gain`` that was declared, documented and
            never read by anything -- the PN law has always been fully
            three-dimensional. Dead configuration is worse than no
            configuration: it invites exactly the wrong diagnosis, and did.
        search_alt_offset_m: Centre of the search's vertical excursion, relative
            to the altitude the *engagement* started at -- fixed once per
            episode, and that is load-bearing. Re-datuming on every
            search/pursue transition looks harmless and is a ratchet: each cycle
            re-centres the excursion at the altitude the last one left the
            aircraft, adds the offset again, and walks it upward a few metres at
            a time until it is pinned at the ceiling. Bounded relative to a datum
            that keeps moving is not bounded.
        search_alt_amplitude_m: Half-height of that excursion.
        search_alt_gain: Proportional gain from altitude error to climb rate.

            The search has to move vertically at all because this camera's blind
            region is asymmetric -- 15.5 degrees of view above the boresight
            against 32 below -- so the targets it cannot see are overwhelmingly
            the high ones, and altitude is the only control that changes
            elevation on an airframe that cannot pitch.

            But it must be an *excursion around a datum*, not a climb rate, and
            that distinction cost two rounds of failures. A constant climb is
            an unbounded integrator: 3 m/s across a 20-second search is 60 m,
            which pins the aircraft at its ceiling looking down at terrain --
            where a target at the original altitude is not merely far away but
            *below the bottom of the frame* at close range, and the seeker
            latches onto ground clutter instead. Sandbox episodes end in six
            seconds and never exposed it; the 45-second timeout cases did.

            Biased upward rather than centred because the blind region is.
        search_yaw_rate: Yaw rate used for the *slew* between looks (rad/s).
        search_dwell_s: How long the search holds still between slews. The search
            is step-and-stare, not a continuous sweep, and that is the difference
            between acquiring and not.

            A real detector does not see a target the first time it enters the
            frame; it sees it on some fraction of the frames it is present for --
            measured on this rig, 25 percent at 9-14 px. Acquisition therefore
            needs *looks*, and a continuous sweep converts a target into a
            smear of single chances. Worse, it throws away the best look of the
            engagement: at the start of a pursuit the target is often already in
            view, and a search that begins by turning away from it is spending
            the one moment the range is shortest.

            Measured, and the reason this exists: against a perfect sensor a
            continuous 1.4 rad/s sweep acquires in 0.15 s and the guidance scores
            5/5. Against the trained YOLO the same sweep never acquired at all --
            the chaser turned away from a 12-pixel target sitting dead ahead,
            and by the time the sweep came back the evader was at 80 m, where
            the span is 5 px and nothing is detectable. Same guidance, same
            geometry; the sensor was the only difference.
        search_first_dwell_s: Length of the *opening* look, which is longer than
            the rest because it is worth more. At the start of an engagement the
            target is most likely in frame and at its closest -- the highest
            detection probability the whole run will offer -- whereas every later
            look is a uniform prior over a circle. Measured failure this fixes:
            an evader that climbs out of the camera's narrow upward field of view
            (15.5 degrees) is only detectable for the first couple of seconds,
            and a 0.55 s opening look was not reliably long enough to pair two
            detections of a 12-pixel target at a 25 percent hit rate.
        search_step_rad: Angular spacing between looks. Below the field of view
            (1.33 rad here) so consecutive looks overlap and a target cannot fall
            between two of them.

            The looks are ordered *outward and alternating* -- straight ahead,
            one step left, one step right, two steps left, and so on -- rather
            than sweeping one way around the circle. Two reasons, and the second
            is the one that was measured. A target is most likely near the
            heading the engagement started on, so searching outward from there
            finds it soonest on average. And a one-way sweep is asymmetric in a
            way that shows up directly in results: with a left-handed sweep, a
            target 55 degrees to the *right* was found and intercepted while the
            mirror-image scenario 55 degrees to the *left* timed out, because
            the search had to travel most of a circle to reach it.
        search_climb_mps: Amplitude of the vertical sweep folded into the
            acquisition search (m/s). A yaw-only search covers azimuth and
            nothing else, and this camera cannot look up: its principal point
            sits high, so the field of view reaches 15.5 degrees above the
            boresight against 32 below. A target 20 degrees up is not merely
            hard to see, it is *outside the image*, and no amount of turning on
            the spot will ever put it inside one. Since the airframe cannot
            pitch (the camera is bolted to it), the only control that changes
            elevation is altitude -- climbing lowers the angle to something
            above. Measured: this is the whole difference between 32 of 33
            scenarios and all 33.
        search_climb_period_s: Period of the vertical excursion. Long enough
            that the aircraft can actually track the commanded altitude (a 10 m
            swing at a 3-second period would ask for 21 m/s) and that each
            altitude is held long enough for the detector to get several looks
            from it.
        reacquire_hold_s: Coast on the predicted LOS this long after a loss
            before starting to sweep. Most losses are a frame or two of the
            detector blinking, and sweeping through those makes them worse.
        reacquire_timeout_s: Give up re-acquiring and go back to SEARCH.
        confirm_hits: Consecutive detections needed to leave ACQUIRE.
        confirm_miss_tolerance: Misses tolerated inside the confirmation streak.
    """

    nav_gain: float = 4.0
    lookahead_s: float = 0.45
    max_lateral_speed: float = 9.0
    min_los_samples: int = 3
    """Line-of-sight-rate samples required before the rate is believed. The first
    difference after a lock is a step from "no estimate" to "some estimate" and
    carries a rate that never happened; acting on it throws the aircraft
    sideways at the exact moment the track is newest and least trustworthy."""

    approach_speed: float = 14.0
    """Commanded closing speed in cruise, m/s. **Zero means the airframe's own
    horizontal limit.**

    The sentinel exists because this is one of three absolute speeds in this
    config that were tuned against a 14.4 m/s interceptor, and an absolute
    speed stops meaning anything the moment the scenario sets the airframe's
    limits from its own speed advantage. Carried over unchanged to an 18 m/s
    aircraft it gives away a fifth of the closing speed silently -- which costs
    nothing in a suite where the interceptor has all day, and costs the
    building in one where it does not.
    """

    terminal_range_m: float = 6.0
    terminal_speed: float = 16.0
    hit_range_m: float = 1.0

    capture_speed_scale: float = 0.25
    boresight_soft_deg: float = 8.0
    boresight_hard_deg: float = 32.0

    kp_yaw: float = 1.4
    yaw_lead: float = 1.0
    yaw_deadband_rad: float = 0.004

    vertical_lead_scale: float = 1.6
    """Chosen by sweep over the 120-scenario stress matrix on both a perfect and
    a noisy camera; 1.6 minimised both the vertical bias and the total closest
    approach on each. Not the naive 3.2 (the full authority ratio) -- past about
    2 the extra lead starts costing more in overshoot than it recovers in lag."""
    search_alt_offset_m: float = 6.0
    search_alt_amplitude_m: float = 10.0
    search_alt_gain: float = 1.2

    search_yaw_rate: float = 1.4
    search_dwell_s: float = 0.55
    search_first_dwell_s: float = 1.30
    search_step_rad: float = 0.60
    search_climb_mps: float = 3.5
    search_climb_period_s: float = 6.0
    reacquire_hold_s: float = 0.4
    reacquire_timeout_s: float = 6.0
    terminal_nav_scale: float = 1.5
    """Steering gain inside the terminal phase, relative to cruise. An earlier
    version used 0.5 on the reasoning that late corrections are unreliable; that
    is true of corrections computed from a *stale* line of sight and false of
    ones computed from a fresh measurement, and halving both cost 1-3 metres of
    miss distance on every high-latency run."""

    terminal_fresh_s: float = 0.12
    """A terminal-phase measurement younger than this is used directly rather
    than falling back to the committed line of sight."""

    terminal_blind_s: float = 1.2
    """How long a committed terminal run keeps flying with no detection before
    concluding it flew past rather than that the target got too big to see."""

    confirm_hits: int = 3

    promote_window_s: float = 0.5
    promote_hits: int = 2
    promote_score_at_10px: float = 0.70
    promote_score_slope: float = 0.08
    promote_score_min: float = 0.62
    promote_score_max: float = 0.84
    """Confidence a track must reach before guidance will *steer* on it.

    The measured answer to urban clutter, and the third state this loop was
    missing. Seeding stays cheap (``TrackerConfig.init_score`` = 0.20) so a
    faint target at the edge of the envelope still starts a track; what changes
    is that the aircraft will not commit to flying at one until it has produced
    ``promote_hits`` detections inside ``promote_window_s`` scoring at least
    ``T(span)``. Before that the track exists and is tracked; it just does not
    get to point the aircraft.

    The threshold is **normalised by pixel span**, and that is the whole reason
    this works where a flat threshold did not::

        T(span) = clip(a + b * log2(span / 10), lo, hi)
                = 0.62 @ 4 px,  0.70 @ 10 px,  0.78 @ 20 px

    A four-pixel drone cannot be expected to score like a twenty-pixel one. A
    flat 0.55 bar was tried live and destroyed acquisition -- a genuine target
    at 80 m falls under it -- while a bar low enough to admit that target admits
    every rooftop. Conditioning on span dissolves the conflict, and the slope is
    the load-bearing parameter: at b=0 the town still suffers 3-5 early wrong
    commits at every intercept of a, at b=0.08 it suffers 1-2, consistently
    across a 270-cell sweep.

    Measured over 46 recorded episodes -- seconds spent steering at something
    that was not the drone:

    | scene | before | after |
    |---|---|---|
    | rivermark | 279.1 s | **69.0 s** |
    | rivermark (new detector) | 144.6 s | **11.8 s** |
    | skydome | 124.4 s | **7.2 s** |

    Seconds spent on the actual drone barely move (27.00 -> 26.25), which is the
    point: this is not a filter on detections, it is a licence to steer.

    The flag **latches**. Without that it chatters, and PN is handed a lock that
    appears and disappears -- worse than either state held steadily.
    """
    confirm_miss_tolerance: int = 1

    lock_probation_s: float = 8.0
    lock_min_closure_m: float = 3.0
    lock_refractory_s: float = 2.5

    score_probation_frames: int = 0
    score_probation_hits: int = 2
    score_probation_min: float = 0.6
    """A new lock must prove itself with confident detections, or die.

    The single most effective thing in this file for urban flight, and it works
    because it separates the two jobs a confidence threshold was being asked to
    do at once. Seeding must stay cheap: a real drone at the far edge of the
    envelope is a handful of pixels and scores badly, and a bar high enough to
    exclude a rooftop excludes it too -- measured, raising the seed threshold to
    0.55 turned a clean 0.38 m intercept into a run that never acquired.

    So seed cheaply and audit immediately. Within ``score_probation_s`` of a
    track being confirmed it must have produced ``score_probation_hits``
    detections scoring at least ``score_probation_min``; otherwise the lock is
    dropped and the refractory period keeps it dropped. A genuine target
    produces those easily -- its median detector score is 0.80 -- while clutter
    in Rivermark sits at 0.24 and only rarely spikes.

    **Off by default (0), because it did not survive its own measurement.**
    That figure came from an analysis that segmented locks by ego-compensated
    pixel jumps; re-derived against whole held tracks it is far weaker. Swept
    over window, hit count and threshold on the recorded engagements, the best
    setting that keeps every true lock removes only about 30 percent of false
    ones -- and the true-lock sample is n=4, which is not enough to tune a
    threshold against without fitting it to noise.

    It also almost never fires as originally specified: false locks in the town
    have a median life of 9 frames, so a 20-detector-frame window expires after
    the lock has already ended on its own. In a live Rivermark run it rejected
    exactly nothing.

    Kept and configurable because the underlying separation is real -- detector
    score does distinguish a drone from a rooftop at AUC 0.95 -- and it may
    become useful once the detector produces fewer confident false positives.
    The honest fix for that is upstream: train on the clutter it is getting
    wrong (:mod:`pursuit.tools.mine_negatives`), rather than filter afterwards.

    It is a *probation*, not a filter: nothing is suppressed while it runs, so a
    real target loses nothing. Only the verdict at the end of the window acts.

    The window is counted in **detector frames, not seconds**, and that
    distinction is load-bearing. A wall-clock window expires while a perfectly
    good track is coasting through a dropout -- it would have executed exactly
    the target the reacquire logic exists to protect. Counting opportunities
    means a track is only ever judged on evidence it actually had the chance to
    produce.
    """

    lock_max_hold_s: float = 45.0
    """Hard ceiling on how long one lock may be held without arriving.

    A backstop for everything the smarter tests miss, and set from measurement
    rather than intuition. The longest *successful* lock across every suite is
    37.5 s -- a crossing target first seen at 95 m, which is a real intercept
    and takes that long because a crossing engagement is flown twice. Anything
    at or below 37 s therefore costs genuine kills, which is why this is 45 and
    not the 20 that feels right: a cap tight enough to be the discriminator is
    tight enough to throw away the hardest real engagements, and the static test
    below is the discriminator instead.

    The closure probation above cannot catch this case, and the reason is worth
    stating: a lamp post *does* close. Fly at a fixed object and its range falls
    exactly as fast as you approach, so it passes a closure test more
    convincingly than a fleeing drone does. Something else has to notice that it
    is a post, and until it does, this stops the aircraft spending the whole
    engagement on it.
    """

    static_reject_s: float = 0.0
    """Window of bearings for the static-object test. **Zero: the test is off.**

    Kept, with its implementation, as a recorded negative result -- it is an
    obvious idea, it nearly works, and the reason it cannot is worth knowing
    before anyone builds it again.

    The idea: a drone moves and a rooftop does not, and that is measurable from
    bearings alone, which is the one thing this sensor reports exactly. Fit a
    single fixed world point to the recent (position, bearing) pairs; a small
    residual means everything seen is consistent with one static object.

    It fails on geometry, not on noise. Measured over a 2.5 s window with the
    chaser at 14 m/s:

    | target | residual |
    |---|---|
    | static post | 0.05 m |
    | drone crossing at 9 m/s | 0.62 m |
    | **drone fleeing at 9 m/s** | **0.00 m** |

    A target moving *parallel to the observer* generates bearings that all pass
    through one point, so it is geometrically indistinguishable from something
    nailed down -- and parallel motion is exactly what a tail chase is. There is
    no tolerance that separates them: 0.25 m would catch the post and still
    reject the fleeing drone. A false rejection discards a real target, which is
    far worse than wasting time on a post.

    There is a second, independent reason it cannot carry the load. Pursuit is a
    collision course by construction, so almost all of the chaser's translation
    is *along* the line of sight and contributes no parallax at all: after 35 m
    of closing, the perpendicular baseline is centimetres. The test is weakest
    exactly when it is needed.

    Set it non-zero only with a deliberate cross-LOS manoeuvre to generate
    parallax; without one it will reject real drones.
    """

    _static_reject_doc_window: float = 2.5
    """Window the test uses when enabled. Separate from the on/off switch so
    turning it on does not also require rediscovering a sane window."""

    static_reject_tol_m: float = 1.2
    """Residual below which the bearings are judged consistent with one fixed
    point. Sized from the sensor rather than fitted: a pixel of bearing noise at
    100 m is 0.11 m of cross-range, so a true static object fits to a few tens
    of centimetres, while a drone crossing at 9 m/s leaves tens of metres."""

    static_reject_baseline_frac: float = 0.12
    """Chaser translation needed, as a fraction of range, before the test is
    allowed to fire. Triangulation from a short baseline is ill-conditioned and
    would call a real drone static -- which is far worse than missing a post."""
    """A lock must *close*. Fly at something for ``lock_probation_s`` and if the
    estimated range has not fallen by ``lock_min_closure_m``, the lock is not a
    target and the mission goes back to searching.

    This is the only check in the file that can tell a persistent false lock from
    a real one, and it works because it appeals to physics rather than to
    appearance. Measured failure it exists for: the seeker locked onto a fixed
    feature on the horizon -- a detection that stayed at pixel (694, 242) for
    forty seconds while the chaser flew 400 m at it. Its estimated range sat at
    35 m the entire time, because a thing that does not move in the image and
    does not grow is infinitely far away. No confidence threshold, gate or
    corroboration rule can see that; only time can.

    The threshold is generous on purpose. A genuine pursuit closes 35 m in about
    seven seconds, and even the hardest scenario in the suite (a 13 m/s evader
    against a 14.4 m/s chaser) closes 11 m in eight. Three metres asks only that
    the range is going the right way.

    ``lock_refractory_s`` is what makes the rejection stick. Dropping the lock
    alone is useless against *persistent* clutter: the offending detection is
    still there on the next frame, and the seeker re-confirms it three frames
    later and starts the eight-second clock again. So a rejection also blocks
    confirmation for a couple of seconds and sends the search straight into its
    next slew -- by the time it can lock again it is looking somewhere else."""

    sensor_latency_s: float = 0.0
    """Age of a bearing by the time guidance acts on it, in seconds.

    A per-deployment **calibration**, not a tuning knob, and its default has to
    be the value that is true for an uncalibrated pipeline. For this rig that is
    genuinely zero: the frame guidance acts on at time ``t`` was rendered from
    the poses the aircraft are at during ``t``, so there is no age to correct.
    (The five-render flush in the simulator exists precisely so that latency is
    something an experiment adds on purpose.)

    Declaring a latency you do not have is not harmless -- it is the same
    over-prediction as too much lookahead. Declaring 0.05 against a true 0 costs
    measurable robustness (``full`` dropout 0.75: 23/42 -> 16/42), and declaring
    0.30 against a true 0.05 collapses the loop (120/120 -> 65/120).

    On hardware, set it from a *measured* frame-to-bearing age. It is worth
    real money when it is right: at a true 150 ms, an uncompensated loop scores
    17/42 and a correctly-declared one 36/42.
    """

    omnidirectional: bool = False
    """The sensor sees every direction, so pointing is no longer a manoeuvre.

    Three behaviours in this file exist only because a single forward camera
    cannot see behind itself, and all three become wrong -- not merely
    unnecessary -- the moment a ring is fitted:

    **The search stops slewing.** Step-and-stare covered azimuth by turning the
    aircraft, at 10 s for a full circle. There is nothing left to turn toward,
    and turning is now actively harmful: the long-range half of the ring's
    sensing is frame differencing against a *stationary* background (see
    :class:`~pursuit.ring.RingMotionDetector`), and a camera that is slewing has
    no stationary background. SEARCH becomes holding still and paying attention,
    which is also what a point-defence interceptor is supposed to do.

    **The re-acquisition sweep stops.** Same reason. A lost target is somewhere
    in a sphere the aircraft is already watching all of; sweeping cannot improve
    that and does destroy the differencing.

    **The closing speed stops being gated by boresight angle.**
    ``capture_speed_scale`` throttles the aircraft to a quarter speed at a
    target 40 degrees off the nose, because closing hard on something near the
    frame edge sweeps it *out* of frame -- the aircraft outrunning its own
    camera. With 360 degree coverage there is no frame edge to sweep out of, and
    the throttle costs exactly what it used to buy: in a race against an
    intruder that is going somewhere, three quarters of the closing speed is the
    engagement.

    Yaw still tracks the target, and that is not a leftover. It keeps the target
    off the 6 degree seams where two cameras hand it over, and it points the
    airframe where it is going.
    """

    los_tau_s: float = 0.12
    """EMA time constant of the line-of-sight rate filter. It trades noise
    rejection against its own phase lag, and that lag adds to the sensor's --
    so the right value depends on how late the measurements already are."""

    max_extrapolation_s: float = 0.35
    """Ceiling on how far forward a measurement is propagated. Beyond a few
    tenths of a second an extrapolated line of sight is a guess about what the
    target decided to do, and a confident guess is worse than an admitted
    absence of information."""


@dataclass
class GuidanceState:
    """Everything one guidance tick decided, for logging and the HUD."""

    mode: str = SEARCH
    command: BodyCommand = field(default_factory=BodyCommand)
    az: Optional[float] = None
    el: Optional[float] = None
    range_est: Optional[float] = None
    los_rate: float = 0.0
    lateral_speed: float = 0.0
    closing_speed: float = 0.0
    boresight_deg: Optional[float] = None
    confirmed: bool = False
    streak: int = 0
    lost_for_s: float = 0.0
    age_s: float = 0.0
    note: str = ""


# -------------------------------------------------------------------- guidance

class PursuitGuidance:
    """Bearing in, body velocity out: the intercept law and the mode it is in.

    Args:
        intr: Camera the bearings come from.
        limits: The chaser's own limits, so the commanded speed is never larger
            than the airframe can hold (which would silently change the ratio of
            closing to lateral speed the law is trying to set).
        target_span_m: Physical span of the target, for monocular range.
        config: Tuning.
    """

    def __init__(self, intr: Intrinsics, limits: Limits, target_span_m: float,
                 config: Optional[GuidanceConfig] = None) -> None:
        self.intr = intr
        self.limits = limits
        self.target_span_m = float(target_span_m)
        self.cfg = config or GuidanceConfig()
        self.approach_speed = (self.cfg.approach_speed
                               if self.cfg.approach_speed > 0.0
                               else limits.max_speed_xy)
        self.los = LosRateFilter(tau=self.cfg.los_tau_s)
        self.rng = RangeFilter(max_closing_mps=limits.max_speed_xy * 2.5)
        self.reset()

    def reset(self) -> None:
        self.mode = SEARCH
        self.streak = 0
        self.misses = 0
        self.confirmed = False
        self.lost_for_s = 0.0
        self.t = 0.0
        self._search_dir = 1.0
        self._search_hold_s = 0.0
        self._search_slewed = 0.0
        self._search_looks = 0
        self._search_alt0: Optional[float] = None
        self._chaser_z: float = 0.0
        self._probation_t0: Optional[float] = None
        self._probation_r0: Optional[float] = None
        self._promote_hist: deque = deque(maxlen=60)   # (t, score, span)
        self._promoted: bool = False
        self._lock_t0: Optional[float] = None
        self._score_probe_frames: int = -1     # -1 = not armed
        self._score_probe_hits: int = 0
        # (t, chaser position, world-frame line of sight) at each *detection*.
        # Coasted predictions are excluded: they are generated from the model
        # being tested, so feeding them in would let any hypothesis fit itself.
        self._static_hist: deque = deque(maxlen=120)
        self._refractory_until: float = -1.0
        self._terminal_los: Optional[Tuple[float, float, float]] = None
        self._terminal_range: Optional[float] = None
        # Heading history, for de-rotating a bearing by the yaw the aircraft
        # actually held when the frame was captured. A second is ample: nothing
        # is compensated beyond `max_extrapolation_s` anyway.
        self._yaw_hist: deque = deque(maxlen=64)
        self.los.reset()
        self.rng.reset()

    def _yaw_at(self, t_query: float, fallback: float) -> float:
        """The chaser's heading at ``t_query``, interpolated from the history.

        Interpolated rather than nearest-sample: at 2 rad/s a half-tick of
        rounding is three degrees, which is most of the error this whole
        mechanism exists to remove.
        """
        hist = self._yaw_hist
        if not hist:
            return fallback
        if t_query >= hist[-1][0]:
            return hist[-1][1]
        if t_query <= hist[0][0]:
            return hist[0][1]
        prev = hist[0]
        for cur in hist:
            if cur[0] >= t_query:
                span = cur[0] - prev[0]
                if span <= 1e-9:
                    return cur[1]
                u = (t_query - prev[0]) / span
                return prev[1] + wrap_pi(cur[1] - prev[1]) * u
            prev = cur
        return hist[-1][1]

    # -- main tick ----------------------------------------------------------

    def step(self, t: float, dt: float, chaser_xyz, chaser_yaw: float, chaser_vel,
             measurement) -> GuidanceState:
        """Advance one control tick.

        Args:
            t: Monotonic time (s).
            dt: Seconds since the previous tick.
            chaser_xyz: Own world position (used only for logging and the
                vertical term's geometry).
            chaser_yaw: Own heading (rad) -- needed to rotate the body-frame
                bearing into the world frame before differentiating it.
            chaser_vel: Own world velocity ``(vx, vy, vz)``. The PN law needs it
                because it commands the chaser's *across-LOS* velocity, and the
                part of that it already has is not an error to be corrected.
            measurement: A :class:`~pursuit.perception.TargetEstimate`, or None
                when the target was not seen this frame.

        Returns:
            The :class:`GuidanceState` for this tick, whose ``command`` is what
            should be flown.
        """
        self.t = float(t)
        cfg = self.cfg
        st = GuidanceState(mode=self.mode)
        self._yaw_hist.append((self.t, float(chaser_yaw)))
        self._chaser_z = float(chaser_xyz[2])

        # Two different questions, and conflating them was a bug.
        #
        #   `tracked`  -- is there a lock at all? The tracker's coast answers
        #                 yes, and that is exactly its job: bridging the frames
        #                 the detector missed is what keeps the mission in
        #                 PURSUE instead of sweeping for a target that is right
        #                 there.
        #   `measured` -- did a *camera* see it this frame? Only that may touch
        #                 the bearing filter.
        #
        # Feeding a coast into the line-of-sight filter reintroduces, through
        # the back door, precisely the failure the latency machinery above
        # exists to prevent. The tracker coasts by predicting constant motion in
        # the *image*, which knows nothing about the chaser's heading; de-rotate
        # that into the world with a yaw that has since changed and the filter
        # reports the chaser's own yaw rate as line-of-sight rotation. Measured
        # with both aircraft frozen (true rate identically zero) and the chaser
        # yawing at 1.5 rad/s: the reported rate relaxes to 1.45 rad/s. In the
        # closed loop at 0.75 dropout it costs four intercepts out of 33.
        tracked = measurement is not None and measurement.valid
        measured = tracked and measurement.source == "detector"
        if measured:
            self._ingest(measurement, chaser_yaw, chaser_xyz)
        if tracked:
            st.az, st.el = measurement.az, measurement.el
            self.lost_for_s = 0.0
        else:
            self.lost_for_s += float(dt)
        # Whether or not this tick had a measurement, the guidance law is given
        # an estimate of *now*: the last accepted bearing rotated forward at its
        # own rate, and the last accepted range advanced at its own rate.
        st.range_est = self.rng.predict(self.t)
        st.age_s = round(self._measurement_age(), 3)

        self._advance_mode(tracked, measured, st.range_est, dt)
        if self.confirmed and self._lock_t0 is None:
            self._lock_t0 = self.t
            self._score_probe_frames = 0
            self._score_probe_hits = 0
        # Three independent ways for a lock to be wrong, cheapest first. They
        # catch different things and none subsumes another: a post closes
        # (so probation passes) but does not move (so the static test fires);
        # a target lost in clutter may do neither, and only the clock notices.
        if (self.confirmed and self._score_probe_frames >= 0
                and cfg.score_probation_frames > 0
                and self._score_probe_frames >= cfg.score_probation_frames
                and self._score_probe_hits < cfg.score_probation_hits):
            st.note = ("lock rejected: never scored above "
                       f"{cfg.score_probation_min} -- clutter, not a drone")
            self._drop_lock()
        elif self._lock_is_going_nowhere(st.range_est):
            st.note = "lock rejected: no closure"
            self._drop_lock()
        elif self.confirmed and self._looks_static(st.range_est):
            st.note = "lock rejected: fixed object, not a drone"
            self._drop_lock()
        elif (self.confirmed and self._lock_t0 is not None
                and cfg.lock_max_hold_s > 0.0
                and self.t - self._lock_t0 > cfg.lock_max_hold_s):
            st.note = "lock rejected: held too long without arriving"
            self._drop_lock()
        st.mode = self.mode
        st.confirmed = self.confirmed
        st.streak = self.streak
        st.lost_for_s = round(self.lost_for_s, 3)
        st.los_rate = round(self.los.rate_mag, 5)

        # ACQUIRE flies the closure like PURSUE does, which is right for one
        # forward camera -- there, moving toward a candidate is how you find out
        # what it is. With a ring it is exactly wrong, and it cost the first
        # live city run every engagement. Closing on an unpromoted contact
        # starts the aircraft moving, and the moment it moves two things die at
        # once: the background model that sees a 3 px drone at 140 m, and the
        # only test that can tell a building from an aircraft (a fixed object's
        # bearing is constant *from a fixed observer* and moves like anything
        # else from a moving one). So the seeker talks itself into a rooftop,
        # and the evidence that would have refuted it is destroyed by the act of
        # chasing it. Hold station until the lock is worth moving for.
        holding = (self.mode == SEARCH
                   or (cfg.omnidirectional and self.mode == ACQUIRE
                       and not self.confirmed))
        if holding:
            st.command = self._search_command()
            self._advance_search_phase(float(dt))
            st.note = {"search:watch": "watching all quarters",
                       "search:look": "looking"}.get(
                           st.command.source, "slewing to the next look")
        elif self.mode == REACQUIRE:
            st.command = self._reacquire_command(chaser_yaw, chaser_vel, st)
        else:
            # PURSUE, TERMINAL and HIT all fly the closure. HIT is a *report*
            # that the estimated range crossed the threshold, not an instruction
            # to stop -- the episode is scored on the true closest approach, and
            # braking on a monocular range that reads long would end every run
            # hovering just outside the target.
            st.command = self._closure_command(chaser_yaw, chaser_vel, st)
        return st

    # -- measurement --------------------------------------------------------

    def _ingest(self, m, chaser_yaw: float, chaser_xyz) -> None:
        # A bearing describes the moment its frame was captured, not the moment
        # it arrived. Stamping it correctly is what lets the filters below
        # differentiate against real elapsed time, and de-rotating it by the
        # heading held *then* is what keeps the chaser's own yaw out of the
        # line-of-sight rate. See the module docstring.
        t_meas = self.t - max(0.0, self.cfg.sensor_latency_s)
        yaw_meas = self._yaw_at(t_meas, chaser_yaw)
        # A ring reports the direction itself, because a target 170 degrees off
        # the nose has no tangent-plane azimuth to report. One forward camera
        # cannot produce that case, so it keeps reporting the pinhole angles it
        # measures and they are rebuilt here.
        s_body = (m.los_body if getattr(m, "los_body", None) is not None
                  else bearing_to_los(m.az, m.el))
        s_world = body_to_world(yaw_meas, *s_body)
        self.los.update(s_world, t_meas)
        # Recorded here and nowhere else: only a real detection may enter the
        # static-object test. A coasted prediction is produced by the very model
        # under test, so including one would let the hypothesis confirm itself.
        self._static_hist.append(
            (t_meas, tuple(float(v) for v in chaser_xyz), s_world))
        self._promote_hist.append((t_meas, float(m.score),
                                   float(m.span_px or 0.0)))
        self._update_promotion()
        if self._score_probe_frames >= 0:
            self._score_probe_frames += 1
            if m.score >= self.cfg.score_probation_min:
                self._score_probe_hits += 1
        # Where in the image the box sits is part of what its width means: a
        # pinhole stretches by sec^2 toward the frame edge, which is 2.2x at the
        # corner of a 96-degree ring camera. See geometry.offaxis_scale.
        r = range_from_span(self.intr, m.span_px, self.target_span_m, m.u, m.v)
        if m.range_override is not None:
            r = m.range_override
        self.rng.update(r, t_meas)
        self._clamp_los_rate()

    def _measurement_age(self) -> float:
        if self.los.t_last is None:
            return 0.0
        return max(0.0, self.t - self.los.t_last)

    def _los_now(self):
        """The line of sight propagated from its measurement time to now."""
        s = self.los.s
        if s is None:
            return None
        age = min(self._measurement_age(), self.cfg.max_extrapolation_s)
        if age <= 0.0:
            return s
        return _unit(_add(s, _scale(self.los.ds, age)))

    def _clamp_los_rate(self) -> None:
        """Reject a line-of-sight rate no real geometry could have produced.

        ``ds/dt`` is a differentiated measurement de-rotated by a measured
        heading, and both of those can be wrong at once. The physical bound is
        not: the line of sight can only rotate as fast as the transverse
        component of the relative velocity divided by the range, and neither
        aircraft can fly faster than its own limits. Anything above that came
        from a bad box, a stale bearing or a rotating frame -- and since the
        steering command is proportional to this number, letting one through
        buys a violent turn at nothing.

        A clamp rather than a discard: the *direction* of an over-large rate is
        usually still right, and steering hard in the correct direction is
        recoverable in a way that steering at a rejected zero is not.
        """
        r = self.rng.value
        if r is None:
            return
        v_rel_max = 2.2 * self.limits.max_speed_xy
        lim = v_rel_max / max(2.0, r)
        mag = _norm(self.los.ds)
        if mag > lim > 0.0:
            self.los.ds = _scale(self.los.ds, lim / mag)

    # -- modes --------------------------------------------------------------

    def _advance_mode(self, tracked: bool, measured: bool,
                      range_est: Optional[float], dt: float) -> None:
        """Advance the mission mode.

        ``tracked`` decides whether there is still a lock to fly on; ``measured``
        decides whether the *detector* saw the target, and only that is allowed
        to build the acquisition streak -- a lock confirmed by the tracker's own
        predictions would be a lock confirming itself.
        """
        cfg = self.cfg
        if measured:
            self.streak += 1
            self.misses = 0
        else:
            self.misses += 1
            if self.misses > cfg.confirm_miss_tolerance and not self.confirmed:
                self.streak = 0

        if not self.confirmed:
            # A lock just rejected for not closing must not be re-made from the
            # very detections that produced it. Hold confirmation off until the
            # search has moved somewhere else.
            if self.t < self._refractory_until:
                self.streak = 0
                self.mode = SEARCH
                return
            if self.streak >= cfg.confirm_hits and self._promoted:
                self.confirmed = True
                self.mode = PURSUE
            else:
                self.mode = ACQUIRE if self.streak > 0 else SEARCH
            return

        if not tracked:
            if self.lost_for_s >= cfg.reacquire_timeout_s:
                self.confirmed = False
                self.streak = 0
                self.mode = SEARCH
                self._search_hold_s = 0.0
                self._search_slewed = 0.0
                self._search_looks = 0
                self.los.reset()
                self.rng.reset()
                self._terminal_los = None
            elif self.mode in (TERMINAL, HIT):
                # Inside the commit range the target is *expected* to stop being
                # detectable -- it is metres away, filling the frame, and far
                # outside anything the detector was trained on -- so a lost
                # detection here is not evidence of a lost target and the answer
                # is to keep going. Bounded, though: past `terminal_blind_s` the
                # more likely explanation is that the aircraft flew past, and
                # then re-acquisition is exactly right.
                if self.lost_for_s > cfg.terminal_blind_s:
                    self.mode = REACQUIRE
                    self._terminal_los = None
            else:
                self.mode = REACQUIRE
            return

        # Modes are recomputed from range every tick rather than latched. An
        # intercept that misses flies *past* the target, and a latched terminal
        # state would keep committing to a line of sight that is now behind the
        # aircraft. Falling back to PURSUE is what turns a miss into a second
        # attempt, which is the difference between "closes on the target" and
        # "closes on the target, once".
        if range_est is None:
            self.mode = PURSUE
        elif range_est <= cfg.hit_range_m:
            self.mode = HIT
        elif range_est <= cfg.terminal_range_m:
            if self.mode not in (TERMINAL, HIT):
                self._terminal_los = self._los_now()
                self._terminal_range = range_est
            self.mode = TERMINAL
        else:
            self.mode = PURSUE
            self._terminal_los = None

    def _lock_is_going_nowhere(self, range_est: Optional[float]) -> bool:
        """True when the thing being chased is provably not being closed on.

        See ``lock_probation_s``. The window restarts every time real progress is
        made, so a long pursuit that keeps closing is never interrupted -- only
        one that has stopped closing, or never started.
        """
        cfg = self.cfg
        if not self.confirmed or self.mode in (SEARCH, HIT) or range_est is None:
            self._probation_t0 = None
            return False
        if self._probation_t0 is None or self._probation_r0 is None:
            self._probation_t0, self._probation_r0 = self.t, range_est
            return False
        if self.t - self._probation_t0 < cfg.lock_probation_s:
            return False
        closed = self._probation_r0 - range_est
        if closed >= cfg.lock_min_closure_m:
            self._probation_t0, self._probation_r0 = self.t, range_est
            return False
        return True

    def _promotion_floor(self, span_px: float) -> float:
        """Confidence a detection of this pixel size must reach to count.

        Normalised by span because detector confidence is a function of how many
        pixels the target occupies, and treating a 4 px contact and a 20 px one
        with the same bar guarantees being wrong at one end of the range.
        """
        cfg = self.cfg
        sp = max(1.0, float(span_px))
        t = cfg.promote_score_at_10px + cfg.promote_score_slope * math.log2(sp / 10.0)
        return min(cfg.promote_score_max, max(cfg.promote_score_min, t))

    def _update_promotion(self) -> None:
        """Latch once the track has earned the right to steer the aircraft."""
        cfg = self.cfg
        if self._promoted or cfg.promote_hits <= 0:
            if cfg.promote_hits <= 0:
                self._promoted = True
            return
        hits = sum(1 for t, sc, sp in self._promote_hist
                   if self.t - t <= cfg.promote_window_s
                   and sc >= self._promotion_floor(sp))
        if hits >= cfg.promote_hits:
            self._promoted = True

    def _looks_static(self, range_est: Optional[float]) -> bool:
        """True when the recent bearings all pass through one fixed world point.

        This is the test that separates a drone from a rooftop, and it exists
        because the closure test cannot: fly at a lamp post and its range falls
        exactly as fast as you approach it, so it passes closure more
        convincingly than a fleeing drone does.

        Bearings only. A static world point ``X`` has ``X - p_i`` parallel to
        ``s_i`` for every sample, so ``X`` is the least-squares solution of
        ``sum (I - s_i s_i^T)(X - p_i) = 0`` and the residual says how well one
        fixed point explains what was seen. Range never enters the fit, which
        matters: range here is monocular and poor, and a test built on it would
        inherit that.

        Two guards, both there to protect the *drone* rather than to catch the
        post. The chaser must have translated enough to triangulate at all,
        because a short baseline makes everything look static; and the normal
        matrix must be usable, because a target on a collision course produces
        parallel bearings that meet nowhere. Both cases mean "cannot tell", and
        that must never be reported as "not a drone" -- a false rejection throws
        away a real target, which is far worse than chasing a post for a while.
        """
        cfg = self.cfg
        if cfg.static_reject_s <= 0.0 or len(self._static_hist) < 6:
            return False
        window = [h for h in self._static_hist
                  if self.t - h[0] <= cfg.static_reject_s]
        if len(window) < 6 or (window[-1][0] - window[0][0]) < cfg.static_reject_s * 0.6:
            return False

        # Baseline PERPENDICULAR to the line of sight, not total distance
        # flown. Flying straight at something produces no parallax however far
        # you travel, and a pursuit is by construction nearly a collision
        # course -- so total translation says the geometry is informative at
        # exactly the moments it is not. Measured against the current bearing,
        # 35 m of closing gives a perpendicular baseline of centimetres, which
        # is the honest number and correctly refuses to answer.
        s_now = window[-1][2]
        p0 = window[0][1]
        perp = 0.0
        for _t, q, _s in window:
            d = [q[i] - p0[i] for i in range(3)]
            along = sum(d[i] * s_now[i] for i in range(3))
            perp = max(perp, math.sqrt(max(0.0, sum(c * c for c in d) - along * along)))
        if range_est is None or perp < cfg.static_reject_baseline_frac * range_est:
            return False

        a = [[0.0] * 3 for _ in range(3)]
        b = [0.0, 0.0, 0.0]
        for _t, p, sdir in window:
            for i in range(3):
                for j in range(3):
                    m = (1.0 if i == j else 0.0) - sdir[i] * sdir[j]
                    a[i][j] += m
                    b[i] += m * p[j]
        x = _solve3(a, b)
        if x is None:
            return False

        acc = 0.0
        for _t, p, sdir in window:
            d = [x[i] - p[i] for i in range(3)]
            along = sum(d[i] * sdir[i] for i in range(3))
            acc += sum((d[i] - along * sdir[i]) ** 2 for i in range(3))
        return math.sqrt(acc / len(window)) <= cfg.static_reject_tol_m

    def _drop_lock(self) -> None:
        """Abandon the current lock and start searching again, from scratch."""
        self.confirmed = False
        self.streak = 0
        self.misses = 0
        self.mode = SEARCH
        self.lost_for_s = 0.0
        self._refractory_until = self.t + self.cfg.lock_refractory_s
        # Start in the *slew*, not a look: the thing just rejected is straight
        # ahead, and looking at it again is the one thing not worth doing.
        self._search_hold_s = float("inf")
        self._search_slewed = 0.0
        self._probation_t0 = None
        self._probation_r0 = None
        self._promote_hist.clear()
        self._promoted = False
        self._lock_t0 = None
        self._score_probe_frames = -1
        self._score_probe_hits = 0
        self._static_hist.clear()
        self._terminal_los = None
        self.los.reset()
        self.rng.reset()

    # -- commands -----------------------------------------------------------

    def _closure_command(self, chaser_yaw: float, chaser_vel,
                         st: GuidanceState) -> BodyCommand:
        """Proportional navigation (see the module docstring for the derivation)."""
        cfg = self.cfg
        s = self._los_now()
        if s is None:
            return self._search_command()

        v_own = tuple(float(c) for c in chaser_vel)
        v_own_perp = _sub(v_own, _scale(s, _dot(v_own, s)))

        # -- the steering term ------------------------------------------------
        # a_perp = N * Vc * ds/dt, converted to the velocity increment this
        # platform accepts. Vc is the CLOSING speed; see the module docstring for
        # why the chaser's own speed is not a stand-in for it.
        ds = self.los.ds if self.los.n >= cfg.min_los_samples else (0.0, 0.0, 0.0)
        vc = self._closing_speed(v_own, s)
        a_perp = _scale(ds, cfg.nav_gain * vc)
        lat = _add(v_own_perp,
                   _lead(a_perp, cfg.lookahead_s, cfg.vertical_lead_scale))

        # Cap the across-LOS component on its own, before the total is
        # saturated, so a hard correction cannot consume the entire speed
        # budget and leave nothing to close with.
        lat_mag = _norm(lat)
        if lat_mag > cfg.max_lateral_speed:
            lat = _scale(lat, cfg.max_lateral_speed / lat_mag)
            lat_mag = cfg.max_lateral_speed
        st.lateral_speed = round(lat_mag, 3)

        bore = math.acos(max(-1.0, min(1.0, _dot(
            s, body_to_world(chaser_yaw, 1.0, 0.0, 0.0)))))
        st.boresight_deg = round(math.degrees(bore), 2)

        if self.mode in (TERMINAL, HIT):
            # Commit. Inside a few metres three things break at once: the target
            # is large enough that the detector trained on small ones starts
            # missing it, one pixel of box error is a large fraction of the
            # remaining range, and there is no longer time to fly out a
            # correction anyway. The aircraft flies the LOS it had when it
            # committed, at full speed, and stops arguing with the seeker.
            #
            # It also does not stop at "arrived". The mission is a collision, so
            # a terminal state that brakes at the declared hit range would hold
            # station one metre short of the only thing being asked for --
            # especially since that range is a monocular estimate that is
            # routinely 10 percent long.
            # Commit to the frozen line of sight only while actually blind. A
            # fresh bearing this close in is the most valuable measurement of
            # the whole engagement -- one degree of correction at 5 m is 9 cm of
            # miss distance -- and throwing it away in the name of "committing"
            # is how a run ends at 1.2 m instead of 0.4 m.
            blind = self._measurement_age() > cfg.terminal_fresh_s
            s_cmd = (self._terminal_los or s) if blind else s
            v_close = cfg.terminal_speed
            # Scale the PN *correction*, not the across-LOS velocity the
            # aircraft already has. Multiplying the whole term -- which contains
            # v_own_perp -- feeds the plant's own output back into its reference
            # with gain > 1, so the lateral velocity grows every tick until it
            # hits its saturation, whatever the geometry is asking for.
            lat_t = _add(v_own_perp,
                         _lead(a_perp, cfg.lookahead_s * cfg.terminal_nav_scale,
                               cfg.vertical_lead_scale))
            lat_mag_t = _norm(lat_t)
            if lat_mag_t > cfg.max_lateral_speed:
                lat_t = _scale(lat_t, cfg.max_lateral_speed / lat_mag_t)
            desired = _add(_scale(s_cmd, v_close), lat_t)
            st.note = "terminal (blind commit)" if blind else "terminal (live)"
        else:
            v_close = self.approach_speed * self._speed_gate(bore)
            desired = _add(_scale(s, v_close), lat)
            st.note = f"PN N={cfg.nav_gain} Vc={vc:.1f}"

        # Saturate as a vector: scaling all three components together preserves
        # the ratio of closing to across-LOS speed, which IS the commanded
        # intercept geometry. Clamping components independently would quietly
        # fly a different course than the one that was computed.
        vx, vy, vz = desired
        h = math.hypot(vx, vy)
        if h > self.limits.max_speed_xy:
            k = self.limits.max_speed_xy / h
            vx, vy, vz = vx * k, vy * k, vz * k
        # ...and the same reasoning applies to the vertical ceiling, which used
        # to be clamped component-wise two lines below a comment explaining why
        # that is wrong. A bare clamp on vz flattens the commanded course
        # whenever the climb saturates -- it does not slow the aircraft down, it
        # points it somewhere else, which is precisely the failure the vector
        # saturation above exists to avoid.
        if abs(vz) > self.limits.max_speed_z:
            k = self.limits.max_speed_z / abs(vz)
            vx, vy, vz = vx * k, vy * k, vz * k
        st.closing_speed = round(_dot((vx, vy, vz), s), 3)

        return BodyCommand(vx, vy, vz, self._yaw_rate(chaser_yaw, st),
                           source=f"pn:{self.mode.lower()}", frame="world")

    def _closing_speed(self, v_own, s) -> float:
        """Best available closing speed for the PN gain, metres per second.

        Preferred source is the range filter's own rate -- that is the real
        closure, target motion included. It needs a few samples to mean
        anything, so until then (and whenever it reads as opening, which happens
        while a faster target pulls away) the chaser's speed along the line of
        sight stands in. Floored well above zero: ``Vc`` multiplies the steering
        gain, so letting it reach zero would switch the seeker off exactly when
        the geometry is worst.
        """
        vc = None
        if self.rng.value is not None and self.rng.t_last is not None:
            vc = -self.rng.rate
        if vc is None or vc < 1.0:
            vc = _dot(v_own, s)
        return max(1.5, min(40.0, vc))

    def _speed_gate(self, boresight_rad: float) -> float:
        """Closing-speed scale from how far off-boresight the target is.

        Full speed at a target the camera is looking straight at; scaled toward
        ``capture_speed_scale`` as it moves out toward the edge of frame. This is
        not politeness, it is the only thing stopping the aircraft from
        out-flying its own fixed camera: translate hard toward something 40
        degrees off the nose and the bearing rate it induces is larger than the
        yaw rate that could follow it, so the target leaves the frame *because*
        of the closure.
        """
        cfg = self.cfg
        if cfg.omnidirectional:
            # Nothing to fall out of. See GuidanceConfig.omnidirectional.
            return 1.0
        soft = math.radians(cfg.boresight_soft_deg)
        hard = math.radians(cfg.boresight_hard_deg)
        if boresight_rad <= soft:
            return 1.0
        if boresight_rad >= hard:
            return cfg.capture_speed_scale
        u = (boresight_rad - soft) / max(1e-6, hard - soft)
        return 1.0 + u * (cfg.capture_speed_scale - 1.0)

    def _yaw_rate(self, chaser_yaw: float, st: GuidanceState) -> float:
        """Keep the boresight on the target: P on azimuth, plus LOS-rate feed-forward."""
        cfg = self.cfg
        s = self._los_now()
        if s is None:
            return 0.0
        want = math.atan2(s[1], s[0])
        err = wrap_pi(want - chaser_yaw)
        if abs(err) < cfg.yaw_deadband_rad:
            err = 0.0
        # Azimuth rate of the LOS in the horizontal plane: d/dt atan2(sy, sx).
        h2 = s[0] * s[0] + s[1] * s[1]
        lead = 0.0
        if h2 > 1e-9:
            lead = (s[0] * self.los.ds[1] - s[1] * self.los.ds[0]) / h2
        rate = cfg.kp_yaw * err + cfg.yaw_lead * lead
        return max(-self.limits.max_yaw_rate, min(self.limits.max_yaw_rate, rate))

    def _reacquire_command(self, chaser_yaw: float, chaser_vel,
                           st: GuidanceState) -> BodyCommand:
        """Coast on the predicted LOS, then widen into a sweep.

        The order matters. Most lost frames are the detector blinking, not the
        target leaving, and the cheapest way to get those back is to keep flying
        the course that was working while the LOS rotates on at its last known
        rate. Only once the loss has outlasted that does it become worth giving
        up the geometry to go looking.
        """
        cfg = self.cfg
        if self.lost_for_s <= cfg.reacquire_hold_s and self.los.s is not None:
            st.note = "coasting on predicted LOS"
            return self._closure_command(chaser_yaw, chaser_vel, st)

        if cfg.omnidirectional:
            # The target is somewhere in a sphere already fully under
            # observation. Stop and watch: holding still restores the static
            # background the long-range detector needs, and it stops adding the
            # aircraft's own translation to a bearing error it is trying to
            # resolve.
            st.note = "holding, watching all quarters"
            return BodyCommand(0.0, 0.0, 0.0, 0.0, source="reacquire:hold",
                               frame="world")

        st.note = "re-acquisition sweep"
        s = self._los_now()
        if s is not None:
            want = math.atan2(s[1], s[0])
            err = wrap_pi(want - chaser_yaw)
            self._search_dir = 1.0 if err >= 0.0 else -1.0
        # Hold position while sweeping: flying on through a loss adds the
        # aircraft's own translation to the bearing error we are trying to undo.
        # The vertical sweep is halved here -- a target that was in frame a
        # moment ago is very unlikely to be an elevation problem, so most of the
        # search effort belongs on the axis it probably left by.
        return BodyCommand(0.0, 0.0, 0.5 * self._search_climb(),
                           self._search_dir * cfg.search_yaw_rate,
                           source="reacquire", frame="world")

    def _search_command(self) -> BodyCommand:
        """Step-and-stare: hold still for a look, slew, hold again.

        The first look starts at the current heading and costs nothing, which
        matters because a pursuit usually begins with the target already in
        frame -- turning away from it is the most expensive thing a search can
        do. See ``search_dwell_s``.
        """
        cfg = self.cfg
        if cfg.omnidirectional:
            # Not a degenerate search -- a different one. Every direction is
            # already in a frame, so the aircraft holds its station and its
            # heading, which is also the only state in which the ring's frame
            # differencing has a background worth subtracting.
            return BodyCommand(0.0, 0.0, self._station_climb(), 0.0,
                               source="search:watch", frame="world")
        vz = self._search_climb()
        if self._search_hold_s < self._dwell_target():
            return BodyCommand(0.0, 0.0, vz, 0.0, source="search:look",
                               frame="world")
        return BodyCommand(0.0, 0.0, vz,
                           self._search_dir * cfg.search_yaw_rate,
                           source="search:slew", frame="world")

    def _station_climb(self) -> float:
        """Hold the altitude the watch started at.

        A position loop rather than zero, because the aircraft arrives at SEARCH
        with whatever vertical velocity the last phase left it, and an
        interceptor that drifts down through its own overwatch altitude between
        engagements is one that loses line of sight to the far side of the city.
        """
        if self._search_alt0 is None:
            self._search_alt0 = self._chaser_z
        rate = self.cfg.search_alt_gain * (self._search_alt0 - self._chaser_z)
        lim = self.limits.max_speed_z
        return max(-lim, min(lim, rate))

    def _slew_target(self) -> float:
        """How far this slew travels: one step further out than the last."""
        return self.cfg.search_step_rad * (self._search_looks + 1)

    def _dwell_target(self) -> float:
        """How long the current look lasts -- longer for the very first one."""
        cfg = self.cfg
        return (cfg.search_first_dwell_s if self._search_looks == 0
                else cfg.search_dwell_s)

    def _advance_search_phase(self, dt: float) -> None:
        """Run the dwell/slew clock. Called once per tick while searching."""
        cfg = self.cfg
        if cfg.omnidirectional:
            return
        if self._search_hold_s < self._dwell_target():
            self._search_hold_s += dt
            return
        self._search_slewed += abs(cfg.search_yaw_rate) * dt
        if self._search_slewed >= self._slew_target():
            self._search_hold_s = 0.0
            self._search_slewed = 0.0
            self._search_looks += 1
            # Alternate sides so the looks walk outward from the starting
            # heading: 0, +s, -s, +2s, -2s, ... Each slew is one step longer
            # than the last and in the opposite direction, which is what turns
            # a growing relative slew into a symmetric absolute pattern.
            self._search_dir = -self._search_dir

    def _search_climb(self) -> float:
        """Climb rate that walks the search altitude around its datum.

        Returns a *rate*, but it is the output of a position loop: the datum is
        the altitude the search started at, and the aircraft is driven to a
        slowly oscillating offset above it. That is what keeps the excursion
        bounded no matter how long the search runs -- see
        ``search_alt_offset_m``.
        """
        cfg = self.cfg
        if self._search_alt0 is None:
            self._search_alt0 = self._chaser_z
        amp = cfg.search_alt_amplitude_m
        period = max(1e-3, cfg.search_climb_period_s)
        want = (self._search_alt0 + cfg.search_alt_offset_m
                + amp * math.sin(2.0 * math.pi * self.t / period))
        rate = cfg.search_alt_gain * (want - self._chaser_z)
        lim = self.limits.max_speed_z
        return max(-lim, min(lim, rate))
