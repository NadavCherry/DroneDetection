"""What the fleeing drone does.

The evader exists to make the intercept *hard*, so these policies are written as
a difficulty ladder rather than as one plausible behaviour. Every one of them is
a different way to break a guidance law, and a pursuit that only ever works
against a target flying in a straight line has not been tested:

``straight``
    Constant velocity. The baseline: any law that cannot intercept this is
    broken, and the closure rate here is exactly the speed difference.
``flee``
    Runs directly away from the chaser. The pure tail chase, and the case where
    a speed advantage is the *only* thing that closes the range.
``weave``
    Flees while sliding side to side. Aimed squarely at the seeker: a sinusoidal
    line of sight is the classic way to make a proportional-navigation loop
    chase its own lag, and it is the case where LOS-rate filtering earns its
    keep or loses the target.
``break_turn``
    Flies straight until the chaser is inside a trigger range, then pulls the
    hardest turn it can. This is the manoeuvre that beats *pure pursuit*
    outright -- the pursuer aimed at where the target was and has to fly the
    whole turn radius to get back -- so it is the sharpest test of whether the
    lead is real.
``jink``
    Random direction changes at random intervals. Unpredictable by construction,
    so no amount of target-motion modelling helps; what survives this is a law
    that reacts fast, not one that extrapolates well.
``orbit``
    Circles a fixed point at constant speed: a sustained, constant-rate turn,
    which is the steady-state case a break turn only visits briefly.
``climb_flee``
    Flees while climbing hard, pushing the target toward the top of a frame
    whose principal point already sits high -- the axis with the least room.
``sweep``
    Back and forth along a fixed lateral axis, across the chaser's view. The
    plainest possible *cross-range* motion and the natural first rung of a
    difficulty ladder: unmistakable on video, and the exact opposite of the
    radial flight the other policies mostly produce.
``barrel``
    Flees along a helix -- lateral and vertical oscillation at once. The first
    policy that makes the seeker work in both image axes simultaneously.
``evasive``
    Actively tries to escape rather than merely to leave. It flies *across* the
    line of sight, which is the direction that maximises the rate the seeker has
    to null, reverses that direction on a timer so no lead can settle, jinks in
    altitude on a different period so the two never phase-lock, and pulls
    everything its airframe will give. This is the hard one.

## A note on radial flight

Most of the simple policies run almost straight away from the chaser, and that
is deceptive twice over. On video the target merely swells, so the motion is
invisible. Dynamically it is the *easiest* case a seeker ever gets: a target
receding along the line of sight produces almost no line-of-sight rate, which is
the only quantity proportional navigation has to work with. ``sweep``,
``barrel`` and ``evasive`` exist because a pursuit suite made of radial flight
flatters the algorithm on both counts.

Every policy returns a **desired world velocity**; the evader's own
:class:`~pursuit.dynamics.Airframe` then applies its speed, acceleration and
turn limits, so no policy can ask for a manoeuvre the aircraft could not fly.

Randomness is drawn from a seeded ``random.Random`` held per policy, so a
scenario named ``jink`` with seed 7 is the same flight every time it is run --
without which a "fix" cannot be told from a different roll of the dice.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .dynamics import Airframe, BodyCommand, Limits
from .geometry import world_to_body


def _unit_xy(x: float, y: float) -> Tuple[float, float]:
    n = math.hypot(x, y)
    return (1.0, 0.0) if n < 1e-9 else (x / n, y / n)


@dataclass
class EvaderConfig:
    """Tuning shared by the policies.

    Attributes:
        speed: Cruise speed the evader tries to hold (m/s).
        weave_period_s: Period of the lateral oscillation in ``weave``.
        weave_amplitude: Lateral speed of the weave as a fraction of ``speed``.
        break_trigger_m: Range at which ``break_turn`` commits to its turn.
        break_hold_s: How long the break turn is held before resuming.
        jink_min_s / jink_max_s: Bounds on the interval between ``jink`` changes.
        orbit_radius_m: Radius of the ``orbit`` circle.
        climb_mps: Climb rate used by ``climb_flee``.
        sweep_half_length_m: Half-length of the ``sweep`` axis, so the target
            runs from -L to +L and back across the chaser's view.
        barrel_period_s: Period of the ``barrel`` helix.
        barrel_climb_mps: Vertical amplitude of that helix.
        evasive_reverse_s: How often ``evasive`` flips which way it crosses the
            line of sight. Short enough that no lead can settle, long enough
            that the airframe can actually complete the reversal.
        evasive_vertical_s: Period of its altitude jink -- deliberately not a
            multiple of the reversal period, so the two never phase-lock into
            something predictable.
        altitude_band: ``(low, high)`` above ground the evader keeps itself in,
            so a policy cannot win by flying into the floor or out of the world.
        arena_radius_m: Horizontal radius the evader stays inside, turning back
            when it reaches the edge. Without it ``flee`` is not an evasion at
            all -- it is a straight line to infinity, which either ends the run
            on a boundary or, worse, quietly measures the speed difference and
            calls it an intercept. A bounded arena is also the realistic case:
            the thing being defended is somewhere, and the evader has a reason
            to stay near it.
        arena_margin_m: Distance from the edge at which the turn-back begins, so
            the evader curves away rather than bouncing off a wall.
    """

    speed: float = 9.0
    weave_period_s: float = 3.0
    weave_amplitude: float = 0.8
    break_trigger_m: float = 22.0
    break_hold_s: float = 2.5
    jink_min_s: float = 0.8
    jink_max_s: float = 2.0
    orbit_radius_m: float = 25.0
    climb_mps: float = 3.5
    sweep_half_length_m: float = 20.0
    barrel_period_s: float = 4.0
    barrel_climb_mps: float = 5.0
    evasive_reverse_s: float = 2.2
    evasive_vertical_s: float = 4.7
    altitude_band: Tuple[float, float] = (8.0, 45.0)
    arena_radius_m: float = 90.0
    arena_margin_m: float = 25.0


POLICIES = ("straight", "flee", "weave", "break_turn", "jink", "orbit",
            "climb_flee", "sweep", "barrel", "evasive")

LADDER = ("sweep", "weave", "barrel", "orbit", "break_turn", "jink", "evasive")
"""The policies in rough order of how hard they are to intercept, for building
a difficulty ramp. ``sweep`` is pure predictable cross-range motion; ``evasive``
changes direction, altitude and sense faster than a lead can be established."""


class Evader:
    """The fleeing drone's brain: relative geometry in, world velocity out.

    The evader is allowed to know exactly where the chaser is. That is not
    cheating in its favour so much as removing a second perception problem from
    an experiment about the first one -- and it makes the evader *harder*, which
    is the direction an evaluation should err in.
    """

    def __init__(self, policy: str, cfg: Optional[EvaderConfig] = None,
                 seed: int = 0, ground_z: float = 0.0,
                 heading0: Optional[float] = None,
                 centre_xy: Optional[Sequence[float]] = None) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown evader policy {policy!r}; choose from {POLICIES}")
        self.policy = policy
        self.cfg = cfg or EvaderConfig()
        self.rng = random.Random(seed)
        self.ground_z = float(ground_z)
        self.centre_xy = tuple(centre_xy) if centre_xy is not None else (0.0, 0.0)

        self.heading = (float(heading0) if heading0 is not None
                        else self.rng.uniform(-math.pi, math.pi))
        self._t_next_jink = 0.0
        self._sweep_origin: Optional[Tuple[float, float]] = None
        self._sweep_axis: Tuple[float, float] = (0.0, 1.0)
        self._sweep_dir = 1.0
        self._evasive_sign = 1.0 if self.rng.random() < 0.5 else -1.0
        self._break_started: Optional[float] = None
        self._break_sign = 1.0 if self.rng.random() < 0.5 else -1.0
        self._orbit_sign = 1.0 if self.rng.random() < 0.5 else -1.0

        # -- ingress ---------------------------------------------------------
        # Before it knows it has been seen, an intruder is not evading; it is
        # just flying somewhere. Modelling that first leg matters because it is
        # the only part of the engagement that exercises *acquisition*: a target
        # placed in frame at t=0 has already skipped the hardest thing the
        # system does.
        self._aim: Optional[Tuple[float, float, float]] = None
        self._transit_speed = 0.0
        self._commit = False
        self._evade = 0.0
        self.revealed = True
        self.revealed_at: Optional[float] = None

    def arm_ingress(self, aim_xyz: Sequence[float], speed: float,
                    commit: bool = False, evade: float = 0.0) -> None:
        """Fly a straight course to ``aim_xyz`` until :meth:`reveal` is called.

        Args:
            aim_xyz: Where it is going.
            speed: Speed on that leg.
            commit: Keep going *after* being seen. The default -- breaking off
                into an evasion policy the moment the interceptor locks on --
                describes a reconnaissance drone, not a strike. A one-way attack
                aircraft has a target and no interest in the thing chasing it,
                and the difference is the whole point of a defence scenario:
                only a committed intruder puts a clock on the engagement, and
                only a committed intruder can *win* by arriving.
            evade: How hard a committed intruder jinks about its course, as a
                fraction of its speed. Zero is a straight line, which is both
                the easiest thing to intercept and the least realistic thing to
                fly through defended airspace; 0.5 is a target that is genuinely
                hard to lead and still arrives.
        """
        self._aim = tuple(float(v) for v in aim_xyz)
        self._transit_speed = float(speed)
        self._commit = bool(commit)
        self._evade = float(evade)
        self.revealed = False
        self.revealed_at = None

    def reveal(self, t: float) -> None:
        """The intruder realises it is being chased and starts evading.

        Idempotent, and it re-seeds the policy's heading from the course the
        transit leg was actually flying. Without that, a policy like
        ``straight`` or ``break_turn`` would snap to the heading it was given at
        construction time -- a discontinuity in the middle of the engagement
        that the guidance law would have to absorb for no physical reason.
        """
        if self.revealed:
            return
        self.revealed = True
        self.revealed_at = float(t)
        # heading already tracks the transit course (updated in desired_velocity)
        self._t_next_jink = float(t)
        self._sweep_origin = None
        self._break_started = None

    # -- policy -------------------------------------------------------------

    def desired_velocity(self, t: float, own_xyz, chaser_xyz) -> Tuple[float, float, float]:
        """World-frame velocity this policy wants at time ``t``."""
        cfg = self.cfg
        if self._aim is not None and (not self.revealed or self._commit):
            dx = self._aim[0] - float(own_xyz[0])
            dy = self._aim[1] - float(own_xyz[1])
            dz = self._aim[2] - float(own_xyz[2])
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            if n <= 1e-6:
                # Arrived without ever being seen. Carry straight on rather than
                # stopping, so the pass continues out the far side of the frame.
                hx, hy = math.cos(self.heading), math.sin(self.heading)
                return (hx * self._transit_speed, hy * self._transit_speed, 0.0)
            self.heading = math.atan2(dy, dx)
            s = self._transit_speed / n
            vx, vy, vz = dx * s, dy * s, dz * s
            if self.revealed and self._evade > 0.0:
                # A strike drone that knows it is being intercepted does not
                # abandon its target; it makes itself hard to lead on the way
                # in. Superimposed on the course rather than replacing it, so
                # the aim point still attracts and the clock keeps running --
                # which is what stops this becoming an ordinary evasion suite
                # with a building drawn on it.
                t0 = self.revealed_at or 0.0
                ph = 2.0 * math.pi * (t - t0) / max(0.4, cfg.weave_period_s)
                ax, ay = _unit_xy(dx, dy)
                amp = self._evade * self._transit_speed * math.sin(ph)
                vx += -ay * amp
                vy += ax * amp
                vz += 0.45 * self._evade * self._transit_speed * math.sin(
                    2.0 * math.pi * (t - t0) / max(0.4, cfg.evasive_vertical_s))
                vz = self._altitude_guard(float(own_xyz[2]), vz)
            return (vx, vy, vz)
        dx = float(own_xyz[0]) - float(chaser_xyz[0])
        dy = float(own_xyz[1]) - float(chaser_xyz[1])
        rng = math.dist(tuple(float(v) for v in own_xyz),
                        tuple(float(v) for v in chaser_xyz))
        away = _unit_xy(dx, dy)

        vz = 0.0
        if self.policy == "straight":
            hx, hy = math.cos(self.heading), math.sin(self.heading)
        elif self.policy == "flee":
            hx, hy = away
        elif self.policy == "weave":
            hx, hy = away
            phase = 2.0 * math.pi * t / max(0.2, cfg.weave_period_s)
            # Lateral is perpendicular to the run-away direction, so the weave
            # is across the line of sight -- which is the axis the seeker sees.
            lx, ly = -hy, hx
            amp = cfg.weave_amplitude * math.sin(phase)
            hx, hy = _unit_xy(hx + lx * amp, hy + ly * amp)
        elif self.policy == "break_turn":
            if self._break_started is None and rng <= cfg.break_trigger_m:
                self._break_started = t
            if (self._break_started is not None
                    and t - self._break_started <= cfg.break_hold_s):
                # Turn across the line of sight, the direction that costs a
                # pursuer the most: it has to reverse its own lateral velocity.
                ax, ay = away
                hx, hy = (-ay * self._break_sign, ax * self._break_sign)
            else:
                hx, hy = math.cos(self.heading), math.sin(self.heading)
        elif self.policy == "jink":
            if t >= self._t_next_jink:
                self.heading += self.rng.uniform(-2.2, 2.2)
                self._t_next_jink = t + self.rng.uniform(cfg.jink_min_s, cfg.jink_max_s)
            hx, hy = math.cos(self.heading), math.sin(self.heading)
        elif self.policy == "orbit":
            rx = float(own_xyz[0]) - self.centre_xy[0]
            ry = float(own_xyz[1]) - self.centre_xy[1]
            rr = math.hypot(rx, ry)
            if rr < 1e-6:
                hx, hy = math.cos(self.heading), math.sin(self.heading)
            else:
                tx, ty = -ry / rr, rx / rr
                # Steer back toward the nominal radius so the circle is stable.
                err = (cfg.orbit_radius_m - rr) / max(1.0, cfg.orbit_radius_m)
                hx, hy = _unit_xy(tx * self._orbit_sign + (rx / rr) * -err * 2.0,
                                  ty * self._orbit_sign + (ry / rr) * -err * 2.0)
        elif self.policy == "sweep":
            # A straight lateral axis, laid across the line of sight at the
            # moment the run starts, flown -L to +L and back. Pure cross-range
            # motion: the plainest thing a seeker can be asked to follow, and
            # the plainest thing to *see* on video.
            if self._sweep_origin is None:
                self._sweep_origin = (float(own_xyz[0]), float(own_xyz[1]))
                ax, ay = away
                self._sweep_axis = (-ay, ax)     # perpendicular to the LOS
            ox_, oy_ = self._sweep_origin
            axs, ays = self._sweep_axis
            along = ((float(own_xyz[0]) - ox_) * axs
                     + (float(own_xyz[1]) - oy_) * ays)
            if along >= cfg.sweep_half_length_m:
                self._sweep_dir = -1.0
            elif along <= -cfg.sweep_half_length_m:
                self._sweep_dir = 1.0
            hx, hy = axs * self._sweep_dir, ays * self._sweep_dir
        elif self.policy == "barrel":
            # A helix: flee, but corkscrew while doing it, so the seeker has to
            # work both image axes at once instead of one at a time.
            ax, ay = away
            lx, ly = -ay, ax
            phase = 2.0 * math.pi * t / max(0.2, cfg.barrel_period_s)
            hx, hy = _unit_xy(ax + lx * math.sin(phase), ay + ly * math.sin(phase))
            vz = cfg.barrel_climb_mps * math.cos(phase)
        elif self.policy == "evasive":
            # The one that actually tries to escape. Across the line of sight is
            # the direction that maximises the rate the seeker must null, so it
            # flies that -- and reverses on a timer, because a constant crossing
            # is just an orbit and a lead settles on it within a second or two.
            ax, ay = away
            if int(t / max(0.2, cfg.evasive_reverse_s)) % 2 == 0:
                self._evasive_sign = 1.0
            else:
                self._evasive_sign = -1.0
            cross = (-ay * self._evasive_sign, ax * self._evasive_sign)
            # Mostly across, a little away: pure crossing lets the chaser close
            # for free, pure fleeing hands it an easy tail chase.
            hx, hy = _unit_xy(cross[0] * 0.85 + ax * 0.35,
                              cross[1] * 0.85 + ay * 0.35)
            # Altitude jinks on a period that does not divide the reversal one,
            # so the two never settle into a pattern worth predicting.
            vz = cfg.climb_mps * 1.2 * math.sin(
                2.0 * math.pi * t / max(0.2, cfg.evasive_vertical_s))
        else:  # climb_flee
            hx, hy = away
            vz = cfg.climb_mps

        hx, hy = self._arena_guard(own_xyz, hx, hy)
        vz = self._altitude_guard(float(own_xyz[2]), vz)
        return (hx * cfg.speed, hy * cfg.speed, vz)

    def _arena_guard(self, own_xyz, hx: float, hy: float) -> Tuple[float, float]:
        """Blend the policy's heading toward the arena centre near the edge.

        A blend rather than a hard reflection: a policy that gets its heading
        replaced outright at the boundary produces a corner in the flight path
        that is not something an aircraft does, and the resulting LOS-rate spike
        would flatter or punish the seeker for a modelling artefact.
        """
        cfg = self.cfg
        rx = float(own_xyz[0]) - self.centre_xy[0]
        ry = float(own_xyz[1]) - self.centre_xy[1]
        rr = math.hypot(rx, ry)
        edge = cfg.arena_radius_m
        if rr <= edge - cfg.arena_margin_m or rr < 1e-6:
            return hx, hy
        u = min(1.0, (rr - (edge - cfg.arena_margin_m)) / max(1e-6, cfg.arena_margin_m))
        inward = (-rx / rr, -ry / rr)
        return _unit_xy(hx * (1.0 - u) + inward[0] * u,
                        hy * (1.0 - u) + inward[1] * u)

    def _altitude_guard(self, z: float, vz: float) -> float:
        """Hold the evader inside its altitude band, overriding the policy.

        An earlier version *added* a bounded correction to whatever the policy
        asked for, and a bounded correction loses to an unbounded policy: at the
        ceiling, ``climb_flee``'s 3.5 m/s climb against a 3 m/s push-back is a
        net 0.5 m/s, and the target simply leaves upward for the rest of the
        episode -- 68 m against a 45 m ceiling, out of the camera's reach the
        whole way. That is not an evasion the chaser lost, it is a scenario that
        was never winnable, and it hides whatever the guidance law was actually
        doing.

        So the band wins outright: past the edge the vertical command is
        *replaced* by a return, and inside it the policy is untouched.
        """
        low = self.ground_z + self.cfg.altitude_band[0]
        high = self.ground_z + self.cfg.altitude_band[1]
        if z < low:
            return max(vz, min(3.0, low - z))
        if z > high:
            return min(vz, -min(3.0, z - high))
        return vz

    # -- driving the airframe ----------------------------------------------

    def command(self, t: float, frame: Airframe, chaser_xyz) -> BodyCommand:
        """The body command to hand :meth:`Airframe.step`.

        The evader's nose is put on its own velocity vector, which is both what
        an aircraft does and what makes it hardest to see: presented head-on or
        tail-on it is a handful of pixels, and the chaser never gets the
        broadside view that makes a quadrotor obvious.
        """
        wx, wy, wz = self.desired_velocity(t, frame.xyz, chaser_xyz)
        want_yaw = math.atan2(wy, wx) if math.hypot(wx, wy) > 1e-6 else frame.yaw
        err = (want_yaw - frame.yaw + math.pi) % (2.0 * math.pi) - math.pi
        return BodyCommand(wx, wy, wz, 2.0 * err,
                           source=f"evader:{self.policy}", frame="world")


def make_evader(policy: str, seed: int, ground_z: float,
                cfg: Optional[EvaderConfig] = None,
                heading0: Optional[float] = None,
                centre_xy: Optional[Sequence[float]] = None) -> Evader:
    return Evader(policy, cfg=cfg, seed=seed, ground_z=ground_z,
                  heading0=heading0, centre_xy=centre_xy)


def evader_limits(cfg: Optional[EvaderConfig] = None) -> Limits:
    """Limits sized to the evader's cruise speed, with a real turn capability.

    The vertical ceiling has to cover the most demanding policy, not the mildest.
    Sizing it from ``climb_mps`` alone silently clipped ``barrel`` and
    ``evasive`` to 3.5 m/s, which turned their altitude changes into a 4 m
    wobble -- present in the telemetry, invisible in the engagement. A target
    asked to fly in three dimensions needs the authority to do it.
    """
    c = cfg or EvaderConfig()
    vz = max(3.0, c.climb_mps, c.barrel_climb_mps, 1.2 * c.climb_mps)
    return Limits(max_speed_xy=c.speed, max_speed_z=vz,
                  max_accel_xy=10.0, max_accel_z=7.0,
                  max_yaw_rate=3.0, max_yaw_accel=12.0, min_agl=3.0,
                  max_agl=60.0)
