"""Flight dynamics for the two aircraft, owned by the brain rather than the sim.

The simulator's drones are kinematic -- they go exactly where their transform is
set -- so somebody has to decide what "exactly where" means from one tick to the
next, and it should not be the guidance law. If the controller's output went
straight to a pose, a step change in the commanded velocity would teleport the
aircraft and every intercept would look effortless: no lag, no overshoot, no
minimum turn radius, none of the things that make a real pursuit hard.

So the brain integrates a small rigid-body model instead. The command is a
*desired* body velocity; this module turns it into a pose by rate-limiting the
acceleration toward it, saturating speed, and slewing yaw under its own rate and
acceleration limits. The result is an aircraft that leans into a turn, takes
time to reverse, and cannot instantly point its camera somewhere else -- and a
guidance law that works against it is one that has been tested against something.

Defaults are quadrotor-shaped: a racing-class airframe pulls well over 1 g
laterally and holds 15-20 m/s, and the *chaser is faster than the evader* --
without a speed advantage a tail chase never closes, which is a fact about
kinematics, not about the algorithm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence, Tuple

from .geometry import body_to_world, wrap_pi


@dataclass(frozen=True)
class Limits:
    """What the airframe can do.

    Attributes:
        max_speed_xy: Horizontal speed ceiling (m/s).
        max_speed_z: Climb/descend speed ceiling (m/s).
        max_accel_xy: Horizontal acceleration ceiling (m/s^2). 12 is ~1.2 g,
            an unremarkable number for a small quadrotor and the thing that sets
            the minimum turn radius: ``v^2 / a``, so 12 m/s at 12 m/s^2 turns in
            12 m and no guidance law can do better.
        max_accel_z: Vertical acceleration ceiling (m/s^2).
        max_yaw_rate: Yaw rate ceiling (rad/s).
        max_yaw_accel: Yaw acceleration ceiling (rad/s^2) -- the camera cannot
            snap to a new heading, which is what makes a target that leaves the
            frame expensive to get back.
        min_agl: Floor height above the scene ground (m); the aircraft is
            clamped rather than allowed to fly into terrain.
        max_agl: Ceiling above the scene ground (m). The evader has always had an
            altitude band; the chaser having only a floor was a real defect, and
            a slow one to notice because it needs a *long* episode to bite. The
            search climbs on purpose (this camera is blind above 15.5 degrees, so
            altitude is the only way to look up), and an unbounded climb over a
            45-second search puts the aircraft 135 m up, above everything,
            looking down at terrain -- where it promptly locked onto ground
            clutter and stayed there. Measured: three of three ``climb_flee``
            scenarios, tracking something 92 percent of the time that was the
            drone 1 percent of the time.
    """

    max_speed_xy: float = 14.0
    max_speed_z: float = 5.0
    max_accel_xy: float = 12.0
    max_accel_z: float = 6.0
    max_yaw_rate: float = 2.5
    max_yaw_accel: float = 10.0
    min_agl: float = 2.0
    max_agl: float = 60.0


@dataclass
class BodyCommand:
    """A desired velocity plus a yaw rate.

    ``frame="body"`` (REP-103): ``vx`` forward, ``vy`` left, ``vz`` up.
    ``frame="world"``: the same three numbers along world x/y/z.
    ``yaw_rate`` is CCW about world +z either way.

    The frame is explicit because getting it wrong is invisible and expensive.
    Guidance reasons in the world frame -- a line of sight rotates in the world,
    not in a frame bolted to an aircraft that is itself rotating -- and an
    earlier version converted that answer to body coordinates using the yaw at
    the *start* of the tick, which the airframe then resolved through the yaw at
    the *end*. At a saturated 2.5 rad/s that is seven degrees of silent error in
    the commanded direction every tick, applied in whichever direction the
    aircraft happened to be turning: a yaw oscillation and a velocity
    oscillation feeding each other, which showed up as a chaser circling its
    target at constant range and looked for all the world like a bad PN gain.

    A quadrotor's translation and heading are independent anyway, so keeping the
    velocity in the world frame and letting yaw serve only the camera is both
    simpler and what the vehicle actually does.
    """

    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0
    source: str = ""
    frame: str = "body"

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.vx, self.vy, self.vz, self.yaw_rate)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else float(v))


def _limit_norm(x: float, y: float, lim: float) -> Tuple[float, float]:
    """Scale ``(x, y)`` down to length ``lim``, preserving direction.

    Component-wise clamping would be wrong here and wrong in a way that matters:
    it lets the aircraft accelerate ``sqrt(2)`` times harder on a diagonal than
    on an axis, and it *rotates* the commanded direction whenever one component
    saturates -- so a hard turn would come out pointing somewhere other than
    where the guidance law asked.
    """
    n = math.hypot(x, y)
    if n <= lim or n <= 1e-12:
        return x, y
    s = lim / n
    return x * s, y * s


@dataclass
class Airframe:
    """A level, yaw-only aircraft integrated by the brain.

    State is world-frame position and velocity plus a heading and its rate.
    Attitude beyond yaw is deliberately absent: the camera is bolted to the body,
    so a chaser that pitched to centre a target would roll the horizon through
    frame and hand the detector a moving background for no gain -- the same
    reasoning that keeps the simulator's aircraft level.
    """

    xyz: Tuple[float, float, float]
    yaw: float = 0.0
    limits: Limits = field(default_factory=Limits)
    ground_z: float = 0.0
    vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_rate: float = 0.0

    def __post_init__(self) -> None:
        self.xyz = tuple(float(v) for v in self.xyz)
        self.vel = tuple(float(v) for v in self.vel)
        self.yaw = float(self.yaw)

    # -- integration --------------------------------------------------------

    def step(self, cmd: BodyCommand, dt: float) -> None:
        """Advance one control tick under ``cmd``.

        A body-frame command is resolved through the heading the aircraft had
        when the command was computed, *before* this tick's rotation is applied.
        Using the post-rotation heading instead silently rotates the commanded
        velocity by ``yaw_rate * dt`` -- see :class:`BodyCommand`.
        """
        dt = float(dt)
        if dt <= 0.0:
            return
        lim = self.limits

        # -- velocity, in the frame the command was written in ---------------
        if cmd.frame == "world":
            wx, wy, wz = cmd.vx, cmd.vy, cmd.vz
        else:
            wx, wy, wz = body_to_world(self.yaw, cmd.vx, cmd.vy, cmd.vz)

        # -- heading -------------------------------------------------------
        want_rate = _clamp(cmd.yaw_rate, -lim.max_yaw_rate, lim.max_yaw_rate)
        d_rate = _clamp(want_rate - self.yaw_rate,
                        -lim.max_yaw_accel * dt, lim.max_yaw_accel * dt)
        self.yaw_rate += d_rate
        self.yaw = wrap_pi(self.yaw + self.yaw_rate * dt)
        wx, wy = _limit_norm(wx, wy, lim.max_speed_xy)
        wz = _clamp(wz, -lim.max_speed_z, lim.max_speed_z)

        dvx, dvy = _limit_norm(wx - self.vel[0], wy - self.vel[1],
                               lim.max_accel_xy * dt)
        dvz = _clamp(wz - self.vel[2], -lim.max_accel_z * dt, lim.max_accel_z * dt)
        vx, vy, vz = self.vel[0] + dvx, self.vel[1] + dvy, self.vel[2] + dvz
        vx, vy = _limit_norm(vx, vy, lim.max_speed_xy)
        vz = _clamp(vz, -lim.max_speed_z, lim.max_speed_z)

        # -- position ------------------------------------------------------
        x = self.xyz[0] + vx * dt
        y = self.xyz[1] + vy * dt
        z = self.xyz[2] + vz * dt
        floor = self.ground_z + lim.min_agl
        ceiling = self.ground_z + lim.max_agl
        if z < floor:
            z = floor
            vz = max(0.0, vz)
        elif z > ceiling:
            z = ceiling
            vz = min(0.0, vz)
        self.vel = (vx, vy, vz)
        self.xyz = (x, y, z)

    # -- queries ------------------------------------------------------------

    @property
    def speed(self) -> float:
        return math.sqrt(sum(v * v for v in self.vel))

    @property
    def speed_xy(self) -> float:
        return math.hypot(self.vel[0], self.vel[1])

    def pose(self) -> dict:
        """The dict the simulator's ``step`` RPC expects."""
        return {"xyz": [round(v, 5) for v in self.xyz], "yaw": round(self.yaw, 6)}

    def snapshot(self) -> dict:
        return {"xyz": [round(v, 4) for v in self.xyz], "yaw": round(self.yaw, 5),
                "vel": [round(v, 4) for v in self.vel],
                "speed": round(self.speed, 3), "yaw_rate": round(self.yaw_rate, 4)}


def chaser_limits(speed_advantage: float = 1.0, base: Optional[Limits] = None) -> Limits:
    """Chaser limits scaled to hold a speed advantage over the evader.

    A tail chase against an equally fast target does not converge -- the range
    stays wherever it started, forever, no matter how good the guidance is.
    Scaling here rather than hard-coding two constants keeps the advantage an
    explicit, sweepable experiment variable.
    """
    b = base or Limits()
    return replace(b,
                   max_speed_xy=b.max_speed_xy * speed_advantage,
                   max_speed_z=b.max_speed_z * speed_advantage,
                   max_accel_xy=b.max_accel_xy * speed_advantage,
                   max_accel_z=b.max_accel_z * speed_advantage)
