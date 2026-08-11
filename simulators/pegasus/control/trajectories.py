"""Where each drone is at time t.

Trajectories are plain functions of simulation time rather than controllers
chasing setpoints, because at this stage the point of the rig is to produce a
*known* relative geometry. If the observer is 30 m from a target that sweeps a
20 m line, then the target's angular size, its speed across the frame and the
range at every instant are all arithmetic -- which means a detection can be
scored against the range it happened at without anyone having to estimate the
range first.

Every trajectory returns ``(x, y, z)`` in world metres for a given ``t`` in
simulation seconds.
"""
from __future__ import annotations

import math


class Hover:
    """Stay put.

    The observer's trajectory for stage one: hold a fixed point at a fixed
    height and let the target move through the field of view.
    """

    def __init__(self, position):
        self.position = tuple(float(v) for v in position)

    def at(self, t: float) -> tuple:
        return self.position

    def describe(self) -> dict:
        return {"kind": "hover", "position": list(self.position)}


class LineSweep:
    """Slide back and forth between two points at constant speed.

    The default target trajectory: out along the line, back, repeat. Reversing
    rather than looping matters for a motion-based detector -- at each end the
    target's image velocity passes through zero, which is where a
    frame-differencing channel has the least to work with. A trajectory that
    only ever moves one way never exercises that case, and the first time it
    appears would be in the field.

    Args:
        start: World ``(x, y, z)`` of one end.
        end: World ``(x, y, z)`` of the other.
        speed: Metres per second along the line.
        dwell_s: Seconds to hold still at each end before reversing. Zero gives
            a triangle-wave position with a velocity discontinuity at the turn;
            a short dwell is both more like a real vehicle and kinder to any
            tracker's velocity estimate.
    """

    def __init__(self, start, end, speed: float = 3.0, dwell_s: float = 0.5):
        self.start = tuple(float(v) for v in start)
        self.end = tuple(float(v) for v in end)
        self.speed = float(speed)
        self.dwell_s = float(dwell_s)

        self._delta = tuple(e - s for s, e in zip(self.start, self.end))
        self.length = math.sqrt(sum(d * d for d in self._delta))
        if self.length == 0.0:
            raise ValueError("LineSweep start and end are the same point")
        if self.speed <= 0.0:
            raise ValueError(f"speed must be positive, got {self.speed}")

        self.transit_s = self.length / self.speed
        self.period_s = 2.0 * (self.transit_s + self.dwell_s)

    def at(self, t: float) -> tuple:
        phase = t % self.period_s
        leg = self.transit_s + self.dwell_s

        if phase < self.transit_s:
            u = phase / self.transit_s
        elif phase < leg:
            u = 1.0
        elif phase < leg + self.transit_s:
            u = 1.0 - (phase - leg) / self.transit_s
        else:
            u = 0.0

        return tuple(s + u * d for s, d in zip(self.start, self._delta))

    def describe(self) -> dict:
        return {
            "kind": "line_sweep",
            "start": list(self.start),
            "end": list(self.end),
            "speed_mps": self.speed,
            "dwell_s": self.dwell_s,
            "length_m": round(self.length, 3),
            "transit_s": round(self.transit_s, 3),
            "period_s": round(self.period_s, 3),
        }


class Orbit:
    """Circle a centre point at fixed radius and height.

    Not used by the stage-one recording, but it is the natural next target
    motion: a sweep only ever shows the target's side, whereas an orbit
    presents every aspect angle, which is what tells you whether a detector has
    learned "drone" or "drone seen from the left".
    """

    def __init__(self, centre, radius: float, height: float,
                 period_s: float = 20.0, phase0: float = 0.0):
        self.centre = tuple(float(v) for v in centre[:2])
        self.radius = float(radius)
        self.height = float(height)
        self.period_s = float(period_s)
        self.phase0 = float(phase0)

    def at(self, t: float) -> tuple:
        a = self.phase0 + 2.0 * math.pi * (t / self.period_s)
        return (self.centre[0] + self.radius * math.cos(a),
                self.centre[1] + self.radius * math.sin(a),
                self.height)

    def describe(self) -> dict:
        return {"kind": "orbit", "centre": list(self.centre), "radius_m": self.radius,
                "height_m": self.height, "period_s": self.period_s}


def build_line_sweep(axis_start, axis_end, height: float, speed: float,
                     dwell_s: float = 0.5) -> LineSweep:
    """A :class:`LineSweep` from 2D endpoints at one height.

    Args:
        axis_start: ``(x, y)`` of one end.
        axis_end: ``(x, y)`` of the other.
        height: Altitude to fly the whole line at, world metres.
        speed: Metres per second.
        dwell_s: Hold at each end, seconds.
    """
    return LineSweep((axis_start[0], axis_start[1], height),
                     (axis_end[0], axis_end[1], height),
                     speed=speed, dwell_s=dwell_s)
