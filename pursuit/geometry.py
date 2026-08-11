"""Turning pixels into a bearing, a range and a closing rate.

Everything the guidance law needs about *where the target is* comes through
here. The chaser has one camera and no depth sensor, so this module is the whole
sensing model:

* **Bearing** is exact. A pixel plus intrinsics is a ray, full stop.
* **Range** is not. With a single camera the only scale cue is that the target
  is a known-size object, so range is ``f * S / s`` -- focal length times the
  target's physical span over its pixel span. That is a real technique and it is
  what a real interceptor with one camera does, but its error is proportional to
  the *square* of the range (a one-pixel error at 10 px costs 10% of the range;
  at 4 px it costs 25%), and it is only as good as the assumed span.

The consequence shapes the whole design downstream: bearing is trusted and range
is not. The guidance law is built so that lateral steering -- the part that
actually decides whether the two aircraft meet -- depends only on bearing, and
range enters only as a speed schedule and a terminal trigger, where being 20%
wrong costs some time rather than the intercept.

Frames used here:

``image``
    OpenCV: origin top-left, ``+u`` right, ``+v`` down.
``camera/body``
    REP-103 on a level yaw-only airframe: ``+x`` forward (boresight), ``+y``
    left, ``+z`` up. The camera is bolted to the nose, so body and camera axes
    coincide and no extrinsic calibration appears anywhere in this file.
``world``
    ``+z`` up, yaw measured CCW from ``+x``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class Intrinsics:
    """A pinhole camera. Mirrors ``simulators.pegasus.camera.Intrinsics``."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        return cls(width=int(d["width"]), height=int(d["height"]),
                   fx=float(d["fx"]), fy=float(d["fy"]),
                   cx=float(d["cx"]), cy=float(d["cy"]))

    @property
    def hfov_deg(self) -> float:
        return math.degrees(math.atan2(self.cx, self.fx)
                            + math.atan2(self.width - self.cx, self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(math.atan2(self.cy, self.fy)
                            + math.atan2(self.height - self.cy, self.fy))


def bearing_from_pixel(intr: Intrinsics, u: float, v: float) -> Tuple[float, float]:
    """Azimuth and elevation of the ray through pixel ``(u, v)``, radians.

    Azimuth is positive to the **left** (``+y``, REP-103), elevation positive
    **up** -- both matching the body frame the controller commands in, so no
    sign flip is needed anywhere downstream.

    ``atan`` rather than the small-angle ``(u - cx) / fx``: this camera is 76
    degrees across, and at the frame edge the linear approximation is off by
    several degrees. A target being chased spends much of the acquisition phase
    exactly there.
    """
    az = -math.atan2(float(u) - intr.cx, intr.fx)
    el = -math.atan2(float(v) - intr.cy, intr.fy)
    return az, el


def pixel_from_bearing(intr: Intrinsics, az: float, el: float) -> Tuple[float, float]:
    """Inverse of :func:`bearing_from_pixel` (used to draw predictions)."""
    u = intr.cx - intr.fx * math.tan(az)
    v = intr.cy - intr.fy * math.tan(el)
    return u, v


def normalized_offset(intr: Intrinsics, u: float, v: float) -> Tuple[float, float]:
    """Offset from the principal point as a fraction of the half-frame.

    ``+x`` right, ``+y`` down (image convention), each in roughly ``[-1, 1]``.
    This is what the reference visual servo consumes; it is kept for the
    centring deadbands and gates, where "how far off-centre as a fraction of
    frame" is the natural unit, while the guidance law itself uses true angles.
    """
    ox = (float(u) - intr.cx) / (0.5 * intr.width)
    oy = (float(v) - intr.cy) / (0.5 * intr.height)
    return ox, oy


def offaxis_scale(intr: Intrinsics, u: Optional[float], v: Optional[float]) -> float:
    """Pixels per radian at ``(u, v)``, as a multiple of the on-axis ``fx``.

    A pinhole camera is not uniform. ``u - cx = fx tan(a)``, so the local scale
    is ``fx sec^2(a)`` -- an object of fixed *angular* size covers more pixels
    the further off-axis it sits. On the boresight this is 1.0 and can be
    ignored; at the edge of a 96-degree ring camera it is **2.2**, and a
    monocular range that ignores it reads 55 percent short.

    That error is not academic here. A ring means the target no longer has to
    be centred to be tracked, so it now spends whole engagements at 40 degrees
    off-axis where the single forward camera would have been yawing to bring it
    back. Measured on the rig's own sweep: the rendered span of a target at a
    fixed 40 m grows from 8 px on the boresight to 15 px at 43 degrees off it,
    which is ``sec^2(43) = 1.87`` to within the pixel.

    Returns the larger of the two axis scales, matching
    :attr:`~pursuit.perception.Box.span`, which is the larger side of the box.
    """
    if u is None or v is None:
        return 1.0
    tx = (float(u) - intr.cx) / intr.fx
    ty = (float(v) - intr.cy) / intr.fy
    return max((1.0 + tx * tx), (intr.fy / intr.fx) * (1.0 + ty * ty))


def range_from_span(intr: Intrinsics, span_px: float, span_m: float,
                    u: Optional[float] = None, v: Optional[float] = None
                    ) -> Optional[float]:
    """Monocular range from a known-size target, metres.

    ``range = fx * span_m / span_px``, corrected for where in the image the
    target sits when ``u``/``v`` are supplied -- see :func:`offaxis_scale`.
    Returns None for a span too small to mean anything: below ~2 px the box is
    quantisation noise and the implied range is a number with no information in
    it.
    """
    if span_px is None or span_m <= 0.0:
        return None
    span_px = float(span_px)
    # `span_px < 2.0` is False for NaN, so a comparison alone lets a NaN through
    # and it comes back out as a NaN range -- which then poisons the range
    # filter's EMA permanently rather than being treated as a missing
    # measurement. A degenerate box from a real detector is exactly how one gets
    # here. Infinity is the same story from the other end: it reads as a target
    # at zero range, i.e. contact.
    if not math.isfinite(span_px) or span_px < 2.0:
        return None
    return (float(intr.fx) * offaxis_scale(intr, u, v)
            * float(span_m) / span_px)


def span_from_range(intr: Intrinsics, range_m: float, span_m: float) -> float:
    """Pixel span a target of ``span_m`` subtends at ``range_m`` (the inverse)."""
    if range_m <= 0.0:
        return float("inf")
    return float(intr.fx) * float(span_m) / float(range_m)


def body_to_world(yaw: float, vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
    """Rotate a body-frame velocity into world axes for a level, yawed airframe."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (vx * c - vy * s, vx * s + vy * c, vz)


def world_to_body(yaw: float, wx: float, wy: float, wz: float) -> Tuple[float, float, float]:
    """Rotate a world-frame velocity into body axes (inverse of :func:`body_to_world`)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (wx * c + wy * s, -wx * s + wy * c, wz)


def wrap_pi(a: float) -> float:
    """Wrap an angle to ``[-pi, pi)``.

    Half-open at the *bottom*: exactly ``+pi`` comes back as ``-pi``. That is
    what this arithmetic does, and the interval is stated the way it is because
    the docstring used to claim the other one -- which matters only for a caller
    sitting precisely on the seam, where it would read a heading error of the
    opposite sign.
    """
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------------------------- camera rings

def pixel_to_body(intr: Intrinsics, u: float, v: float,
                  mount_yaw: float = 0.0) -> Tuple[float, float, float]:
    """Unit direction in **body** axes of the ray through a ring camera's pixel.

    Three steps, and the middle one is the whole reason this function exists
    rather than an addition of two azimuths. A pixel gives *tangent-plane*
    angles about that camera's own boresight; those are not additive across a
    mount rotation. ``az = 40`` in a camera bolted 90 degrees round is not
    ``az = 130`` in the body frame -- it is whatever
    ``Rz(90) . unit(1, tan 40, tan el)`` says, and the two differ by degrees as
    soon as the elevation is non-zero. Rotating the *direction* is exact at any
    mount angle and any elevation, which a ring needs and a nose camera never
    did.

    Returns ``(x, y, z)`` with ``+x`` forward, ``+y`` left, ``+z`` up.
    """
    az = -math.atan2(float(u) - intr.cx, intr.fx)
    el = -math.atan2(float(v) - intr.cy, intr.fy)
    fx_, ly, lz = 1.0, math.tan(az), math.tan(el)
    n = math.sqrt(fx_ * fx_ + ly * ly + lz * lz)
    fx_, ly, lz = fx_ / n, ly / n, lz / n
    c, s = math.cos(mount_yaw), math.sin(mount_yaw)
    return (fx_ * c - ly * s, fx_ * s + ly * c, lz)


def body_to_pixel(intr: Intrinsics, los_body: Sequence[float],
                  mount_yaw: float = 0.0) -> Optional[Tuple[float, float]]:
    """Inverse of :func:`pixel_to_body`; None when the ray is behind the camera.

    "Behind" is a real answer, not an error. With four cameras every direction
    is in front of exactly one or two of them, and asking each in turn -- taking
    None for an answer three times -- is how a bearing is turned back into the
    camera that owns it.
    """
    c, s = math.cos(-mount_yaw), math.sin(-mount_yaw)
    x, y, z = (float(v) for v in los_body)
    fwd = x * c - y * s
    left = x * s + y * c
    if fwd <= 1e-6:
        return None
    return (intr.cx - intr.fx * (left / fwd), intr.cy - intr.fy * (z / fwd))


def body_bearing(los_body: Sequence[float]) -> Tuple[float, float]:
    """``(azimuth, elevation)`` of a body direction, radians, full circle.

    **Spherical**, unlike :func:`bearing_from_pixel`'s tangent-plane pair, and
    the difference is deliberate rather than an inconsistency. Tangent-plane
    angles are what a pinhole *measures* and they are undefined at 90 degrees
    off the nose -- exactly where a ring routinely has its target. These are
    for reporting, for the mode logic and for the tracker's smoothing, all of
    which need an angle that keeps meaning something all the way round. Anything
    that steers uses the direction vector itself and never these.
    """
    x, y, z = (float(v) for v in los_body)
    return (math.atan2(y, x),
            math.atan2(z, math.hypot(x, y)))


def bearing_to_body(az: float, el: float) -> Tuple[float, float, float]:
    """Inverse of :func:`body_bearing`: spherical angles back to a unit vector."""
    ce = math.cos(el)
    return (ce * math.cos(az), ce * math.sin(az), math.sin(el))


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    """Great-circle angle between two directions, radians.

    The association metric for a ring tracker. Differencing azimuths would look
    equivalent and is not: near the zenith a degree of azimuth is no distance at
    all, and across the +/-180 seam it is a full turn. Neither failure is
    survivable in a gate.
    """
    d = sum(float(a[i]) * float(b[i]) for i in range(3))
    na = math.sqrt(sum(float(c) * float(c) for c in a))
    nb = math.sqrt(sum(float(c) * float(c) for c in b))
    if na <= 1e-12 or nb <= 1e-12:
        return math.pi
    return math.acos(max(-1.0, min(1.0, d / (na * nb))))


def yaw_homography(intr: Intrinsics, dpsi: float):
    """Pixel mapping induced in one camera by the airframe yawing ``dpsi``.

    ``H = K R K^-1``, exactly, for a rotation about the camera's own down axis.
    The point of computing it instead of estimating it: the interceptor knows
    its own heading to machine precision, and optical flow on an empty sky does
    not. A motion detector that falls back to *identity* when the flow solve
    fails -- which is what happens against sky -- declares the entire frame to
    be moving the moment the aircraft turns, and buries a 3-pixel drone in it.

    Args:
        intr: The camera.
        dpsi: Heading change since the frame being warped, radians CCW.

    Returns:
        3x3 ``numpy`` array mapping old pixel coordinates to new ones.
    """
    import numpy as np

    c, s = math.cos(float(dpsi)), math.sin(float(dpsi))
    # Optical axes are x right, y down, z forward, so a body yaw is a rotation
    # about +y_opt. Derived rather than guessed: a fixed world point must move
    # to larger u when the aircraft turns left, which is the sign below.
    r = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    k = np.array([[intr.fx, 0.0, intr.cx],
                  [0.0, intr.fy, intr.cy],
                  [0.0, 0.0, 1.0]], dtype=float)
    return k @ r @ np.linalg.inv(k)


def los_unit(chaser_xyz: Sequence[float], target_xyz: Sequence[float]):
    """Unit line-of-sight vector from chaser to target, and the range."""
    dx = float(target_xyz[0]) - float(chaser_xyz[0])
    dy = float(target_xyz[1]) - float(chaser_xyz[1])
    dz = float(target_xyz[2]) - float(chaser_xyz[2])
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    if r <= 1e-9:
        return (1.0, 0.0, 0.0), 0.0
    return (dx / r, dy / r, dz / r), r
