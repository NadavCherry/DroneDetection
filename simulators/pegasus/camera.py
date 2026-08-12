"""The camera calibration, and what it implies about what can be seen.

The intrinsics are read from the PEGASUS platform's own YAML rather than
restated here, so this rig renders the same camera as the rest of the project's
simulated data and a change to the calibration cannot be applied in one place
and missed in the other.

Two camera fits live here, and they are different sensors on purpose:

``nose``
    One PEGASUS Iris camera bolted to the nose. 76 degrees across, high angular
    resolution, and the aircraft has to turn to look anywhere else. Every result
    in ``pursuit/README.md`` up to the ring was flown on this.
``ring``
    Four wide cameras at 90 degree spacing -- the interceptor that does not have
    to turn around. See :func:`ring_mounts` for why four cameras cannot be the
    nose camera four times over.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CALIBRATION = "camera_pegasus_iris_720x420.yaml"

# Rotor-tip to rotor-tip on the Iris airframe, metres. Used only to predict how
# many pixels a target subtends -- it is a sanity figure for choosing a
# standoff, not a measurement anything depends on.
IRIS_SPAN_M = 0.47


@dataclass(frozen=True)
class Intrinsics:
    """A pinhole camera, matching the external platform's own Intrinsics type."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def hfov_deg(self) -> float:
        """Full horizontal field of view, degrees."""
        return math.degrees(math.atan2(self.cx, self.fx)
                            + math.atan2(self.width - self.cx, self.fx))

    @property
    def vfov_deg(self) -> float:
        """Full vertical field of view, degrees."""
        return math.degrees(math.atan2(self.cy, self.fy)
                            + math.atan2(self.height - self.cy, self.fy))

    def scaled(self, width: int, height: int) -> "Intrinsics":
        """The same physical camera rendered at another resolution."""
        sx, sy = width / self.width, height / self.height
        return Intrinsics(width=width, height=height,
                          fx=self.fx * sx, fy=self.fy * sy,
                          cx=self.cx * sx, cy=self.cy * sy)

    def pixel_span(self, size_m: float, range_m: float) -> float:
        """How many pixels wide an object of ``size_m`` is at ``range_m``."""
        if range_m <= 0:
            return float("inf")
        return self.fx * size_m / range_m

    def as_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx,
                "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "hfov_deg": round(self.hfov_deg, 2),
                "vfov_deg": round(self.vfov_deg, 2)}


def load_intrinsics(pegasus_config_dir, name: str = DEFAULT_CALIBRATION,
                    resolution=None) -> Intrinsics:
    """Read one of ``robots/PEGASUS/config/*.yaml``.

    Args:
        pegasus_config_dir: The ``robots/PEGASUS/config`` directory.
        name: Which calibration file.
        resolution: Optional ``(width, height)`` to rescale to.

    Raises:
        FileNotFoundError: If the calibration is not where it was expected.
    """
    import yaml

    path = Path(pegasus_config_dir) / name
    if not path.is_file():
        raise FileNotFoundError(
            f"camera calibration not found at {path}. Point --pegasus-config at "
            f"the external platform's robots/PEGASUS/config directory."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    intr = Intrinsics(width=data["image_width"], height=data["image_height"],
                      fx=data["fx"], fy=data["fy"], cx=data["cx"], cy=data["cy"])
    if resolution is not None:
        intr = intr.scaled(int(resolution[0]), int(resolution[1]))
    return intr


def project(intr: Intrinsics, cam_pos, cam_yaw: float, point):
    """Where a world point lands in the image, for a level yaw-only camera.

    Only yaw is modelled because :class:`~simulators.pegasus.drone.KinematicDrone`
    keeps the airframe level -- see its ``look_at`` docstring for why.

    Args:
        intr: The camera.
        cam_pos: Camera world ``(x, y, z)``.
        cam_yaw: Camera heading, radians CCW from +X.
        point: World ``(x, y, z)`` to project.

    Returns:
        ``(u, v, depth_m)``, or None if the point is behind the camera. ``u``/``v``
        may be outside the image -- the caller decides what "visible" means.
    """
    dx = point[0] - cam_pos[0]
    dy = point[1] - cam_pos[1]
    dz = point[2] - cam_pos[2]

    c, s = math.cos(-cam_yaw), math.sin(-cam_yaw)
    forward = dx * c - dy * s
    left = dx * s + dy * c

    if forward <= 1e-6:
        return None

    # Optical frame: x right (= -left), y down (= -up), z forward.
    u = intr.cx + intr.fx * (-left) / forward
    v = intr.cy + intr.fy * (-dz) / forward
    return u, v, forward


def in_frame(intr: Intrinsics, uv, margin_px: float = 0.0) -> bool:
    """Whether a projected point falls inside the image."""
    if uv is None:
        return False
    u, v = uv[0], uv[1]
    return (-margin_px <= u < intr.width + margin_px
            and -margin_px <= v < intr.height + margin_px)


# --------------------------------------------------------------- camera rings

RING_HFOV_DEG = 96.0
"""Horizontal field of view of one ring camera, degrees.

Four cameras 90 degrees apart need **more** than 90 degrees each or the ring has
holes, and a hole in a counter-UAS sensor is a direction an intruder can arrive
from unseen. 96 leaves 6 degrees of overlap at every seam -- enough that a target
is handed from one camera to the next while both can see it, rather than
vanishing for a frame in between.
"""

RING_RESOLUTION = (2048, 704)
"""Pixels per ring camera, and the number the whole city mission turns on.

Angular resolution is ``width / (2 tan(hfov/2))``. At 2048 px across 96 degrees
that is ``fx = 921.9`` -- **the same 16.1 px/deg as the PEGASUS nose camera** --
which is the clean way to say what the ring is: not a cheaper sensor, the same
sensor four times over, each seeing 96 degrees instead of 76.

It is chosen, not inherited, and what it was chosen against is the mission
clock. Defending a structure ``d`` metres away from a head-on intruder requires
acquiring that intruder beyond ``d (1 + v_i/v_c)``, which at a 1.5x speed
advantage is ``1.67 d``. Every metre of detection range is therefore 0.6 m of
defended radius, and nothing else in the system trades that favourably: a
1600 px ring (12.6 px/deg) defends a 68 m radius at a 3 px detection floor and
this one defends 86 m, which is the difference between covering one plaza
building and covering four. See ``pursuit/city.py``.

The shape is **wide and short**, and that is where the cost is paid. 704 rows
gives a 41.8 degree vertical field of view against the nose camera's 47.8 -- the
only thing actually given up -- and the ring spends what is left better: its
principal point is centred rather than carrying the real payload's downward
tilt, so it sees 20.9 degrees *above* the horizon where the nose camera sees
15.5. An intruder crossing a city at altitude is above the horizon, and "the
camera cannot look up" is the sensor limit that cost this project the most
engagements.

A 0.47 m Iris subtends

| range | ring span | nose span |
|---|---|---|
| 20 m | 21.7 px | 21.7 px |
| 50 m | 8.7 px | 8.7 px |
| 100 m | 4.3 px | 4.3 px |
| 150 m | 2.9 px | 2.9 px |

and the last two rows are the ones the mission lives on, far below anything an
appearance model finds. What reaches them is the thing the ring makes possible:
an interceptor holding station has four *stationary* cameras, and a 3 px mover
against a stationary background is what the rest of this repository detects for
a living. See :class:`pursuit.ring.RingMotionDetector`.
"""


@dataclass(frozen=True)
class Mount:
    """One camera bolted to the airframe, facing ``yaw_deg`` off the nose.

    Yaw only, and level, for the same reason the airframe is: the cameras are
    rigidly attached, so the only thing that changes where a camera points is
    where the aircraft points.
    """

    name: str
    yaw_deg: float

    @property
    def yaw(self) -> float:
        return math.radians(self.yaw_deg)


NOSE_MOUNT = Mount("nose", 0.0)

RING_MOUNT_NAMES = ("fwd", "left", "aft", "right")
"""Body-relative, not compass-relative.

The mission talks about north/south/east/west because the interceptor holds a
heading while it watches, but the cameras are bolted to the *airframe*: the
moment it yaws, "north" would be a lie and "left" would still be true. Every
bearing in this system is body-relative for exactly that reason, and the
compass mapping is a property of the current heading, not of the hardware.
"""


def ring_mounts(n: int = 4) -> tuple:
    """``n`` cameras spaced evenly around the airframe, first one on the nose."""
    if n <= 0:
        raise ValueError("a ring needs at least one camera")
    step = 360.0 / n
    return tuple(Mount(RING_MOUNT_NAMES[i] if n == 4 else f"cam{i}", i * step)
                 for i in range(n))


def ring_intrinsics(width: int = RING_RESOLUTION[0], height: int = RING_RESOLUTION[1],
                    hfov_deg: float = RING_HFOV_DEG) -> Intrinsics:
    """A wide, square-pixel, centred camera for the ring.

    Centred principal point, unlike the nose camera, whose ``cy`` sits high
    because the real PEGASUS payload is tilted down. A ring watching for
    intruders arriving across a city wants its field of view spread evenly about
    the horizon, and inheriting a downward tilt from a different sensor would be
    copying a number rather than choosing one.
    """
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return Intrinsics(width=int(width), height=int(height), fx=f, fy=f,
                      cx=width / 2.0, cy=height / 2.0)


def ring_coverage_deg(intr: Intrinsics, n: int) -> tuple:
    """``(covered, overlap_per_seam)`` degrees for ``n`` of this camera.

    A ring with a hole in it is worse than one camera, because the hole is
    invisible in every summary statistic -- so this is computed, printed at
    boot, and asserted rather than assumed.
    """
    hfov = intr.hfov_deg
    return (n * hfov, hfov - 360.0 / n)


def mount_position(body_xyz, body_yaw: float, mount: Mount,
                   radius: float = 0.22, height: float = 0.06) -> tuple:
    """World position of a ring camera's optical centre.

    The offset is 22 cm, which is nothing at 100 m and 2.5 degrees at 5 m --
    and 5 m is the terminal phase, where the analytic projection is
    cross-checked against the rendered box. Dropping it would show up as a
    "render lag" that grows as the aircraft closes.
    """
    a = float(body_yaw) + mount.yaw
    return (float(body_xyz[0]) + radius * math.cos(a),
            float(body_xyz[1]) + radius * math.sin(a),
            float(body_xyz[2]) + height)
