"""A drone in the scene: the Iris airframe, plus the camera bolted to its nose.

## Why this does not use Pegasus's ``Multirotor``

The PEGASUS platform's :class:`PegasusIrisVehicle` wraps Pegasus's ``Multirotor``,
which brings a PX4 SITL autopilot, a simulated IMU/GPS/barometer/magnetometer
suite, four ``World.add_physics_callback`` registrations, and -- because Isaac
Sim 6.0.1 silently stops dispatching those callbacks after ~2 calls -- a
hand-rolled driver that calls them back by hand every step.

All of that exists to make the aircraft *fly itself*. For this rig it is
overhead with a failure mode attached, because at this stage nothing is flying
itself: one drone hovers and the other slides along a straight line, both on
trajectories written down in advance. So this class references the Iris **mesh**
and moves it kinematically, and the only Isaac Sim machinery involved is the
stage and a camera.

What that buys, concretely: no PX4 build, no MAVLink ports, no ``/tmp/px4_lock``
files, no arming, no EKF that can refuse to take off, no physics-callback
workaround, and a trajectory that is *exactly* what was asked for rather than
what a position controller converged to. A recording made this way is
repeatable to the pixel, which is what you want when the recording is the input
to a detector you are trying to measure.

What it costs: the drones have no dynamics. They do not bank into turns, they
do not have rotor wash, and their motion is as smooth as the trajectory
function. When the pursuit half of this project needs real dynamics, the
replacement is a ``PegasusDrone`` implementing the same three methods
(:meth:`set_pose`, :meth:`rgb`, :meth:`position`) with PX4 underneath -- the
recorder and the trajectories do not need to know which one they have.

## The camera, and the two things about it that are easy to get wrong

The intrinsics come from the PEGASUS platform's own calibration
(``config/camera_pegasus_iris_720x420.yaml``), so imagery from this rig is
geometrically the same camera as the rest of the project's simulated data.

Two details are inherited deliberately rather than reinvented:

* **The 20 cm forward offset.** The Iris body mesh reaches x=+0.156; mounting at
  x=+0.2 clears it without reaching the rotor arms at x=+0.267. Mount it at the
  body origin instead and every frame has propeller in it.
* **The mounting convention.** A USD camera looks down its own local -Z with
  +Y up, so an identity local rotation aims it at the sky with the horizon down
  one side. Isaac's ``Camera.set_local_pose(..., camera_axes="world")`` does
  that conversion, and mounting through it -- rather than by handing a
  hand-built quaternion to the ``Camera`` constructor, which takes a raw USD
  rotation and applies no conversion at all -- is what makes "identity" mean
  "pointing where the aircraft is pointing". Getting this wrong renders a
  perfectly sharp, perfectly exposed, upside-down world.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Where the Iris mesh lives inside the patched Pegasus extension checkout. Only
# the .usd is used -- none of the extension's Python is imported.
IRIS_USD_REL = "pegasus/simulator/assets/Robots/Iris/iris.usd"

CAMERA_OFFSET_FLU = (0.2, 0.0, 0.0)

RING_RADIUS_M = 0.22
RING_HEIGHT_M = 0.06
"""Where a ring camera sits, relative to the body origin.

Radially outward along its own boresight and a little above the deck, so a
96-degree lens frames sky and city instead of the aircraft it is bolted to.
"""

CAMERA_CLIPPING = (0.05, 1000.0)
"""Near/far clip, metres.

The far plane is 1000 m, not Pegasus's 100 m default. Indoors 100 m is past the
far wall of any building; outdoors it is *inside* the scene -- Rivermark is
~900 m across -- and everything beyond it is clipped to background. A target
drone at 40 m against a skyline that has been clipped away is a much easier
detection than the real thing.
"""


class KinematicDrone:
    """An Iris airframe with a forward-facing camera, moved by setting its pose.

    Args:
        pegasus_root: Path to the ``pegasus.simulator`` extension checkout --
            only ``assets/Robots/Iris/iris.usd`` is read from it.
        stage_prefix: Stage path to spawn at. Must be unique per drone; two
            drones at the same path is the single easiest way to end up
            recording one aircraft twice.
        intrinsics: Camera calibration with ``width/height/fx/fy/cx/cy``.
        position: Initial world ``(x, y, z)``.
        yaw: Initial heading, radians CCW from +X.
        with_camera: Spawn the onboard camera. The target drone in a
            detection-only run does not need one, and each camera costs a render
            product plus a GPU->CPU readback on every rendered step.
    """

    def __init__(self, pegasus_root: Path, stage_prefix: str, intrinsics,
                 position=(0.0, 0.0, 10.0), yaw: float = 0.0, with_camera: bool = True,
                 mounts=None):
        from isaacsim.core.utils.stage import add_reference_to_stage

        self.stage_prefix = stage_prefix
        self.intrinsics = intrinsics
        self._camera = None
        self.cameras: "dict" = {}
        self.mounts = tuple(mounts) if mounts else ()

        iris_usd = Path(pegasus_root) / IRIS_USD_REL
        if not iris_usd.is_file():
            raise FileNotFoundError(
                f"Iris mesh not found at {iris_usd}. Point --pegasus-root at a "
                f"PegasusSimulator/extensions/pegasus.simulator checkout."
            )
        add_reference_to_stage(usd_path=str(iris_usd), prim_path=stage_prefix)

        self._translate_op, self._orient_op = _make_xformable(stage_prefix)
        self._neutralize_physics(stage_prefix)
        self.set_pose(position, yaw)

        if with_camera:
            if self.mounts:
                for m in self.mounts:
                    self.cameras[m.name] = self._make_camera(mount=m)
                self._camera = self.cameras[self.mounts[0].name]
            else:
                self._camera = self._make_camera()
                self.cameras["nose"] = self._camera

    # -- pose ---------------------------------------------------------------

    def set_pose(self, position, yaw: float) -> None:
        """Place the aircraft. Takes effect on the next render.

        Args:
            position: World ``(x, y, z)``, metres.
            yaw: Heading, radians CCW from +X.
        """
        from pxr import Gf

        self._position = tuple(float(v) for v in position)
        self._yaw = float(yaw)
        self._translate_op.Set(Gf.Vec3d(*self._position))
        self._orient_op.Set(np.degrees(self._yaw))

    def position(self) -> tuple:
        """Last commanded world ``(x, y, z)``."""
        return self._position

    def yaw(self) -> float:
        """Last commanded heading, radians."""
        return self._yaw

    def look_at(self, target_xyz) -> float:
        """Point the aircraft's nose at a world point and return the yaw used.

        Yaw only -- the airframe stays level. That is not a simplification for
        its own sake: the camera is rigidly bolted to the body, so pitching the
        aircraft to centre a target vertically also rolls the horizon through
        the frame, and an observer that keeps its target centred by attitude
        produces a recording where the background moves and the target does not
        -- the exact opposite of what a motion-based detector needs to see.
        """
        dx = float(target_xyz[0]) - self._position[0]
        dy = float(target_xyz[1]) - self._position[1]
        yaw = float(np.arctan2(dy, dx))
        self.set_pose(self._position, yaw)
        return yaw

    # -- camera -------------------------------------------------------------

    def rgb(self, name: str = None):
        """Current RGB frame as ``(H, W, 3)`` uint8, or None before warm-up.

        Returns a **copy**. ``Camera.get_rgb()`` hands back a slice of the
        annotator's buffer, and holding that across the next render is a
        use-after-overwrite that shows up as two identical panes in a
        side-by-side video rather than as an error. With a ring of four cameras
        read back into one message that stops being a subtle bug and becomes
        four identical panes.

        Args:
            name: Which mount to read. None means the primary camera (the nose,
                or the first ring mount).
        """
        cam = self._camera if name is None else self.cameras.get(name)
        if cam is None:
            return None
        data = cam.get_rgb()
        if data is None or data.size == 0:
            return None
        return np.array(data[:, :, :3], dtype=np.uint8, copy=True)

    @property
    def camera(self):
        """The Isaac ``Camera``, for callers that need its annotators.

        Rendered ground truth (a tight bounding box, depth, segmentation) is
        attached to the camera rather than computed from poses, so the pursuit
        server needs the object itself and not just its pixels.
        """
        return self._camera

    def camera_world_pose(self) -> tuple:
        """World ``(position, quaternion)`` of the camera, scalar-first."""
        if self._camera is None:
            return None
        return self._camera.get_world_pose()

    def _make_camera(self, mount=None):
        import math

        from isaacsim.sensors.camera import Camera

        intr = self.intrinsics
        if mount is None:
            path = f"{self.stage_prefix}/body/front_camera"
            offset = np.array(CAMERA_OFFSET_FLU)
            quat = np.array([1.0, 0.0, 0.0, 0.0])       # scalar-first identity
        else:
            path = f"{self.stage_prefix}/body/ring_camera_{mount.name}"
            # Pushed out along the camera's own boresight rather than sitting at
            # the body origin: the Iris body reaches 0.156 m and the rotor arms
            # 0.267 m, so a 96-degree camera at the origin frames its own
            # airframe. 0.22 m clears the body and puts the arms 71 degrees off
            # axis, outside the 48-degree half-angle.
            a = mount.yaw
            offset = np.array([RING_RADIUS_M * math.cos(a),
                               RING_RADIUS_M * math.sin(a), RING_HEIGHT_M])
            quat = np.array([math.cos(a / 2.0), 0.0, 0.0, math.sin(a / 2.0)])
        camera = Camera(prim_path=path, resolution=(intr.width, intr.height))
        # Mount it with set_local_pose's "world" camera axes (+Z up, +X forward),
        # NOT by handing a quaternion to the constructor. The constructor's
        # `orientation` is a raw USD local rotation, and a USD camera looks down
        # its own -Z with +Y up -- so an identity there points the camera
        # straight up with the horizon down one side. Passing identity through
        # `camera_axes="world"` means what it looks like it means: aligned with
        # the aircraft's nose, level, +Z up -- and a yaw quaternion through the
        # same conversion means "rotated that far around the airframe", which is
        # what a ring mount is.
        camera.set_local_pose(
            translation=offset,
            orientation=quat,
            camera_axes="world",
        )
        camera.initialize()
        camera.set_clipping_range(*CAMERA_CLIPPING)
        # OpenCV pinhole rather than Isaac's focal-length/aperture model: the
        # calibration this project uses everywhere else is an OpenCV intrinsics
        # matrix, and converting it into focal length + aperture and back is a
        # lossy round trip that quietly changes the field of view.
        camera.set_lens_distortion_model("OmniLensDistortionOpenCvPinholeAPI")
        camera.set_opencv_pinhole_properties(
            cx=intr.cx, cy=intr.cy, fx=intr.fx, fy=intr.fy)
        return camera

    # -- physics ------------------------------------------------------------

    @staticmethod
    def _neutralize_physics(stage_prefix: str) -> None:
        """Make every rigid body under the drone kinematic.

        The Iris USD carries ``RigidBodyAPI`` and an articulation, because it is
        meant to be flown. Referenced as-is into a world with physics running,
        PhysX takes ownership the moment the timeline starts and the aircraft
        falls out of the sky -- and, worse, it *keeps* falling while the
        trajectory code sets its transform every step, so the two fight and the
        result is a drone that jitters downward rather than one that visibly
        drops.

        Setting ``physics:kinematicEnabled`` hands control back: PhysX still
        knows the body exists (so it can collide with things), but integrates
        nothing and moves it only where its transform says.
        """
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        import omni.usd
        from pxr import Usd

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(stage_prefix)
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)


def _make_xformable(prim_path: str):
    """Give ``prim_path`` a clean translate+rotateZ op pair and return them.

    Referenced assets arrive with whatever transform ops their author left on
    them. Appending a second translate to an existing one does not replace it,
    it composes -- so a drone commanded to (0, 0, 10) ends up wherever the
    asset's own offset put it, plus ten metres. Clearing the op order first is
    what makes ``set_pose`` mean an absolute world pose.

    Returns:
        ``(translate_op, rotate_z_op)``.
    """
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"nothing at {prim_path} -- the Iris reference did not compose")

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    translate = xform.AddTranslateOp()
    rotate = xform.AddRotateZOp()
    translate.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    rotate.Set(0.0)
    return translate, rotate
