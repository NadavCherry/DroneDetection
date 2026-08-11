#!/usr/bin/env python3
"""Two drones outdoors: one hovers and watches, one sweeps a line. Records both.

Stage one of the air-to-air pipeline. The observer (drone 1) holds a fixed point
at a fixed height with its nose on the target. The target (drone 2) slides back
and forth along a straight axis at constant height. The recording is a
side-by-side of **both aircraft's own onboard cameras** -- observer left, target
right.

Must run inside the ``isaac-sim`` container, under Isaac Sim's own Python:

    docker exec isaac-sim bash -c "cd /tmp/dev/dronedet && /isaac-sim/python.sh \\
        simulators/pegasus/scripts/run_two_drone.py --scene rivermark --seconds 30"

From the host, use ``simulators/pegasus/scripts/run_in_container.sh``, which
syncs both repos in and copies the recording back out.

## The frame budget, and why the loop looks the way it does

Physics runs at 250 Hz and rendering is expensive, so the loop renders only
every Nth step (:func:`render_every_n_steps`). Both cameras refresh on the same
rendered step, which is what keeps the two panes of the video synchronised
without any alignment pass -- they are literally the same instant of the same
clock. This is the main reason to record both aircraft in one process rather
than flying them separately and stacking the videos afterwards.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# This file lives at simulators/pegasus/scripts/; the package root is three up.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PHYSICS_HZ = 250.0
PHYSICS_DT = 1.0 / PHYSICS_HZ
# The onboard camera ignores its first ~100 render callbacks before producing a
# frame. Counted in RENDERED ticks, not physics steps.
CAMERA_WARMUP_RENDERS = 120


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="rivermark",
                   choices=["rivermark", "skydome", "rivermark_props"],
                   help="outdoor environment (default: rivermark, the town)")
    p.add_argument("--sky", default="clear",
                   help="HDRI sky for the non-rivermark scenes (rivermark ships its own)")
    p.add_argument("--seconds", type=float, default=30.0,
                   help="simulated seconds to record")
    p.add_argument("--fps", type=float, default=20.0, help="capture rate, Hz")
    p.add_argument("--altitude", type=float, default=20.0,
                   help="flight height above the scene's ground, metres")
    p.add_argument("--standoff", type=float, default=20.0,
                   help="observer-to-axis distance, metres. Drives how many pixels "
                        "the target subtends -- the script prints the number")
    p.add_argument("--axis-half-length", type=float, default=10.0,
                   help="target sweeps -L..+L across the observer's boresight "
                        "(default 10, i.e. the -10,0 .. 10,0 axis)")
    p.add_argument("--target-speed", type=float, default=3.0, help="target speed, m/s")
    p.add_argument("--dwell", type=float, default=0.5,
                   help="target hold at each end of the axis, seconds")
    p.add_argument("--target-faces", default="observer",
                   choices=["observer", "travel"],
                   help="where the target's own camera points: back at the observer "
                        "(both panes then show an aircraft) or along its track")
    p.add_argument("--resolution", default="1440x840",
                   help="render size WxH per pane. The default is an exact 2x of the "
                        "platform calibration's native 720x420 -- a UNIFORM scale, so "
                        "the field of view is unchanged and the target simply lands on "
                        "twice as many pixels. Pass 720x420 for native. Avoid sizes of "
                        "a different aspect ratio (1280x720 is 1.778 against the "
                        "camera's 1.714): fx and fy then scale by different factors, "
                        "which is a stretched camera, not the same one rendered larger")
    p.add_argument("--out-dir", default="/tmp/dev/air2air/run",
                   help="where to write the recording")
    p.add_argument("--pegasus-root", default="/tmp/dev/PegasusSimulator/extensions/pegasus.simulator")
    p.add_argument("--pegasus-config", default=None,
                   help="robots/PEGASUS/config dir. Defaults next to --platform-root")
    p.add_argument("--platform-root", default="/tmp/dev/platform",
                   help="container-side PEGASUS platform tree, for the camera calibration")
    p.add_argument("--load-timeout", type=float, default=None,
                   help="override the scene's load budget, seconds")
    p.add_argument("--scout", action="store_true",
                   help="render a few overview stills of the scene and exit "
                        "without recording -- for choosing where to fly")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

    from simulators.pegasus.camera import load_intrinsics

    config_dir = (Path(args.pegasus_config) if args.pegasus_config
                  else Path(args.platform_root) / "robots/PEGASUS/config")
    resolution = None
    if args.resolution:
        w, h = args.resolution.lower().split("x")
        resolution = (int(w), int(h))
    intr = load_intrinsics(config_dir, resolution=resolution)
    log(f"camera: {intr.width}x{intr.height} fx={intr.fx:.1f} "
        f"HFOV={intr.hfov_deg:.1f}deg VFOV={intr.vfov_deg:.1f}deg")

    # ---- boot -------------------------------------------------------------
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    log("Isaac Sim up")

    from isaacsim.core.api import World

    from simulators.pegasus.camera import in_frame, project, IRIS_SPAN_M
    from simulators.pegasus.control.trajectories import Hover, build_line_sweep
    from simulators.pegasus.drone import KinematicDrone
    from simulators.pegasus.recording.split_recorder import SplitScreenRecorder
    from simulators.pegasus.scenes.outdoor import (
        load_outdoor_scene, scene_ground_z, scene_origin_xy,
    )

    # Equal physics and rendering timesteps so one world.step() advances exactly
    # PHYSICS_DT whether or not it rendered. Unequal ones make Isaac Sim take
    # int(rendering_dt/physics_dt) substeps per step, so a rendered step and a
    # quiet one advance different amounts of time and every stamp is wrong.
    world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT,
                  stage_units_in_meters=1.0)
    log("world created")

    scene_info = load_outdoor_scene(simulation_app, args.scene, sky=args.sky,
                                    load_timeout_s=args.load_timeout, progress=log)

    # ---- geometry ---------------------------------------------------------
    ground_z = scene_ground_z(args.scene)
    ox, oy = scene_origin_xy(args.scene)
    height = ground_z + args.altitude

    observer_xyz = (ox, oy, height)
    # The target's axis runs across the observer's boresight (which is +X), so
    # the sweep is in Y and the standoff is in X -- i.e. exactly the -10,0 ..
    # 10,0 axis, placed `standoff` metres in front of the observer.
    axis_x = ox + args.standoff
    axis_start = (axis_x, oy - args.axis_half_length)
    axis_end = (axis_x, oy + args.axis_half_length)

    observer_traj = Hover(observer_xyz)
    target_traj = build_line_sweep(axis_start, axis_end, height,
                                   speed=args.target_speed, dwell_s=args.dwell)

    near = args.standoff
    far = math.hypot(args.standoff, args.axis_half_length)
    log(f"ground z={ground_z:.2f}m, flying at z={height:.2f}m "
        f"({args.altitude:.1f}m above ground)")
    log(f"observer hovers at {observer_xyz}")
    log(f"target sweeps {axis_start} .. {axis_end} at {args.target_speed} m/s "
        f"(period {target_traj.period_s:.1f}s)")
    log(f"range {near:.1f}..{far:.1f}m -> target spans "
        f"{intr.pixel_span(IRIS_SPAN_M, far):.1f}..{intr.pixel_span(IRIS_SPAN_M, near):.1f} px")

    if args.scout:
        _scout(world, simulation_app, args, observer_xyz, log)
        simulation_app.close()
        return 0

    # ---- drones -----------------------------------------------------------
    pegasus_root = Path(args.pegasus_root)
    observer = KinematicDrone(pegasus_root, "/World/drone_observer", intr,
                              position=observer_xyz, yaw=0.0)
    target = KinematicDrone(pegasus_root, "/World/drone_target", intr,
                            position=target_traj.at(0.0), yaw=math.pi)
    log("both drones spawned")

    world.reset()
    log("world reset")

    # ---- warm up ----------------------------------------------------------
    # Counted in rendered ticks. Both cameras share the same render pass, so one
    # warm-up loop warms both.
    for i in range(CAMERA_WARMUP_RENDERS):
        world.step(render=True)
        if i % 40 == 0:
            log(f"  camera warm-up {i}/{CAMERA_WARMUP_RENDERS}")
    if observer.rgb() is None or target.rgb() is None:
        log("WARNING: a camera is still not producing frames after warm-up")
    log("cameras warm")

    # ---- record -----------------------------------------------------------
    out_dir = Path(args.out_dir)
    recorder = SplitScreenRecorder(
        out_dir, size=(intr.width, intr.height), fps=args.fps,
        labels=("DRONE 1 - observer (hover)", "DRONE 2 - target (sweep)"))
    log(f"recording to {out_dir}")

    steps_per_frame = max(int(round(PHYSICS_HZ / args.fps)), 1)
    total_frames = int(args.seconds * args.fps)
    sim_t = 0.0
    seen = 0

    for frame in range(total_frames):
        sim_t = frame / args.fps

        tgt_xyz = target_traj.at(sim_t)
        obs_xyz = observer_traj.at(sim_t)

        observer.set_pose(obs_xyz, 0.0)
        obs_yaw = observer.look_at(tgt_xyz)

        target.set_pose(tgt_xyz, 0.0)
        if args.target_faces == "observer":
            tgt_yaw = target.look_at(obs_xyz)
        else:
            # Along the track. The sweep is in +/-Y, so the heading is +/-90 deg
            # depending on which way it is currently going.
            nxt = target_traj.at(sim_t + 0.1)
            tgt_yaw = math.atan2(nxt[1] - tgt_xyz[1], nxt[0] - tgt_xyz[0])
            target.set_pose(tgt_xyz, tgt_yaw)

        for _ in range(steps_per_frame - 1):
            world.step(render=False)
        world.step(render=True)

        uv = project(intr, obs_xyz, obs_yaw, tgt_xyz)
        visible = in_frame(intr, uv)
        seen += int(visible)
        rng = math.dist(obs_xyz, tgt_xyz)

        wrote = recorder.capture(
            observer.rgb(), target.rgb(), sim_t,
            extra={
                "observer_xyz": [round(v, 3) for v in obs_xyz],
                "target_xyz": [round(v, 3) for v in tgt_xyz],
                "observer_yaw_deg": round(math.degrees(obs_yaw), 2),
                "range_m": round(rng, 3),
                # Ground truth, free here and expensive later: where the target
                # actually is in the observer's image, and how big. This is what
                # a detection gets scored against.
                "target_uv": [round(uv[0], 2), round(uv[1], 2)] if uv else None,
                "target_px": round(intr.pixel_span(IRIS_SPAN_M, rng), 2),
                "target_in_frame": bool(visible),
            })
        if not wrote:
            log(f"  frame {frame}: a camera returned nothing, skipped")

        if frame % 50 == 0:
            log(f"  frame {frame}/{total_frames} t={sim_t:.1f}s range={rng:.1f}m "
                f"in_frame={visible}")

    stats = recorder.finish(meta={
        "scene": scene_info,
        "intrinsics": intr.as_dict(),
        "observer": observer_traj.describe(),
        "target": target_traj.describe(),
        "altitude_m": args.altitude,
        "standoff_m": args.standoff,
        "target_faces": args.target_faces,
        "frames_target_in_frame": seen,
        "sim_seconds": args.seconds,
    })
    log(f"RECORDING_OK {stats['frames']} frames, target in frame on {seen} of them")
    log(f"video: {stats['video']}")
    print("DONE", flush=True)

    simulation_app.close()
    return 0


def _scout(world, simulation_app, args, centre, log):
    """Render a few overview stills so a human can pick where to fly.

    Choosing a flight area from coordinates alone is guesswork -- the difference
    between open airspace over a car park and a spot four metres inside a
    building is not visible in a bounding box.
    """
    import numpy as np
    from isaacsim.sensors.camera import Camera
    import isaacsim.core.utils.numpy.rotations as rot_utils
    import imageio.v2 as imageio

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cx, cy, cz = centre

    views = [
        ("topdown", (cx, cy, cz + 80.0), (cx, cy, cz - 20.0)),
        ("oblique", (cx - 60.0, cy - 60.0, cz + 40.0), (cx, cy, cz)),
        ("eye_level", (cx - 30.0, cy, cz), (cx + 40.0, cy, cz)),
    ]
    world.reset()
    for name, pos, look in views:
        d = np.array(look) - np.array(pos)
        yaw = float(np.degrees(np.arctan2(d[1], d[0])))
        pitch = float(np.degrees(np.arctan2(-d[2], np.linalg.norm(d[:2]))))
        cam = Camera(prim_path=f"/World/Scout_{name}", position=np.array(pos),
                     resolution=(1280, 720),
                     orientation=rot_utils.euler_angles_to_quats(
                         np.array([0.0, pitch, yaw]), degrees=True))
        cam.initialize()
        for _ in range(120):
            world.step(render=True)
        rgba = cam.get_rgba()
        if rgba is None:
            log(f"  scout {name}: no frame")
            continue
        arr = np.asarray(rgba)[:, :, :3].astype(np.uint8)
        path = out / f"scout_{name}.png"
        imageio.imwrite(str(path), arr)
        log(f"  scout {name}: mean={arr.mean():.1f} -> {path}")
    log("SCOUT_OK")


if __name__ == "__main__":
    raise SystemExit(main())
