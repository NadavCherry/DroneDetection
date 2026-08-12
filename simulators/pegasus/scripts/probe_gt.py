#!/usr/bin/env python3
"""Probe: can Isaac Sim hand us a pixel-exact bounding box for the target drone?

The projected-centre ground truth ``run_two_drone.py`` writes is analytic -- it
assumes a level yaw-only camera at the *body* origin and it is computed from the
pose we asked for, not the pose that was rendered. Both assumptions cost pixels,
and at a 25-pixel target a few pixels of bias is the difference between a usable
training label and a poisoned one.

Isaac's own ``bounding_box_2d_tight`` annotator has neither problem: it is
measured on the same rendered frame the detector will see. This script checks
that it is reachable in Isaac Sim 6.0.1, that semantics can be attached to a
referenced Iris, and prints the annotator box next to the analytic projection so
the bias is visible rather than assumed.

    docker exec isaac-sim bash -c "cd /tmp/dev/dronedet && /isaac-sim/python.sh \\
        simulators/pegasus/scripts/probe_gt.py"
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PHYSICS_DT = 1.0 / 250.0
WARMUP_RENDERS = 120


def main() -> int:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    from simulators.pegasus.camera import load_intrinsics

    config_dir = Path("/tmp/dev/platform/robots/PEGASUS/config")
    intr = load_intrinsics(config_dir, resolution=(1440, 840))
    log(f"camera {intr.width}x{intr.height} fx={intr.fx:.1f}")

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    log("Isaac Sim up")

    from isaacsim.core.api import World

    from simulators.pegasus.camera import project, IRIS_SPAN_M
    from simulators.pegasus.drone import KinematicDrone
    from simulators.pegasus.scenes.outdoor import (
        load_outdoor_scene, scene_ground_z, scene_origin_xy,
    )

    world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT,
                  stage_units_in_meters=1.0)
    load_outdoor_scene(app, "skydome", sky="clear", progress=log)
    log("scene loaded")

    ground_z = scene_ground_z("skydome")
    ox, oy = scene_origin_xy("skydome")
    height = ground_z + 20.0
    chaser_xyz = (ox, oy, height)
    target_xyz = (ox + 20.0, oy, height)

    pegasus_root = Path("/tmp/dev/PegasusSimulator/extensions/pegasus.simulator")
    chaser = KinematicDrone(pegasus_root, "/World/chaser", intr,
                            position=chaser_xyz, yaw=0.0)
    target = KinematicDrone(pegasus_root, "/World/target", intr,
                            position=target_xyz, yaw=math.pi, with_camera=False)
    log("drones spawned")

    # ---- semantics: label the target so the annotator can find it -----------
    sem_report = {}
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    target_prim = stage.GetPrimAtPath("/World/target")

    for how in ("core_utils", "replicator", "usd_semantics"):
        try:
            if how == "core_utils":
                from isaacsim.core.utils.semantics import add_update_semantics
                add_update_semantics(target_prim, "drone")
            elif how == "replicator":
                import omni.replicator.core as rep
                with rep.get.prims(path_pattern="/World/target"):
                    rep.modify.semantics([("class", "drone")])
            else:
                from pxr import Semantics
                sem = Semantics.SemanticsAPI.Apply(target_prim, "Semantics")
                sem.CreateSemanticTypeAttr().Set("class")
                sem.CreateSemanticDataAttr().Set("drone")
            sem_report[how] = "ok"
            log(f"semantics via {how}: OK")
            break
        except Exception as exc:            # noqa: BLE001 - probing on purpose
            sem_report[how] = f"{type(exc).__name__}: {exc}"
            log(f"semantics via {how}: {type(exc).__name__}: {exc}")

    # ---- annotators --------------------------------------------------------
    cam = chaser._camera
    ann_report = {}
    for name, adder in (("bounding_box_2d_tight", "add_bounding_box_2d_tight_to_frame"),
                        ("semantic_segmentation", "add_semantic_segmentation_to_frame"),
                        ("distance_to_image_plane", "add_distance_to_image_plane_to_frame")):
        try:
            getattr(cam, adder)()
            ann_report[name] = "added"
            log(f"annotator {name}: added")
        except Exception as exc:            # noqa: BLE001
            ann_report[name] = f"{type(exc).__name__}: {exc}"
            log(f"annotator {name}: {type(exc).__name__}: {exc}")

    world.reset()
    for i in range(WARMUP_RENDERS):
        world.step(render=True)
    log("warm")

    # Point the chaser at the target and render a few more.
    yaw = chaser.look_at(target_xyz)
    for _ in range(6):
        world.step(render=True)

    frame = cam.get_current_frame()
    log(f"frame keys: {sorted(frame.keys())}")

    out = {"semantics": sem_report, "annotators": ann_report,
           "frame_keys": sorted(str(k) for k in frame.keys())}

    bb = frame.get("bounding_box_2d_tight")
    log(f"bbox payload type={type(bb)}")
    try:
        data = bb["data"] if isinstance(bb, dict) else bb
        log(f"bbox data: {data}")
        boxes = []
        for row in data:
            boxes.append([float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        out["boxes"] = boxes
        if boxes:
            x1, y1, x2, y2 = boxes[0]
            log(f"ANNOTATOR box  centre=({(x1 + x2) / 2:.1f}, {(y1 + y2) / 2:.1f}) "
                f"size=({x2 - x1:.1f} x {y2 - y1:.1f})")
    except Exception as exc:                # noqa: BLE001
        out["boxes_error"] = f"{type(exc).__name__}: {exc}"
        log(f"bbox decode failed: {type(exc).__name__}: {exc}")

    uv = project(intr, chaser_xyz, yaw, target_xyz)
    rng = math.dist(chaser_xyz, target_xyz)
    log(f"ANALYTIC centre=({uv[0]:.1f}, {uv[1]:.1f}) span={intr.pixel_span(IRIS_SPAN_M, rng):.1f}px")
    out["analytic"] = {"uv": [uv[0], uv[1]], "span_px": intr.pixel_span(IRIS_SPAN_M, rng)}

    # Save the frame so the box can be checked by eye.
    rgb = chaser.rgb()
    if rgb is not None:
        import imageio.v2 as imageio
        Path("/tmp/dev/pursuit").mkdir(parents=True, exist_ok=True)
        imageio.imwrite("/tmp/dev/pursuit/probe_frame.png", rgb)
        log("wrote /tmp/dev/pursuit/probe_frame.png")
        out["frame_mean"] = float(rgb.mean())

    Path("/tmp/dev/pursuit/probe_gt.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log("PROBE_OK")
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
