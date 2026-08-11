#!/usr/bin/env python3
"""Isaac Sim as a long-lived render server for the pursuit brain.

Boots once, loads a scene once, and then answers ``reset``/``step`` calls over a
unix socket for as long as it is left running. The brain (host side, where the
detector and its weights live) drives every pose; this process owns nothing but
the stage, the two aircraft and the camera.

    docker exec isaac-sim bash -c "cd /tmp/dev/dronedet && /isaac-sim/python.sh \\
        simulators/pegasus/scripts/pursuit_server.py --scene skydome"

Why a server and not a script: loading Rivermark costs 25 s warm and minutes
cold, and the brain is the half that is rewritten every few minutes. Keeping the
sim up means a brain restart costs a YOLO load, and one boot serves the whole
scenario matrix -- collect a dataset, fly a pursuit, fly fifty more.

## Ground truth comes from the renderer, not from the pose we asked for

``run_two_drone.py`` writes an *analytic* ground truth: it projects the target's
commanded position through a pinhole model. That is exact only if the frame that
comes back was rendered from exactly those poses, and it is not always -- with
both aircraft moving, the render pipeline is a frame behind the transform we
just authored, which at 20 fps and 3 m/s is tens of pixels of silent bias. A
label that wrong trains a detector to look next to the drone.

So the target carries a replicator semantic and the chaser's camera carries the
``bounding_box_2d_tight`` annotator: the box is measured on the same rendered
image the detector is handed, so whatever the pipeline's latency is, the label
and the pixels agree. The analytic projection is still computed and returned
alongside it -- the gap between the two *is* the render lag, which is worth
watching rather than assuming away.

## What one ``step`` means

Physics and rendering run at the same dt, and that dt is the capture interval:
one ``step`` is one control tick. There is nothing for a 250 Hz physics loop to
integrate here -- both aircraft are kinematic with gravity disabled, so the
brain owns the dynamics and PhysX only has to not interfere.

One tick is **rendered more than once**, though, and that is not wasted work.
Measured on this rig, a single ``world.step(render=True)`` returns an image of
the poses from *two* ticks ago: Isaac's render pipeline is asynchronous and runs
behind the transforms just authored. A hundred milliseconds of hidden,
unspecified sensor delay sitting inside a closed loop is the kind of thing that
makes a control law look worse than it is and makes the reason unfindable. Since
both aircraft are kinematic and their poses are held fixed across the extra
renders, flushing the pipeline costs frames per second and moves nothing --
after which latency is something an experiment *adds on purpose* (and can sweep)
rather than something the renderer imposes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The camera's first ~100 render callbacks produce nothing. Counted in rendered
# ticks, and both the annotator and the RGB buffer need it.
CAMERA_WARMUP_RENDERS = 120

CHASER_PATH = "/World/chaser"
TARGET_PATH = "/World/target"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="skydome",
                   choices=["rivermark", "skydome", "rivermark_props"])
    p.add_argument("--sky", default="clear", help="HDRI for the non-rivermark scenes")
    p.add_argument("--fps", type=float, default=20.0,
                   help="capture rate; one step advances 1/fps seconds")
    p.add_argument("--resolution", default="1440x840",
                   help="WxH. The default is an exact 2x of the platform "
                        "calibration's native 720x420 (a uniform scale, so the "
                        "field of view is unchanged)")
    p.add_argument("--socket", default="/tmp/dev/pursuit/sim.sock")
    p.add_argument("--pegasus-root",
                   default="/tmp/dev/PegasusSimulator/extensions/pegasus.simulator")
    p.add_argument("--pegasus-config", default=None)
    p.add_argument("--platform-root", default="/tmp/dev/platform",
                   help="container-side PEGASUS platform tree, as populated by "
                        "scripts/run_in_container.sh")
    p.add_argument("--load-timeout", type=float, default=None)
    p.add_argument("--cameras", default="nose", choices=["nose", "ring"],
                   help="'nose' is the single PEGASUS Iris camera every earlier "
                        "result was flown on. 'ring' is four wide cameras 90 "
                        "degrees apart -- 360 degree coverage, so the aircraft "
                        "never has to turn around to look for a target")
    p.add_argument("--ring-count", type=int, default=4)
    p.add_argument("--ring-hfov", type=float, default=None,
                   help="degrees; default is camera.RING_HFOV_DEG (96), which "
                        "leaves 6 degrees of overlap at each seam of a 4-ring")
    p.add_argument("--ring-resolution", default=None,
                   help="WxH per ring camera; default 1280x720")
    p.add_argument("--target-camera", action="store_true",
                   help="also render the target's own camera (for split-view "
                        "video); costs a second render product per frame")
    p.add_argument("--depth", action="store_true",
                   help="also return the chaser's depth AOV at the target box")
    p.add_argument("--render-ticks", type=int, default=5,
                   help="renders per step. Measured on this rig, not guessed: a "
                        "freshly authored pose is 36 px stale after 1 render, 18 "
                        "after 3, and under 1.5 px from 5 onward (a sweep is in "
                        "the git history of this file). Both aircraft are "
                        "kinematic and held still across the extra renders, so "
                        "this costs fps and moves nothing")
    p.add_argument("--verify-sync", action="store_true", default=True,
                   help="on boot, sweep the target and report the residual lag "
                        "between the rendered box and the analytic projection")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- gt

def drone_semantic_ids(payload) -> set:
    """Semantic ids the annotator is using for our target, from its own mapping.

    Not optional, and not obvious until it bites. ``bounding_box_2d_tight``
    reports **every** semantically-labelled prim in view, and NVIDIA's Rivermark
    ships its own semantics on the town's assets. Unioning all the rows -- which
    is correct in the empty skydome, where the target is the only labelled thing
    -- fuses the drone's box with a building's, and the result is a "ground
    truth" box 120 pixels away from the aircraft that stays 120 pixels away no
    matter how the render pipeline is flushed. Every label in the dataset would
    have been quietly wrong, and only in the scene with the interesting
    backgrounds.
    """
    info = payload.get("info") if isinstance(payload, dict) else None
    mapping = (info or {}).get("idToLabels") or {}
    ids = set()
    for key, val in mapping.items():
        label = val.get("class", "") if isinstance(val, dict) else str(val)
        if "drone" in str(label).lower():
            try:
                ids.add(int(key))
            except (TypeError, ValueError):
                continue
    return ids


def _union_box(rows, keep_ids):
    """Merge the target's annotator rows into one ``(x1, y1, x2, y2)``, or None.

    A referenced asset whose meshes each inherit the semantic produces several
    rows for the same aircraft, so a union across *its* rows is right. Rows
    belonging to anything else are dropped -- see :func:`drone_semantic_ids`.

    An empty ``keep_ids`` means **no box**, not "keep everything", and the
    distinction is the whole bug. Isaac's ``idToLabels`` describes only what is
    currently *in frame*, so the instant the drone leaves the picture its id
    vanishes from the mapping -- and a filter that reads "nothing to keep" as
    "no filtering needed" then unions all 469 of Rivermark's own labelled props
    (buildings, kerbs, lane markings) into one frame-sized box and calls it the
    target. The frames with no drone id are precisely the frames with no drone.
    Fail closed.
    """
    if not keep_ids:
        return None
    x1 = y1 = float("inf")
    x2 = y2 = float("-inf")
    n = 0
    for row in rows:
        try:
            if int(row[0]) not in keep_ids:
                continue
            bx1, by1, bx2, by2 = (float(row[1]), float(row[2]),
                                  float(row[3]), float(row[4]))
        except (IndexError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (bx1, by1, bx2, by2)):
            continue
        if bx2 <= bx1 or by2 <= by1:
            continue
        x1, y1 = min(x1, bx1), min(y1, by1)
        x2, y2 = max(x2, bx2), max(y2, by2)
        n += 1
    if n == 0:
        return None
    return [x1, y1, x2, y2]


class PursuitSim:
    """The stage, the two aircraft, and one rendered observation per step."""

    def __init__(self, args, log):
        self.args = args
        self.log = log
        self.fps = float(args.fps)
        self.dt = 1.0 / self.fps
        self.render_ticks = max(1, int(args.render_ticks))
        self.frame_idx = 0
        self.sim_t = 0.0

        from simulators.pegasus.camera import (NOSE_MOUNT, RING_HFOV_DEG,
                                               RING_RESOLUTION, load_intrinsics,
                                               ring_coverage_deg, ring_intrinsics,
                                               ring_mounts)

        self.ring = args.cameras == "ring"
        if self.ring:
            if args.ring_resolution:
                rw, rh = (int(v) for v in args.ring_resolution.lower().split("x"))
            else:
                rw, rh = RING_RESOLUTION
            self.intr = ring_intrinsics(rw, rh, args.ring_hfov or RING_HFOV_DEG)
            self.mounts = ring_mounts(int(args.ring_count))
            covered, overlap = ring_coverage_deg(self.intr, len(self.mounts))
            # A ring with a hole in it is a direction an intruder arrives from
            # unseen, and no summary statistic would ever show it -- the
            # engagements that failed would simply be the ones nobody could
            # explain. Refuse to boot instead.
            if overlap <= 0.0:
                raise SystemExit(
                    f"{len(self.mounts)} cameras of {self.intr.hfov_deg:.1f} deg "
                    f"cover {covered:.1f} deg of 360 -- a "
                    f"{-overlap:.1f} deg blind wedge at every seam. Raise "
                    f"--ring-hfov above {360.0 / len(self.mounts):.1f} or add cameras."
                )
            log(f"ring x{len(self.mounts)} {self.intr.width}x{self.intr.height} "
                f"fx={self.intr.fx:.1f} HFOV={self.intr.hfov_deg:.1f} "
                f"VFOV={self.intr.vfov_deg:.1f} -- covers {covered:.0f} deg "
                f"with {overlap:.1f} deg overlap per seam")
        else:
            config_dir = (Path(args.pegasus_config) if args.pegasus_config
                          else Path(args.platform_root)
                          / "robots/PEGASUS/config")
            w, h = args.resolution.lower().split("x")
            self.intr = load_intrinsics(config_dir, resolution=(int(w), int(h)))
            self.mounts = (NOSE_MOUNT,)
            log(f"camera {self.intr.width}x{self.intr.height} fx={self.intr.fx:.1f} "
                f"HFOV={self.intr.hfov_deg:.1f} VFOV={self.intr.vfov_deg:.1f}")

        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": True})
        log("Isaac Sim up")

        from isaacsim.core.api import World
        from simulators.pegasus.drone import KinematicDrone
        from simulators.pegasus.scenes.outdoor import (
            load_outdoor_scene, scene_ground_z, scene_origin_xy,
        )

        # Equal physics and rendering dt so one step() advances exactly one
        # capture interval. Both aircraft are kinematic with gravity disabled,
        # so there is nothing for a faster physics loop to integrate.
        self.world = World(physics_dt=self.dt, rendering_dt=self.dt,
                           stage_units_in_meters=1.0)
        self.scene_info = load_outdoor_scene(self.app, args.scene, sky=args.sky,
                                             load_timeout_s=args.load_timeout,
                                             progress=log)
        self.ground_z = scene_ground_z(args.scene)
        self.origin_xy = scene_origin_xy(args.scene)
        log(f"scene {args.scene}: ground z={self.ground_z} origin={self.origin_xy}")

        start = (self.origin_xy[0], self.origin_xy[1], self.ground_z + 20.0)
        root = Path(args.pegasus_root)
        self.chaser = KinematicDrone(root, CHASER_PATH, self.intr,
                                     position=start, yaw=0.0,
                                     mounts=self.mounts if self.ring else None)
        self.target = KinematicDrone(
            root, TARGET_PATH, self.intr,
            position=(start[0] + 20.0, start[1], start[2]), yaw=math.pi,
            with_camera=bool(args.target_camera))
        log(f"aircraft spawned ({len(self.chaser.cameras)} chaser camera(s): "
            f"{', '.join(self.chaser.cameras)})")

        self._label_target()
        self._add_annotators()

        self.world.reset()
        for i in range(CAMERA_WARMUP_RENDERS):
            self.world.step(render=True)
            if i % 40 == 0:
                log(f"  camera warm-up {i}/{CAMERA_WARMUP_RENDERS}")
        if self.chaser.rgb() is None:
            log("WARNING: chaser camera still produced nothing after warm-up")
        log("cameras warm")

    # -- stage setup --------------------------------------------------------

    def _label_target(self):
        """Attach a replicator semantic so the annotator can find the target.

        ``isaacsim.core.utils.semantics.add_update_semantics`` is gone in 6.0.1
        (the module survives in extsDeprecated without that function), so the
        replicator path is the one that works here.
        """
        import omni.replicator.core as rep
        with rep.get.prims(path_pattern=TARGET_PATH):
            rep.modify.semantics([("class", "drone")])
        self.log("target labelled class=drone")

    def _add_annotators(self):
        # Every ring camera gets its own tight-box annotator. Not optional: the
        # label has to be measured on the same rendered image the detector is
        # handed, and with a ring "the image" is four different images. A single
        # annotator on the nose camera would silently label only the quarter of
        # the sky the aircraft happens to be facing.
        for name, cam in self.chaser.cameras.items():
            cam.add_bounding_box_2d_tight_to_frame()
            if self.args.depth:
                cam.add_distance_to_image_plane_to_frame()
        self.log(f"annotators attached to {len(self.chaser.cameras)} chaser camera(s)")

    # -- rpc ----------------------------------------------------------------

    def info(self) -> dict:
        from simulators.pegasus.camera import IRIS_SPAN_M, ring_coverage_deg
        covered, overlap = ring_coverage_deg(self.intr, len(self.mounts))
        return {
            "ok": True,
            "intrinsics": self.intr.as_dict(),
            # The ring is described once, here, and the brain builds its whole
            # bearing model from it -- mount order included, because the wire
            # payload is the frames concatenated in exactly this order.
            "cameras": [{"name": m.name, "yaw_deg": m.yaw_deg} for m in self.mounts],
            "ring": bool(self.ring),
            "coverage_deg": round(covered, 1),
            "seam_overlap_deg": round(overlap, 1),
            "scene": self.scene_info,
            "scene_name": self.args.scene,
            "ground_z": self.ground_z,
            "origin_xy": list(self.origin_xy),
            "fps": self.fps,
            "dt": self.dt,
            "render_ticks": self.render_ticks,
            "sync": getattr(self, "sync_report", None),
            "target_span_m": IRIS_SPAN_M,
            "target_camera": bool(self.args.target_camera),
        }

    def set_sky(self, sky: str) -> dict:
        """Swap the HDRI on the dome light (lighting variety without a reload)."""
        from simulators.pegasus.scenes.outdoor import SKY_HDRI
        if sky not in SKY_HDRI:
            raise KeyError(f"unknown sky {sky!r}; choose from {sorted(SKY_HDRI)}")
        import omni.usd
        from pxr import UsdLux
        stage = omni.usd.get_context().get_stage()
        n = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.DomeLight):
                UsdLux.DomeLight(prim).CreateTextureFileAttr(SKY_HDRI[sky])
                n += 1
        for _ in range(3):
            self.world.step(render=True)
        return {"ok": True, "sky": sky, "domes": n}

    def _place(self, chaser: dict, target: dict) -> None:
        self.chaser.set_pose(chaser["xyz"], float(chaser.get("yaw", 0.0)))
        self.target.set_pose(target["xyz"], float(target.get("yaw", 0.0)))

    def reset(self, chaser: dict, target: dict, settle: int = 4) -> tuple:
        """Place both aircraft and render until the pipeline shows the new pose.

        ``settle`` renders are not decoration: the annotator and the RGB buffer
        both lag the transform we just authored, so the first frame after a
        teleport shows the previous episode unless we flush it.
        """
        self._place(chaser, target)
        for _ in range(max(self.render_ticks, int(settle))):
            self.world.step(render=True)
        self.frame_idx = 0
        self.sim_t = 0.0
        return self._observe(chaser, target)

    def step(self, chaser: dict, target: dict) -> tuple:
        self._place(chaser, target)
        self._flush()
        self.frame_idx += 1
        self.sim_t += self.dt
        return self._observe(chaser, target)

    def _flush(self) -> None:
        """Render until the image shows the poses that were just authored.

        A pose written between two ``world.step`` calls does not appear in that
        step's render at all -- it lands one step boundary later, and then has to
        walk out through a render pipeline that is itself a couple of frames
        deep. Rendering with the aircraft held still is free of consequences
        (they are kinematic; nothing integrates) and it is the only way to make
        "the frame" and "the pose" mean the same instant.
        """
        for _ in range(self.render_ticks):
            self.world.step(render=True)

    def set_render_ticks(self, n: int) -> dict:
        self.render_ticks = max(1, int(n))
        return {"ok": True, "render_ticks": self.render_ticks}

    def measure_sync(self, n: int = 8) -> dict:
        """Sweep the target and report how far the rendered box lags the poses.

        Run once at boot so the number is in the log rather than in somebody's
        head. A residual above a pixel or two means ``--render-ticks`` is too low
        for this machine, and every downstream measurement inherits the error.
        """
        z = self.ground_z + 20.0
        chaser = {"xyz": [self.origin_xy[0], self.origin_xy[1], z], "yaw": 0.0}
        errs = []
        self.reset(chaser, {"xyz": [self.origin_xy[0] + 30.0, self.origin_xy[1], z],
                            "yaw": math.pi})
        for i in range(n):
            tgt = {"xyz": [self.origin_xy[0] + 30.0, self.origin_xy[1] + 0.6 * i, z],
                   "yaw": math.pi}
            header, _ = self.step(chaser, tgt)
            gt = header["gt"]
            if gt.get("uv") and gt.get("analytic_uv"):
                errs.append(abs(gt["uv"][0] - gt["analytic_uv"][0]))
        out = {"render_ticks": self.render_ticks, "n": len(errs),
               "max_px": round(max(errs), 2) if errs else None,
               "mean_px": round(sum(errs) / len(errs), 2) if errs else None}
        self.log(f"sync check: rendered box vs analytic projection {out}")
        return out

    def _camera_view(self, mount, chaser_xyz, chaser_yaw, target_xyz) -> dict:
        """What one camera measured and what geometry says it should have.

        Both, always, and never one without the other. The rendered box is the
        honest label -- it is measured on the pixels the detector is handed --
        and the pinhole projection is the independent check that says whether
        that box is on the drone or on a car park. With a ring the check earns
        its place twice over, because a mount rotation is exactly the kind of
        sign error that produces a perfectly plausible label on the wrong
        camera.
        """
        from simulators.pegasus.camera import (IRIS_SPAN_M, in_frame,
                                               mount_position, project)

        cam = self.chaser.cameras[mount.name]
        frame = cam.get_current_frame()

        bbox = None
        occlusion = None
        n_rows = n_drone_rows = 0
        raw = frame.get("bounding_box_2d_tight")
        rows = raw.get("data") if isinstance(raw, dict) else raw
        keep = drone_semantic_ids(raw)
        if rows is not None and len(rows):
            n_rows = len(rows)
            mine = [r for r in rows if keep and int(r[0]) in keep]
            n_drone_rows = len(mine)
            bbox = _union_box(rows, keep)
            try:
                occlusion = float(max(r[5] for r in mine)) if mine else None
            except (IndexError, TypeError, ValueError):
                occlusion = None

        cpos = (mount_position(chaser_xyz, chaser_yaw, mount) if self.ring
                else tuple(chaser_xyz))
        cam_rng = math.dist(cpos, tuple(target_xyz))
        uv = project(self.intr, cpos, chaser_yaw + mount.yaw, target_xyz)

        view = {
            "camera": mount.name,
            "mount_yaw_deg": mount.yaw_deg,
            "bbox": None if bbox is None else [round(v, 2) for v in bbox],
            "occlusion": occlusion,
            "analytic_uv": None if uv is None else [round(uv[0], 2), round(uv[1], 2)],
            "analytic_in_frame": bool(in_frame(self.intr, uv)),
            "analytic_span_px": round(self.intr.pixel_span(IRIS_SPAN_M, cam_rng), 2),
            "ann_rows": n_rows,
            "ann_drone_rows": n_drone_rows,
        }
        if bbox is not None:
            bu = (bbox[0] + bbox[2]) / 2
            bv = (bbox[1] + bbox[3]) / 2
            span = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            view.update(uv=[round(bu, 2), round(bv, 2)],
                        span_px=round(span, 2), visible=True)
            if uv is not None:
                view["label_gap_px"] = round(math.hypot(bu - uv[0], bv - uv[1]), 2)
                view["span_ratio"] = round(
                    span / max(1e-6, view["analytic_span_px"]), 3)
            else:
                view["label_gap_px"] = None
                view["span_ratio"] = None
        else:
            view.update(uv=None, span_px=None, visible=False,
                        label_gap_px=None, span_ratio=None)

        if self.args.depth:
            depth = frame.get("distance_to_image_plane")
            d = depth.get("data") if isinstance(depth, dict) else depth
            if d is not None and bbox is not None:
                import numpy as np
                arr = np.asarray(d)
                u = int(round((bbox[0] + bbox[2]) / 2))
                v = int(round((bbox[1] + bbox[3]) / 2))
                if 0 <= v < arr.shape[0] and 0 <= u < arr.shape[1]:
                    val = float(arr[v, u])
                    view["depth_m"] = None if not math.isfinite(val) else round(val, 3)
        return view

    def _best_view(self, views: list):
        """Which camera owns the target this frame.

        In the seam overlap the same drone is genuinely in two images at once,
        so "which camera sees it" has to be answered rather than assumed, and it
        has to be answered the same way every frame or the top-level ground
        truth flickers between two boxes that are both right.

        The rule is *most centred*: whichever camera has the target furthest
        from its own frame edge, measured as a fraction of the half-frame. That
        is the view whose box is least likely to be clipped, and it changes hands
        exactly once as a target crosses a seam -- at the bisector, where the two
        cameras agree.
        """
        def centrality(v):
            uv = v.get("uv") or v.get("analytic_uv")
            if uv is None:
                return -1.0
            du = abs(uv[0] - self.intr.cx) / (0.5 * self.intr.width)
            dv = abs(uv[1] - self.intr.cy) / (0.5 * self.intr.height)
            return -max(du, dv)

        measured = [v for v in views if v.get("visible")]
        pool = measured or [v for v in views if v.get("analytic_in_frame")]
        if not pool:
            return None
        return max(pool, key=centrality)

    def _observe(self, chaser: dict, target: dict) -> tuple:
        cx, cy, cz = (float(v) for v in chaser["xyz"])
        tx, ty, tz = (float(v) for v in target["xyz"])
        yaw = float(chaser.get("yaw", 0.0))
        rng = math.dist((cx, cy, cz), (tx, ty, tz))

        views = []
        cam_meta = []
        parts = []
        for m in self.mounts:
            views.append(self._camera_view(m, (cx, cy, cz), yaw, (tx, ty, tz)))
            rgb = self.chaser.rgb(m.name)
            cam_meta.append({"name": m.name, "yaw_deg": m.yaw_deg,
                             "shape": None if rgb is None else list(rgb.shape),
                             "gt": views[-1]})
            if rgb is not None:
                parts.append(rgb.tobytes())

        gt = {
            "frame": self.frame_idx,
            "t": round(self.sim_t, 4),
            "chaser_xyz": [round(cx, 4), round(cy, 4), round(cz, 4)],
            "chaser_yaw": round(yaw, 5),
            "target_xyz": [round(tx, 4), round(ty, 4), round(tz, 4)],
            "target_yaw": round(float(target.get("yaw", 0.0)), 5),
            "range_m": round(rng, 4),
        }
        best = self._best_view(views)
        # The top-level ground truth is the owning camera's, verbatim, so every
        # tool written against the single-camera rig keeps working unchanged --
        # with one field added saying which camera it came from.
        blank = {"camera": None, "bbox": None, "uv": None, "span_px": None,
                 "visible": False, "occlusion": None, "analytic_uv": None,
                 "analytic_in_frame": False, "label_gap_px": None,
                 "span_ratio": None, "ann_rows": 0, "ann_drone_rows": 0,
                 "analytic_span_px": round(self.intr.pixel_span(0.47, rng), 2)}
        gt.update({k: v for k, v in (best or blank).items()
                   if k != "mount_yaw_deg"})
        if self.ring:
            gt["per_camera"] = {v["camera"]: v for v in views}
            gt["seen_by"] = [v["camera"] for v in views if v.get("visible")]

        header = {"ok": True, "gt": gt, "ring": bool(self.ring),
                  "cameras": cam_meta,
                  "frame_shape": cam_meta[0]["shape"] if cam_meta else None}
        payload = b"".join(parts)
        if self.args.target_camera:
            trgb = self.target.rgb()
            if trgb is not None:
                header["target_frame_shape"] = list(trgb.shape)
                payload = payload + trgb.tobytes()
        return header, payload

    def close(self):
        self.app.close()


# ------------------------------------------------------------------- serving

def serve(sim: PursuitSim, path: str, log) -> None:
    from simulators.pegasus.pursuit_proto import recv_msg, send_msg

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.chmod(path, 0o777)          # the host runs as a different uid
    srv.listen(1)
    log(f"SERVER_READY listening on {path}")

    handlers = {
        "info": lambda kw: (sim.info(), b""),
        "reset": lambda kw: sim.reset(kw["chaser"], kw["target"],
                                      int(kw.get("settle", 4))),
        "step": lambda kw: sim.step(kw["chaser"], kw["target"]),
        "set_sky": lambda kw: (sim.set_sky(kw["sky"]), b""),
        "set_render_ticks": lambda kw: (sim.set_render_ticks(kw["n"]), b""),
        "measure_sync": lambda kw: (sim.measure_sync(int(kw.get("n", 8))), b""),
        "ping": lambda kw: ({"ok": True}, b""),
    }

    stop = False
    while not stop:
        conn, _ = srv.accept()
        log("client connected")
        try:
            while True:
                header, _ = recv_msg(conn)
                cmd = header.get("cmd")
                if cmd in ("bye", "disconnect"):
                    log("client said bye")
                    break
                if cmd == "shutdown":
                    send_msg(conn, {"ok": True})
                    stop = True
                    break
                fn = handlers.get(cmd)
                if fn is None:
                    send_msg(conn, {"error": f"unknown command {cmd!r}"})
                    continue
                try:
                    reply, payload = fn(header)
                except Exception as exc:                      # noqa: BLE001
                    log(f"command {cmd!r} failed: {traceback.format_exc()}")
                    send_msg(conn, {"error": f"{type(exc).__name__}: {exc}"})
                    continue
                send_msg(conn, reply, payload)
        except (ConnectionError, OSError) as exc:
            # A brain that crashes or is killed mid-episode must not take the
            # simulator with it -- that is the whole point of the split.
            log(f"client dropped ({type(exc).__name__}: {exc}); waiting for the next")
        finally:
            conn.close()
    srv.close()
    if os.path.exists(path):
        os.unlink(path)
    log("SERVER_DONE")


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    sim = PursuitSim(args, log)
    sim.sync_report = sim.measure_sync() if args.verify_sync else None
    try:
        serve(sim, args.socket, log)
    finally:
        sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
