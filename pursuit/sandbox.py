#!/usr/bin/env python3
"""The pursuit loop with the renderer replaced by arithmetic.

Isaac Sim renders five frames per control tick to keep the image honest, which
puts one 45-second episode at roughly a minute. Tuning a guidance law against
that is hopeless: the interesting sweeps are hundreds of episodes wide, and a
law is not finished until it survives all of them.

Nothing about the *control* problem needs a renderer, though. The chaser's
sensor produces a bearing and a pixel span, and both are exactly computable from
the two aircraft's poses -- so this module runs the identical
:class:`~pursuit.guidance.PursuitGuidance`, :class:`~pursuit.dynamics.Airframe`
and :class:`~pursuit.evader.Evader` against a synthetic camera and gets through
the same episode in about a millisecond. Roughly fifty thousand times faster,
and every guidance bug found here is a real one.

What it deliberately does **not** model is the thing it cannot: whether a YOLO
finds a 9-pixel drone against a particular sky. That is what the Isaac loop is
for, and the division of labour is the point --

* **sandbox** answers *given a working detector, does the intercept converge* --
  swept over evasion policies, geometries, sensor noise, dropout and latency;
* **Isaac** answers *does the detector work*, on real rendered pixels;

and a failure in the full system is attributable to one or the other because
each has been measured on its own.

The synthetic camera is not generous. It reproduces the real field of view
(including this camera's off-centre principal point, so up and down are not
symmetric), refuses to report a target outside it, refuses to report one whose
pixel span is below a detection floor, and applies whatever dropout, pixel noise,
span bias and latency it is configured with.

    .venv/bin/python -m pursuit.sandbox --suite full
    .venv/bin/python -m pursuit.sandbox --suite full --sweep nav_gain=2,3,4,5,6
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pursuit.dynamics import Airframe, Limits
from pursuit.episode import (VERTICAL_AUTHORITY_RATIO, EpisodeResult,
                             ScenarioConfig, _closure_rate, pass_geometry,
                             place_engagement, segment_cpa)
from pursuit.evader import LADDER, POLICIES, EvaderConfig, evader_limits, make_evader
from pursuit.geometry import (Intrinsics, offaxis_scale, span_from_range,
                              world_to_body)
from pursuit.guidance import HIT, GuidanceConfig, PursuitGuidance
from pursuit.perception import Box, Perception, TrackerConfig

# The rig's real camera: PEGASUS' 720x420 calibration at the 2x render size.
SIM_INTRINSICS = Intrinsics(width=1440, height=840, fx=921.8145952785566,
                            fy=923.9695163260498, cx=691.6137045337061,
                            cy=257.22911647658873)
IRIS_SPAN_M = 0.47


class SyntheticCamera:
    """Poses in, boxes out -- the geometry of the real camera without the pixels.

    Args:
        intr: Camera model.
        span_m: Physical span of the target.
        min_span_px: Pixel span below which the target is simply not reported.
            Every detector has such a floor and pretending otherwise is the
            easiest way to build a guidance law that only works on targets it
            can already see. 4 px is about where a P2 detector stops.
        dropout: Probability a visible target is missed on any given frame.
        noise_px: Standard deviation of box-centre noise.
        span_noise: Fractional standard deviation on the reported span, which is
            what makes the monocular range wobble.
        span_bias: Systematic factor on the reported span. Measured on the rig,
            the rendered Iris subtends about 8 percent less than its rotor-tip
            span because it is rarely seen broadside, so its monocular range
            reads long by the same amount -- ``0.92`` reproduces that.
        latency_frames: Frames of delay between the geometry and the report.
        edge_margin_px: Treat the outer band of the frame as not-detected. A box
            half off the edge is clipped, so its centre and span are both wrong,
            and a detector's recall there is poor in any case.
    """

    def __init__(self, intr: Intrinsics, span_m: float = IRIS_SPAN_M,
                 min_span_px: float = 4.0, dropout: float = 0.0,
                 noise_px: float = 0.0, span_noise: float = 0.0,
                 span_bias: float = 1.0, latency_frames: int = 0,
                 edge_margin_px: float = 8.0, seed: int = 0) -> None:
        self.intr = intr
        self.span_m = float(span_m)
        self.min_span_px = float(min_span_px)
        self.dropout = float(dropout)
        self.noise_px = float(noise_px)
        self.span_noise = float(span_noise)
        self.span_bias = float(span_bias)
        self.latency_frames = int(latency_frames)
        self.edge = float(edge_margin_px)
        self.rng = np.random.default_rng(seed)
        self._queue: List[Optional[Box]] = []

    def reset(self) -> None:
        self._queue = []

    def observe(self, chaser: Airframe, target: Airframe) -> tuple:
        """Return ``(boxes, gt)`` for the current geometry."""
        box, gt = self._project(chaser, target)
        if self.latency_frames > 0:
            self._queue.append(box)
            box = self._queue.pop(0) if len(self._queue) > self.latency_frames else None
        return ([] if box is None else [box]), gt

    def _project(self, chaser: Airframe, target: Airframe):
        dx = target.xyz[0] - chaser.xyz[0]
        dy = target.xyz[1] - chaser.xyz[1]
        dz = target.xyz[2] - chaser.xyz[2]
        rng = math.sqrt(dx * dx + dy * dy + dz * dz)
        fwd, left, up = world_to_body(chaser.yaw, dx, dy, dz)
        gt = {"range_m": rng, "visible": False, "uv": None, "span_px": None,
              "bbox": None}
        if fwd <= 0.05:
            return None, gt

        u = self.intr.cx - self.intr.fx * (left / fwd)
        v = self.intr.cy - self.intr.fy * (up / fwd)
        # A pinhole stretches an off-axis object by sec^2, and the renderer
        # duly does: measured on the rig, a target at a fixed 40 m grows from
        # 8 px on the boresight to 15 px at 43 degrees off it. Leaving that out
        # here made the sandbox quietly disagree with the thing it exists to
        # predict -- and, worse, it hid the matching correction missing from
        # the monocular range, because the two errors cancelled in simulation
        # and only one of them existed in Isaac.
        span = span_from_range(self.intr, rng, self.span_m) * offaxis_scale(
            self.intr, u, v)
        in_frame = (self.edge <= u < self.intr.width - self.edge
                    and self.edge <= v < self.intr.height - self.edge)
        gt.update(uv=[u, v], span_px=span, visible=bool(in_frame),
                  bbox=[u - span / 2, v - span / 8, u + span / 2, v + span / 8])
        if not in_frame or span < self.min_span_px:
            return None, gt
        if self.dropout > 0.0 and self.rng.random() < self.dropout:
            return None, gt

        du = dv = 0.0
        if self.noise_px > 0.0:
            du, dv = self.rng.normal(0.0, self.noise_px, 2)
        s = span * self.span_bias
        if self.span_noise > 0.0:
            s *= max(0.25, 1.0 + self.rng.normal(0.0, self.span_noise))
        return Box(u + du - s / 2, v + dv - s / 8,
                   u + du + s / 2, v + dv + s / 8, 1.0, "drone"), gt


RING_INTRINSICS = Intrinsics(width=2048, height=704, fx=922.013741360988,
                             fy=922.013741360988, cx=1024.0, cy=352.0)
"""One camera of the four-camera ring, matching
``simulators.pegasus.camera.ring_intrinsics()`` at its defaults. Restated rather
than imported so the fast loop stays free of anything that touches Isaac, and
checked against the real one by ``test_ring.py``."""


class SyntheticRing:
    """The camera ring's geometry without the pixels.

    Same job as :class:`SyntheticCamera` -- poses in, labels out, an episode in
    a millisecond -- with the one thing the ring adds: a target inside a seam is
    reported by **both** cameras that can see it, separately and with
    independent noise. That duplicate is not an artefact to be tidied away here;
    it is exactly what the real ring produces, and the fusion that removes it is
    the part most likely to be wrong. Modelling it means the merge is exercised
    tens of thousands of times in the fast loop rather than a handful of times
    against a renderer.
    """

    def __init__(self, ring, span_m: float = IRIS_SPAN_M,
                 min_span_px: float = 3.0, edge_margin_px: float = 6.0,
                 seed: int = 0, **oracle_kw) -> None:
        from pursuit.ring import RingOracle

        self.ring = ring
        self.intr = ring.cameras[0].intr
        self.span_m = float(span_m)
        self.min_span_px = float(min_span_px)
        self.edge = float(edge_margin_px)
        self.oracle = RingOracle(ring, seed=seed, **oracle_kw)

    def reset(self) -> None:
        self.oracle.reset()

    def observe(self, chaser: Airframe, target: Airframe) -> tuple:
        """``([], gt)`` -- the ring's detections are made by the oracle per camera."""
        dx = target.xyz[0] - chaser.xyz[0]
        dy = target.xyz[1] - chaser.xyz[1]
        dz = target.xyz[2] - chaser.xyz[2]
        rng = math.sqrt(dx * dx + dy * dy + dz * dz)
        fwd, left, up = world_to_body(chaser.yaw, dx, dy, dz)

        per = {}
        best = None
        for cam in self.ring.cameras:
            c, s = math.cos(-cam.mount_yaw), math.sin(-cam.mount_yaw)
            f2 = fwd * c - left * s
            l2 = fwd * s + left * c
            if f2 <= 0.05:
                continue
            u = self.intr.cx - self.intr.fx * (l2 / f2)
            v = self.intr.cy - self.intr.fy * (up / f2)
            if not (self.edge <= u < self.intr.width - self.edge
                    and self.edge <= v < self.intr.height - self.edge):
                continue
            span = span_from_range(self.intr, rng, self.span_m) * offaxis_scale(
                self.intr, u, v)
            view = {"camera": cam.name, "range_m": rng, "uv": [u, v],
                    "span_px": span, "visible": span >= self.min_span_px,
                    "bbox": ([u - span / 2, v - span / 8,
                              u + span / 2, v + span / 8]
                             if span >= self.min_span_px else None)}
            per[cam.name] = view
            k = -max(abs(u - self.intr.cx) / (0.5 * self.intr.width),
                     abs(v - self.intr.cy) / (0.5 * self.intr.height))
            if best is None or k > best[0]:
                best = (k, view)

        gt = {"range_m": rng, "per_camera": per,
              "visible": bool(best and best[1]["visible"]),
              "uv": best[1]["uv"] if best else None,
              "span_px": best[1]["span_px"] if best else None,
              "bbox": best[1]["bbox"] if best else None,
              "camera": best[1]["camera"] if best else None,
              "seen_by": [k for k, v in per.items() if v["visible"]]}
        return [], gt


def run_episode(sc: ScenarioConfig, gcfg: GuidanceConfig, ecfg: EvaderConfig,
                camera, dt: float = 0.05,
                tracker_cfg: Optional[TrackerConfig] = None,
                trace: bool = False) -> EpisodeResult:
    """Fly one scenario headless. Mirrors :meth:`pursuit.episode.Episode.run`."""
    intr = camera.intr
    ecfg = replace(ecfg, speed=sc.evader_speed)

    cxyz, cyaw, txyz, tyaw, aim = place_engagement(
        sc, (0.0, 0.0), 0.0, ecfg.altitude_band[0])
    chaser = Airframe(xyz=cxyz, yaw=cyaw, ground_z=0.0)
    v_xy = ecfg.speed * sc.speed_advantage
    chaser.limits = Limits(
        max_speed_xy=v_xy,
        max_speed_z=max(4.0, ecfg.climb_mps * sc.speed_advantage,
                        v_xy / VERTICAL_AUTHORITY_RATIO),
        max_accel_xy=14.0 * sc.speed_advantage, max_accel_z=7.0,
        max_yaw_rate=2.5, max_yaw_accel=10.0, min_agl=2.0)

    target = Airframe(xyz=txyz, yaw=tyaw, ground_z=0.0)
    target.limits = evader_limits(ecfg)
    b = math.atan2(txyz[1] - cxyz[1], txyz[0] - cxyz[0])
    evader = make_evader(sc.policy, sc.seed, 0.0, ecfg, heading0=b,
                         centre_xy=(cxyz[0], cxyz[1]))
    if aim is not None:
        evader.arm_ingress(aim, sc.transit_speed or ecfg.speed,
                           commit=sc.strike_commit, evade=sc.strike_evade)
    defend_xyz = None
    if sc.defend_xy is not None:
        defend_xyz = (float(sc.defend_xy[0]), float(sc.defend_xy[1]),
                      sc.defend_height_m)

    guidance = PursuitGuidance(intr, chaser.limits, camera.span_m, gcfg)
    if isinstance(camera, SyntheticRing):
        from pursuit.ring import RingPerception
        perception = RingPerception(camera.ring, oracle=camera.oracle,
                                    tracker_cfg=tracker_cfg)
    else:
        perception = Perception(_PassThrough(), intr, tracker_cfg or TrackerConfig())
    camera.reset()

    res = EpisodeResult(name=sc.name, policy=sc.policy, seed=sc.seed,
                        config=asdict(sc))
    n_visible = n_detected = n_tracked = 0
    range_errs: List[float] = []
    rows = []
    t = 0.0

    for idx in range(int(sc.max_seconds / dt)):
        boxes, gt = camera.observe(chaser, target)
        est = perception.step(None, idx, t,
                              gt if isinstance(camera, SyntheticRing)
                              else {"boxes": boxes},
                              ego_yaw=chaser.yaw, ego_speed=chaser.speed)
        gs = guidance.step(t, dt, chaser.xyz, chaser.yaw, chaser.vel, est)

        if gt["visible"]:
            n_visible += 1
            if est.valid and est.source == "detector":
                n_detected += 1
        if est.valid:
            n_tracked += 1
            if res.acquire_frame is None and guidance.confirmed:
                res.acquire_frame, res.acquire_time_s = idx, round(t, 3)
            if est.source == "detector" and gs.range_est is not None:
                range_errs.append(abs(gs.range_est - gt["range_m"]))
        if trace:
            rows.append({"f": idx, "t": round(t, 2), "mode": gs.mode,
                         "r": round(gt["range_m"], 2),
                         "re": None if gs.range_est is None else round(gs.range_est, 2),
                         "src": est.source, "bore": gs.boresight_deg,
                         "lat": gs.lateral_speed, "close": gs.closing_speed,
                         "los": gs.los_rate,
                         "cmd": [round(v, 2) for v in gs.command.as_tuple()],
                         "cx": [round(v, 1) for v in chaser.xyz],
                         "tx": [round(v, 1) for v in target.xyz]})

        if not evader.revealed:
            if guidance.confirmed or (
                    sc.reveal_range_m > 0.0
                    and math.dist(chaser.xyz, target.xyz) <= sc.reveal_range_m):
                evader.reveal(t)
                if res.reveal_time_s is None:
                    res.reveal_time_s = round(t, 3)

        if defend_xyz is not None:
            d_asset = math.dist(target.xyz, defend_xyz)
            res.min_asset_range_m = (d_asset if res.min_asset_range_m is None
                                     else min(res.min_asset_range_m, d_asset))
            if d_asset <= sc.strike_radius_m:
                res.struck_asset = True
                res.success = False
                res.outcome = "target_struck"
                break

        p0, q0 = chaser.xyz, target.xyz
        chaser.step(gs.command, dt)
        target.step(evader.command(t, target, chaser.xyz), dt)
        cpa, _ = segment_cpa(p0, chaser.xyz, q0, target.xyz)
        res.miss_distance_m = min(res.miss_distance_m, cpa)
        res.min_sampled_range_m = min(res.min_sampled_range_m, gt["range_m"])
        t += dt
        res.frames = idx + 1
        if cpa <= sc.hit_radius_m:
            res.success = True
            res.outcome = "intercept"
            res.time_to_intercept_s = round(t, 3)
            g = pass_geometry(chaser.xyz, chaser.vel, target.xyz, target.vel,
                              chaser.yaw)
            res.pass_cpa_m = round(g["cpa_m"], 4)
            res.pass_along_m = round(g["along_m"], 4)
            res.pass_lateral_m = round(g["lateral_m"], 4)
            res.pass_vertical_m = round(g["vertical_m"], 4)
            if defend_xyz is not None:
                remaining = math.dist(target.xyz, defend_xyz) - sc.strike_radius_m
                closing = _closure_rate(target.xyz, target.vel, defend_xyz)
                if closing > 0.1:
                    res.strike_margin_s = round(max(0.0, remaining) / closing, 2)
            break
    else:
        res.outcome = "timeout"

    res.detect_rate = round(n_detected / max(1, n_visible), 4)
    res.track_rate = round(n_tracked / max(1, res.frames), 4)
    res.mean_range_err_m = (round(float(np.mean(range_errs)), 3)
                            if range_errs else None)
    res.miss_distance_m = round(res.miss_distance_m, 4)
    res.min_sampled_range_m = round(res.min_sampled_range_m, 4)
    # Guard on "timeout" so a terminal outcome recorded inside the loop is not
    # overwritten. `target_struck` and `never_acquired` can both be true at
    # once, and the one that matters is that the asset was hit -- reporting the
    # cause instead of the consequence made a lost building look like a quiet
    # non-event, and hid it in the exact runs where the system failed worst.
    if not res.success and res.outcome == "timeout" and n_tracked == 0:
        res.outcome = "never_acquired"
    if res.struck_asset and n_tracked == 0:
        res.note = "struck without ever being acquired"
    if trace:
        res.stage_ms["trace"] = rows
    return res


class _PassThrough:
    """Feeds the synthetic camera's boxes through the real :class:`Perception`.

    Going through ``Perception`` rather than around it means the sandbox
    exercises the same tracker, the same gating and the same coasting the Isaac
    loop does -- so a tracker bug is caught in the fast loop instead of in the
    slow one.
    """

    name = "synthetic"
    needs_frame = False

    def detect(self, frame, idx, gt=None):
        return list((gt or {}).get("boxes", []))


# ------------------------------------------------------------------- suites

def build_suite(name: str, base: ScenarioConfig) -> List[ScenarioConfig]:
    out: List[ScenarioConfig] = []
    if name == "smoke":
        return [replace(base, name="straight", policy="straight", seed=1)]
    if name == "core":
        return [replace(base, name=p, policy=p, seed=1)
                for p in ("straight", "flee", "weave", "break_turn", "jink")]
    if name == "full":
        for p in POLICIES:
            for seed in (1, 2, 3):
                out.append(replace(base, name=f"{p}-s{seed}", policy=p, seed=seed))
        for bearing in (-55.0, -30.0, 30.0, 55.0):
            out.append(replace(base, name=f"bear{bearing:+.0f}", policy="flee",
                               seed=4, start_bearing_deg=bearing))
        for rng in (25.0, 60.0, 80.0):
            out.append(replace(base, name=f"rng{rng:.0f}", policy="weave",
                               seed=5, start_range_m=rng))
        for spd in (10.0, 12.0, 13.0):
            out.append(replace(base, name=f"spd{spd:.0f}", policy="break_turn",
                               seed=6, evader_speed=spd))
        for elev in (-20.0, 20.0):
            out.append(replace(base, name=f"elev{elev:+.0f}", policy="climb_flee",
                               seed=7, start_elevation_deg=elev))
        return out
    if name == "ladder":
        # A difficulty ramp, easiest first. Each rung adds one thing the seeker
        # has to cope with: cross-range motion, then oscillation, then two axes
        # at once, then a sustained turn, then a committed reversal, then
        # unpredictability, then all of it together against a target that is
        # actively trying to break the lock.
        for i, pol in enumerate(LADDER):
            out.append(replace(base, name=f"L{i + 1}-{pol}", policy=pol, seed=1))
        return out
    if name == "showcase":
        # One scenario per evader policy, plus the geometries that exercise
        # acquisition rather than closure. Built for *watching*: every distinct
        # behaviour the system has, once each, instead of three seeds of the
        # first thing in the list.
        for p in POLICIES:
            out.append(replace(base, name=p, policy=p, seed=1))
        out.append(replace(base, name="offset+55", policy="flee", seed=4,
                           start_bearing_deg=55.0))
        out.append(replace(base, name="offset-30", policy="weave", seed=4,
                           start_bearing_deg=-30.0))
        out.append(replace(base, name="high+20", policy="jink", seed=7,
                           start_elevation_deg=20.0))
        out.append(replace(base, name="fast12", policy="break_turn", seed=6,
                           evader_speed=12.0))
        return out
    if name == "stress":
        # Every policy, every seed, from every direction: the matrix a law has to
        # clear before "it works" means anything.
        for p in POLICIES:
            for seed in range(1, 5):
                for bearing in (-40.0, 0.0, 40.0):
                    out.append(replace(base, name=f"{p}-s{seed}-b{bearing:+.0f}",
                                       policy=p, seed=seed,
                                       start_bearing_deg=bearing))
        return out
    if name in ("ingress", "ingress-wide"):
        return build_ingress_suite(base, wide=(name == "ingress-wide"))
    if name == "approach":
        return build_approach_suite(base)
    if name == "defend":
        return build_defend_suite(base)
    if name in ("city", "city-hard", "city-all", "city-astern"):
        from pursuit.city import build_city_suite
        if name == "city-all":
            return (build_city_suite(base) + build_city_suite(base, hard=True))
        if name == "city-astern":
            # Eight arrivals, every one of them dead astern. Nothing here is
            # about difficulty -- it is the geometry a single forward camera
            # cannot see at all, flown deliberately rather than sampled.
            return build_city_suite(base, n=8, nose_relative_deg=180.0,
                                    label="astern")
        return build_city_suite(base, hard=(name == "city-hard"))
    if name == "mission":
        # Everything that resembles a real engagement, in one run: an intruder
        # arriving from twelve directions, plus the four relative-motion cases
        # named explicitly (inbound, outbound, left-to-right, right-to-left),
        # plus the difficulty ladder. This is the suite to record.
        return (build_ingress_suite(base)
                + build_approach_suite(base)
                + [replace(base, name=f"L{i + 1}-{p}", policy=p, seed=40 + i,
                           start_range_m=35.0, max_seconds=30.0)
                   for i, p in enumerate(LADDER)])
    if name in POLICIES:
        return [replace(base, name=f"{name}-s{s}", policy=name, seed=s)
                for s in (1, 2, 3)]
    raise ValueError(f"unknown suite {name!r}")


# How an intruder arrives, as (label, bearing°, elevation°, aim-lateral m,
# aim-vertical m). The bearings are all outside the camera's ±38.5° horizontal
# half-angle and the elevations outside its +15.5°/-32° vertical one, so every
# one of these scenarios *begins with an empty frame* -- which is the whole
# point of the suite.
INGRESS_ENTRIES = (
    # label        bearing°  elev°   right m   up m  ahead m
    ("left",         70.0,    0.0,    18.0,    0.0,   28.0),
    ("right",       -70.0,    0.0,   -18.0,    0.0,   28.0),
    ("far-left",    115.0,    0.0,    22.0,    0.0,   55.0),
    ("far-right",  -115.0,    0.0,   -22.0,    0.0,   55.0),
    ("behind",      170.0,    0.0,    28.0,    0.0,   70.0),
    ("high-left",    55.0,   30.0,    12.0,  -14.0,   32.0),
    ("high-right",  -55.0,   30.0,   -12.0,  -14.0,   32.0),
    ("overhead",     25.0,   45.0,     8.0,  -22.0,   24.0),
    ("low-left",     60.0,  -40.0,    14.0,   12.0,   32.0),
    ("low-right",   -60.0,  -40.0,   -14.0,   12.0,   32.0),
    ("head-on",      44.0,    4.0,     9.0,    0.0,   18.0),
    ("crossing",     95.0,   10.0,    45.0,   -6.0,   45.0),
)
"""Positive bearing is to the *left* of the boresight, and every entry's
``transit_miss_m`` is signed to the opposite side so the course sweeps across the
frame rather than sitting frozen at its edge on a constant-bearing collision
course. ``head-on`` is 44 degrees rather than 0 for exactly that reason: an
intruder aimed precisely at the chaser would never appear to move at all."""


# The four relative-motion cases, named the way an operator describes them, each
# flown at three ranges. `ahead`/`right` are the ingress aim point; `pol` is what
# the intruder does once it knows it has been seen.
#
# `outbound` is the odd one and the reason this table exists separately from the
# ingress entries: a target that is already running is a *tail chase*, which
# converges only on the speed difference, and it is the case where the initial
# range matters most because every metre of it has to be flown twice.
APPROACH_CASES = (
    # label            what it does            ahead  right  up   policy  ranges
    ("inbound",       "flies straight at us",   10.0,   0.0,  9.0, "flee",
     (45.0, 70.0, 95.0)),
    ("outbound",      "runs directly away",     -1.0,   0.0,  0.0, "flee",
     (30.0, 45.0, 60.0)),
    ("left-to-right", "crosses L to R",         30.0,  38.0,  0.0, "weave",
     (45.0, 70.0, 95.0)),
    ("right-to-left", "crosses R to L",         30.0, -38.0,  0.0, "weave",
     (45.0, 70.0, 95.0)),
)
"""Ranges are per case, and ``outbound``'s are shorter for two reasons that both
come from it being a tail chase. The evader is held inside an arena of radius
``EvaderConfig.arena_radius_m`` (90 m), so a target *starting* at 95 m is already
outside it and gets pulled back inward -- it cannot flee, and the scenario would
silently stop testing what its name says. And a tail chase closes only on the
speed difference (14.4 - 9 = 5.4 m/s), so 95 m would be 18 seconds of straight
line before the guidance law had anything to do."""


# Bearings the intruder runs in from, and the building it is going for. The
# defended assets sit 45-70 m out from where the interceptor holds station --
# far enough that the intruder is a genuine 5-8 px contact when the clock
# starts, close enough that arriving late actually loses.
DEFEND_CASES = (
    #  label        approach bearing   asset bearing  asset range  intruder range
    ("north",            15.0,            25.0,         55.0,        150.0),
    ("north-east",       70.0,            60.0,         48.0,        135.0),
    ("east",            120.0,           110.0,         62.0,        165.0),
    ("south-east",      165.0,           150.0,         45.0,        140.0),
    ("south",          -150.0,          -160.0,         58.0,        155.0),
    ("south-west",      -95.0,          -105.0,         50.0,        130.0),
    ("west",            -55.0,           -45.0,         66.0,        170.0),
    ("north-west",      -20.0,           -10.0,         52.0,        145.0),
)


def build_defend_suite(base: ScenarioConfig) -> List[ScenarioConfig]:
    """Point defence: hold station, find the intruder, kill it before it arrives.

    The mission the whole system exists for, and the only suite in which the
    interceptor can lose by being *slow* rather than by being wrong.

    Everywhere else the intruder's business is with the chaser, which makes the
    chaser the centre of the engagement and gives it as long as it likes. Here
    the intruder ignores it completely and runs at a building; the chaser starts
    holding station in the middle of the scene, scanning, and has to find a
    5-8 px contact at 130-170 m and reach it before it arrives. Failure has a
    new and more honest name -- ``target_struck`` -- and success carries
    ``strike_margin_s``, the time that was left, because an intercept with
    0.2 s to spare and one with 8 s to spare are not the same result.

    The scan that finds it is the ordinary SEARCH behaviour: 34 degree steps
    with a 0.55 s dwell, which sweeps 360 degrees in 10.2 s at an average
    35 deg/s and overlaps consecutive looks by 55 percent of the field of view.
    That rate is not arbitrary -- the dwell is 11 frames where the tracker needs
    3 to seed and confirm, so a contact that appears anywhere in the sweep is
    held long enough to become a track rather than a glimpse.
    """
    out: List[ScenarioConfig] = []
    for i, (label, approach, asset_b, asset_r, intruder_r) in enumerate(DEFEND_CASES):
        ab = math.radians(asset_b)
        for j, speed in enumerate((9.0, 12.0)):
            out.append(replace(
                base,
                name=f"defend-{label}" + ("-fast" if j else ""),
                entry=label,
                policy="evasive" if j else "weave",
                seed=500 + i * 10 + j,
                ingress=True,
                # Station is the middle of the scene; the chaser holds it and
                # scans rather than being pre-pointed at anything.
                chaser_offset_xy=(0.0, 0.0),
                chaser_yaw_deg=(i * 45.0) % 360.0 - 180.0,
                altitude_m=24.0 + 4.0 * (i % 3),
                start_range_m=intruder_r,
                start_bearing_deg=approach,
                start_elevation_deg=4.0 + 2.0 * (i % 3),
                defend_xy=(asset_r * math.cos(ab), asset_r * math.sin(ab)),
                defend_height_m=12.0 + 3.0 * (i % 4),
                strike_radius_m=10.0,
                transit_speed=speed,
                evader_speed=speed,
                # No proximity reveal here, unlike every other ingress suite.
                # An intruder that has not been detected has no reason to break
                # off, and letting it do so because the interceptor happens to
                # be nearby quietly saves the asset in exactly the runs where
                # the system failed -- it turns a miss into a draw and hides
                # the only failure that matters. Here it presses the attack
                # until it is locked or it arrives.
                reveal_range_m=0.0,
                max_seconds=60.0,
            ))
    return out


def build_approach_suite(base: ScenarioConfig) -> List[ScenarioConfig]:
    """The four relative-motion cases an operator would actually name.

    Kept apart from :data:`INGRESS_ENTRIES`, which enumerates *where the
    intruder comes from*; this enumerates *what it does relative to us*, which
    is a different axis and stresses different parts of the law. A crossing
    target is a line-of-sight-rate problem and exercises PN; an outbound target
    is a closing-speed problem and exercises nothing but the speed advantage.

    ``outbound`` is deliberately not an ingress scenario. A target already
    running away is behind the engagement, not entering it, so it starts in
    view and simply flees -- posing it as an ingress would be dishonest about
    which problem is being tested.
    """
    out: List[ScenarioConfig] = []
    for i, (label, _desc, ahead, right, vert, pol, ranges) in enumerate(APPROACH_CASES):
        for j, rng in enumerate(ranges):
            ang = ((i * 3 + j) * 2.399963) % (2.0 * math.pi)
            rad = 15.0 + 9.0 * ((i + j) % 4)
            inbound = ahead > 0.0
            out.append(replace(
                base,
                name=f"{label}-{rng:.0f}m",
                entry=label,
                policy=pol,
                seed=300 + i * 10 + j,
                ingress=inbound,
                chaser_offset_xy=(rad * math.cos(ang), rad * math.sin(ang)),
                chaser_yaw_deg=((i * 3 + j) * 61.0) % 360.0 - 180.0,
                start_range_m=rng,
                # An inbound or crossing intruder starts outside the +/-39 deg
                # frame; a fleeing one is already in it, by definition.
                start_bearing_deg=(55.0 if inbound and right >= 0 else
                                   -55.0 if inbound else 0.0),
                start_elevation_deg=(6.0 if j == 2 else 0.0),
                altitude_m=20.0 + 6.0 * j,
                transit_ahead_m=max(0.0, ahead),
                transit_miss_m=right,
                transit_vertical_m=vert,
                transit_speed=9.0 + 1.0 * j,
                reveal_range_m=24.0,
                max_seconds=25.0 + 12.0 * j,
            ))
    return out


def build_ingress_suite(base: ScenarioConfig, wide: bool = False
                        ) -> List[ScenarioConfig]:
    """Intruder-entry engagements: empty frame, something arrives, kill it.

    This is the scenario the earlier suites never posed. They placed the target
    at a bearing the camera was already looking at, so the run began after the
    hard part; here the frame starts empty, the intruder transits in from a
    direction the chaser is not watching, and it only begins to evade once the
    chaser has a confirmed track on it.

    Both aircraft are also scattered: the chaser gets a different start position
    and a different heading in every scenario. Flying every engagement from one
    spot on one heading meant the same scenery sat behind the target every time,
    which is a good way to measure a detector's memory of a skybox instead of
    its ability to find a drone.

    ``wide`` adds a second pass over the entry table with harder evasion, longer
    ranges and different altitudes.
    """
    out: List[ScenarioConfig] = []
    policies = ("flee", "weave", "barrel", "evasive")
    for i, (label, bearing, elev, lat, vert, ahead) in enumerate(INGRESS_ENTRIES):
        # Deterministic scatter -- reproducible, but different for every entry.
        ang = (i * 2.399963) % (2.0 * math.pi)          # golden-angle spiral
        rad = 18.0 + 11.0 * ((i * 7) % 5)
        out.append(replace(
            base,
            name=f"in-{label}",
            entry=label,
            policy=policies[i % len(policies)],
            seed=100 + i,
            ingress=True,
            chaser_offset_xy=(rad * math.cos(ang), rad * math.sin(ang)),
            chaser_yaw_deg=(i * 47.0) % 360.0 - 180.0,
            start_range_m=70.0 + 9.0 * (i % 6),
            start_bearing_deg=bearing,
            start_elevation_deg=elev,
            altitude_m=22.0 + 5.0 * (i % 4),
            transit_miss_m=lat,
            transit_vertical_m=vert,
            transit_ahead_m=ahead,
            transit_speed=9.0 + 1.5 * (i % 3),
            reveal_range_m=22.0,
            max_seconds=55.0,
        ))
    if wide:
        for i, (label, bearing, elev, lat, vert, ahead) in enumerate(INGRESS_ENTRIES):
            ang = ((i + 6) * 2.399963) % (2.0 * math.pi)
            rad = 24.0 + 13.0 * ((i * 3) % 4)
            out.append(replace(
                base,
                name=f"in-{label}-hard",
                entry=label,
                policy=("evasive", "jink", "break_turn", "orbit")[i % 4],
                seed=200 + i,
                ingress=True,
                chaser_offset_xy=(rad * math.cos(ang), rad * math.sin(ang)),
                chaser_yaw_deg=(i * 83.0) % 360.0 - 180.0,
                start_range_m=95.0 + 11.0 * (i % 5),
                start_bearing_deg=bearing,
                start_elevation_deg=elev,
                altitude_m=16.0 + 7.0 * (i % 5),
                transit_miss_m=lat * 1.4,
                transit_vertical_m=vert * 1.3,
                transit_ahead_m=ahead * 1.15,
                transit_speed=10.0 + 1.5 * (i % 4),
                evader_speed=10.0 + 1.0 * (i % 3),
                reveal_range_m=25.0,
                max_seconds=60.0,
            ))
    return out


def summarize(results: List[EpisodeResult], verbose: bool = True) -> dict:
    n = len(results)
    hits = [r for r in results if r.success]
    miss = [r for r in results if not r.success]
    tti = [r.time_to_intercept_s for r in hits if r.time_to_intercept_s]
    out = {"n": n, "hits": len(hits),
           "rate": round(len(hits) / max(1, n), 4),
           "median_tti_s": round(float(np.median(tti)), 2) if tti else None,
           "p95_tti_s": round(float(np.percentile(tti, 95)), 2) if tti else None,
           "worst_miss_m": round(max((r.miss_distance_m for r in miss), default=0.0), 2),
           "failures": [f"{r.name}:{r.outcome}:{r.miss_distance_m:.1f}m" for r in miss],
           }
    if verbose:
        print(f"  {out['hits']}/{n} intercepts ({100 * out['rate']:.1f}%)  "
              f"median t={out['median_tti_s']}s p95={out['p95_tti_s']}s")
        if miss:
            print("   failures: " + ", ".join(out["failures"][:12])
                  + (" ..." if len(miss) > 12 else ""))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="full")
    ap.add_argument("--range", type=float, default=40.0, dest="start_range")
    ap.add_argument("--altitude", type=float, default=25.0)
    ap.add_argument("--evader-speed", type=float, default=9.0)
    ap.add_argument("--speed-advantage", type=float, default=1.6)
    ap.add_argument("--max-seconds", type=float, default=45.0)
    ap.add_argument("--hit-radius", type=float, default=1.0)
    ap.add_argument("--arena", type=float, default=90.0)

    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--noise-px", type=float, default=0.0)
    ap.add_argument("--span-noise", type=float, default=0.0)
    ap.add_argument("--span-bias", type=float, default=1.0)
    ap.add_argument("--latency-frames", type=int, default=0)
    ap.add_argument("--compensate-latency", action="store_true",
                    help="tell guidance how old its bearings are (the real "
                         "pipeline's latency is calibrated, not guessed)")
    ap.add_argument("--min-span-px", type=float, default=4.0)
    ap.add_argument("--ring", action="store_true",
                    help="fly the four-camera 360 degree ring instead of the "
                         "single nose camera (implies omnidirectional guidance)")

    ap.add_argument("--sweep", default=None,
                    help="grid-search guidance fields, e.g. "
                         "'nav_gain=3,4,5;lookahead_s=0.3,0.45'")
    ap.add_argument("--trace", default=None, help="dump a per-tick trace for this scenario")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    base = ScenarioConfig(start_range_m=a.start_range, altitude_m=a.altitude,
                          evader_speed=a.evader_speed,
                          speed_advantage=a.speed_advantage,
                          max_seconds=a.max_seconds, hit_radius_m=a.hit_radius)
    scenarios = build_suite(a.suite, base)
    ecfg = EvaderConfig(speed=a.evader_speed,
                        arena_radius_m=(max(a.arena, 400.0)
                                        if a.suite.startswith("city") else a.arena))

    def camera(seed: int):
        if a.ring:
            from pursuit.ring import default_ring
            return SyntheticRing(default_ring(RING_INTRINSICS), seed=seed,
                                 min_span_px=a.min_span_px,
                                 dropout=a.dropout, noise_px=a.noise_px,
                                 span_noise=a.span_noise, span_bias=a.span_bias,
                                 latency_frames=a.latency_frames)
        return SyntheticCamera(SIM_INTRINSICS, dropout=a.dropout,
                               noise_px=a.noise_px, span_noise=a.span_noise,
                               span_bias=a.span_bias,
                               latency_frames=a.latency_frames,
                               min_span_px=a.min_span_px, seed=seed)

    if a.trace:
        sc = next((s for s in scenarios if s.name == a.trace), None)
        if sc is None:
            raise SystemExit(f"no scenario named {a.trace!r} in suite {a.suite!r}")
        r = run_episode(sc, replace(GuidanceConfig(), hit_range_m=a.hit_radius,
                                    omnidirectional=bool(a.ring)),
                        ecfg, camera(sc.seed), trace=True)
        for row in r.stage_ms["trace"]:
            print(json.dumps(row))
        print(f"# {r.outcome} miss={r.miss_distance_m} t={r.time_to_intercept_s}")
        return 0

    grids = {}
    if a.sweep:
        for part in a.sweep.split(";"):
            k, v = part.split("=")
            grids[k.strip()] = [float(x) for x in v.split(",")]

    best = None
    rows = []
    combos = ([dict(zip(grids, vals)) for vals in itertools.product(*grids.values())]
              if grids else [{}])
    for over in combos:
        # Build the override dict first. Passing sensor_latency_s positionally
        # *and* letting a sweep supply it is a TypeError, which made that field
        # the one knob in GuidanceConfig that could never be swept -- and it
        # turned out to be the most valuable one in the config.
        fields = dict(hit_range_m=a.hit_radius)
        if a.compensate_latency:
            fields["sensor_latency_s"] = a.latency_frames * 0.05
        if a.ring:
            fields["omnidirectional"] = True
        fields.update(over)
        gcfg = replace(GuidanceConfig(), **fields)
        if a.suite.startswith("city"):
            from pursuit.city import city_guidance, city_top_speed
            gcfg = city_guidance(gcfg, city_top_speed(scenarios))
        results = [run_episode(sc, gcfg, ecfg, camera(sc.seed)) for sc in scenarios]
        label = ", ".join(f"{k}={v:g}" for k, v in over.items()) or "default"
        print(f"[{label}]")
        s = summarize(results)
        rows.append({"params": over, **s})
        key = (s["rate"], -(s["median_tti_s"] or 1e9))
        if best is None or key > best[0]:
            best = (key, over, s, results)

    if grids and best:
        print(f"\nBEST: {best[1] or 'default'} -> {100 * best[2]['rate']:.1f}% "
              f"median t={best[2]['median_tti_s']}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"args": vars(a), "grid": rows,
             "results": [asdict(r) for r in best[3]]}, indent=1))
        print(f"wrote {a.out}")
    return 0 if best and best[2]["rate"] >= 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
