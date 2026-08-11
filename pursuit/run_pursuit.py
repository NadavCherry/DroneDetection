#!/usr/bin/env python3
"""Fly the complete algorithm: detect, track, close, hit -- over a scenario matrix.

    # start the simulator once (inside the container, stays up across runs)
    docker exec -d isaac-sim bash -c "cd /tmp/dev/dronedet && /isaac-sim/python.sh \\
        simulators/pegasus/scripts/pursuit_server.py --scene skydome"

    # guidance against a perfect sensor -- isolates the control law
    .venv/bin/python -m pursuit.run_pursuit --detector oracle --suite full --out work/pursuit/oracle

    # the real thing, end to end
    .venv/bin/python -m pursuit.run_pursuit --detector yolo \\
        --weights work/runs/sim-drone/weights/best.pt --suite full --out work/pursuit/yolo

A run is scored on one thing: did the two aircraft come within ``--hit-radius``
of each other, measured at the true closest point of approach rather than at
whichever tick happened to sample it. Everything else in the report exists to
explain a failure.

The suites are a difficulty ladder, and the order matters when reading a
regression: ``smoke`` is one straight-line target, ``core`` adds the evasive
policies at one geometry, and ``full`` crosses every policy with off-boresight
starts, longer ranges and faster targets -- the cases where a law that merely
*works* stops working.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pursuit.episode import Episode, EpisodeResult, ScenarioConfig
from pursuit.evader import POLICIES, EvaderConfig
from pursuit.geometry import Intrinsics
from pursuit.guidance import GuidanceConfig
from pursuit.perception import OracleDetector, Perception, TrackerConfig, YoloDetector
from pursuit.sandbox import build_suite
from simulators.pegasus.pursuit_proto import SimClient, host_socket

HOST_SOCKET = host_socket()


# ------------------------------------------------------------------- suites

def suite(name: str, base: ScenarioConfig) -> List[ScenarioConfig]:
    """The scenario list, shared with the headless sandbox.

    Deliberately the *same* function rather than a parallel copy. The sandbox
    exists to answer "does the guidance law converge" and this module to answer
    "does it converge on real pixels", and that comparison is only worth
    anything if both are flying the identical matrix -- two lists that drifted
    apart would turn every difference into an unanswerable question.
    """
    return build_suite(name, base)


# --------------------------------------------------------------------- report

def print_table(results: List[EpisodeResult]) -> None:
    hdr = (f"{'scenario':<18}{'outcome':<14}{'miss m':>8}{'t_hit s':>9}"
           f"{'det':>7}{'trk':>7}{'acq s':>7}{'rng err':>9}"
           f"{'det ms':>8}{'trk ms':>8}{'gui ms':>8}{'FPS':>7}{'FPS p95':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r.name:<18}{r.outcome:<14}{r.miss_distance_m:>8.2f}"
              f"{(r.time_to_intercept_s if r.time_to_intercept_s else float('nan')):>9.2f}"
              f"{r.detect_rate:>7.2f}{r.track_rate:>7.2f}"
              f"{(r.acquire_time_s if r.acquire_time_s is not None else float('nan')):>7.2f}"
              f"{(r.mean_range_err_m if r.mean_range_err_m is not None else float('nan')):>9.2f}"
              f"{r.stage_ms.get('detect_ms', 0):>8.1f}"
              f"{r.stage_ms.get('track_ms', 0):>8.2f}"
              f"{r.stage_ms.get('guidance_ms', 0):>8.2f}"
              f"{r.stage_ms.get('pipeline_fps', 0):>7.1f}"
              f"{r.stage_ms.get('pipeline_fps_p95', 0):>9.1f}")
    n = len(results)
    hits = sum(1 for r in results if r.success)
    print("-" * len(hdr))
    print(f"{'INTERCEPT':<18}{hits}/{n}  ({100.0 * hits / max(1, n):.1f}%)")

    # Rate the brain alone can hold. Deliberately NOT wall_fps: that is
    # dominated by Isaac rendering five frames per control tick, a property of
    # the test rig that will not exist on the aircraft. These are the numbers
    # that carry over to edge hardware.
    def _agg(key):
        xs = [r.stage_ms.get(key) for r in results if r.stage_ms.get(key)]
        return sum(xs) / len(xs) if xs else 0.0

    det, trk, gui = _agg("detect_ms"), _agg("track_ms"), _agg("guidance_ms")
    loop = det + trk + gui
    if loop > 0:
        p95 = _agg("pipeline_p95_ms")
        print(f"{'PIPELINE':<18}detect {det:.1f} ms + track {trk:.2f} ms + "
              f"guidance {gui:.2f} ms = {loop:.1f} ms")
        print(f"{'':<18}{1000.0 / loop:.1f} FPS mean"
              + (f", {1000.0 / p95:.1f} FPS at p95" if p95 > 0 else "")
              + f"   (renderer-inclusive wall rate {_agg('wall_fps'):.1f} FPS)")
    if hits < n:
        bad = [r for r in results if not r.success]
        worst = sorted(bad, key=lambda r: -r.miss_distance_m)[:6]
        print("  failures: " + ", ".join(
            f"{r.name}({r.outcome}, miss {r.miss_distance_m:.1f}m)" for r in worst))


# ----------------------------------------------------------------------- main

def build_ring_perception(a, info: dict):
    """The four-camera front end, wired for whichever sensor was asked for.

    ``oracle`` deliberately runs **without** the motion detector. The oracle
    exists to isolate the guidance law from the perception, and bolting a real
    detector onto a perfect one puts the thing being controlled for back into
    the experiment.
    """
    from pursuit.ring import (MotionConfig, Ring, RingMotionDetector,
                              RingOracle, RingPerception, RingTrackerConfig)

    ring = Ring.from_info(info)
    tcfg = RingTrackerConfig(max_coast_frames=a.max_coast,
                             init_score=a.init_score)
    if a.detector == "oracle":
        oracle = RingOracle(ring, seed=11, dropout=a.oracle_dropout,
                            noise_px=a.oracle_noise_px,
                            span_noise=a.oracle_span_noise,
                            span_bias=a.oracle_span_bias,
                            max_range_m=a.oracle_max_range,
                            latency_frames=a.latency_frames)
        return RingPerception(ring, oracle=oracle, tracker_cfg=tcfg,
                              min_score=a.min_score)

    mcfg = MotionConfig()
    if a.motion_k_static is not None:
        mcfg.k_static = a.motion_k_static
    if a.motion_min_area is not None:
        mcfg.min_area = a.motion_min_area
    motion = None if a.no_motion else RingMotionDetector(ring, mcfg)

    det = None
    if a.detector == "yolo":
        if not a.weights:
            raise SystemExit("--detector yolo needs --weights")
        det = YoloDetector(a.weights, imgsz=a.imgsz, conf=a.conf,
                           half=not a.no_half)
    elif a.detector == "fusion":
        if not a.weights:
            raise SystemExit("--detector fusion needs --weights (a ch=4 checkpoint)")
        from pursuit.perception import FusionDetector
        det = FusionDetector(a.weights, tile=640, conf=a.conf)
    elif a.detector == "motion":
        if motion is None:
            raise SystemExit("--detector motion with --no-motion detects nothing")
    return RingPerception(ring, detector=det, motion=motion, tracker_cfg=tcfg,
                          min_score=a.min_score,
                          max_appearance_cams=a.appearance_cams,
                          crop_px=a.crop_px)


def build_perception(a, intr: Intrinsics) -> Perception:
    if a.detector == "oracle":
        det = OracleDetector(dropout=a.oracle_dropout, noise_px=a.oracle_noise_px,
                             span_noise=a.oracle_span_noise,
                             span_bias=a.oracle_span_bias,
                             max_range_m=a.oracle_max_range,
                             latency_frames=a.latency_frames, seed=11)
    elif a.detector == "fusion":
        if not a.weights:
            raise SystemExit("--detector fusion needs --weights (a ch=4 checkpoint)")
        from pursuit.perception import FusionDetector
        det = FusionDetector(a.weights, tile=640, conf=a.conf)
    else:
        if not a.weights:
            raise SystemExit("--detector yolo needs --weights")
        det = YoloDetector(a.weights, imgsz=a.imgsz, conf=a.conf, half=not a.no_half)
    return Perception(det, intr,
                      TrackerConfig(max_coast_frames=a.max_coast,
                                    init_score=a.init_score),
                      min_score=a.min_score)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--socket", default=HOST_SOCKET)
    ap.add_argument("--out", default="work/pursuit/run")
    ap.add_argument("--suite", default="core")
    ap.add_argument("--limit", type=int, default=None, help="run only the first N scenarios")
    ap.add_argument("--only", default=None,
                    help="comma-separated scenario names to fly, instead of the "
                         "whole suite. For re-recording a handful of clips "
                         "without re-flying an hour of engagements")

    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="override any GuidanceConfig field, e.g. "
                         "--set search_dwell_s=0.25 --set search_step_rad=0.9")
    ap.add_argument("--detector", choices=["oracle", "yolo", "fusion", "motion"],
                    default="oracle")
    ap.add_argument("--no-motion", action="store_true",
                    help="ring only: turn the frame-difference detector off, "
                         "leaving the appearance model alone. The measurement "
                         "that says what the motion half is worth")
    ap.add_argument("--motion-k-static", type=float, default=None)
    ap.add_argument("--motion-min-area", type=int, default=None)
    ap.add_argument("--crop-px", type=int, default=640,
                    help="ring only: run the appearance model on a window this "
                         "wide around whatever the motion stage found, at "
                         "native scale. 0 runs whole frames")
    ap.add_argument("--appearance-cams", type=int, default=2,
                    help="ring only: how many cameras the appearance model may "
                         "run on per tick. Four would be 4x the detector cost "
                         "and the ring is aimed, not swept")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--no-half", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-coast", type=int, default=12)
    ap.add_argument("--init-score", type=float, default=0.35,
                    help="confidence needed to START a lock (kept above "
                         "--conf, which only has to maintain one)")

    ap.add_argument("--oracle-dropout", type=float, default=0.0)
    ap.add_argument("--oracle-noise-px", type=float, default=0.0)
    ap.add_argument("--oracle-span-noise", type=float, default=0.0)
    ap.add_argument("--oracle-span-bias", type=float, default=1.0)
    ap.add_argument("--oracle-max-range", type=float, default=None)
    ap.add_argument("--latency-frames", type=int, default=0,
                    help="delay every detection by N frames (sensor+compute lag)")

    ap.add_argument("--range", type=float, default=40.0, dest="start_range")
    ap.add_argument("--altitude", type=float, default=25.0)
    ap.add_argument("--evader-speed", type=float, default=9.0)
    ap.add_argument("--speed-advantage", type=float, default=1.6)
    ap.add_argument("--max-seconds", type=float, default=45.0)
    ap.add_argument("--hit-radius", type=float, default=1.0)
    ap.add_argument("--arena", type=float, default=90.0)

    ap.add_argument("--nav-gain", type=float, default=None)
    ap.add_argument("--approach-speed", type=float, default=None)
    ap.add_argument("--terminal-range", type=float, default=None)

    ap.add_argument("--video", action="store_true", help="write annotated mp4s")
    ap.add_argument("--video-limit", type=int, default=6,
                    help="cap how many scenarios get a video")
    ap.add_argument("--video-names", default=None,
                    help="comma-separated scenario names to record instead of "
                         "the first N. A suite ordered by arrival bearing makes "
                         "'the first ten' ten neighbouring directions, which is "
                         "the least informative ten clips it could produce")
    ap.add_argument("--sky", default=None, help="switch the HDRI before running")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    want_video = ({s.strip() for s in a.video_names.split(",") if s.strip()}
                  if a.video_names else None)

    base = ScenarioConfig(start_range_m=a.start_range, altitude_m=a.altitude,
                          evader_speed=a.evader_speed,
                          speed_advantage=a.speed_advantage,
                          max_seconds=a.max_seconds, hit_radius_m=a.hit_radius)
    scenarios = suite(a.suite, base)
    if a.only:
        want = {x.strip() for x in a.only.split(",") if x.strip()}
        scenarios = [s for s in scenarios if s.name in want]
        if not scenarios:
            raise SystemExit(f"--only matched nothing in suite {a.suite!r}")
    if a.limit:
        scenarios = scenarios[:a.limit]

    gcfg = GuidanceConfig()
    over = {k: v for k, v in (("nav_gain", a.nav_gain),
                              ("approach_speed", a.approach_speed),
                              ("terminal_range_m", a.terminal_range)) if v is not None}
    over["hit_range_m"] = a.hit_radius
    for spec in a.set:
        field, _, raw = spec.partition("=")
        field = field.strip()
        if not hasattr(gcfg, field):
            raise SystemExit(f"--set: GuidanceConfig has no field {field!r}")
        over[field] = float(raw)
    gcfg = replace(gcfg, **over)
    if a.suite.startswith("city"):
        from pursuit.city import city_guidance, city_top_speed
        gcfg = city_guidance(gcfg, city_top_speed(scenarios))
    ecfg = EvaderConfig(speed=a.evader_speed,
                        # A committed strike run starts 190 m out and is not
                        # trying to stay in an arena; the 90 m default would
                        # be a fence the intruder starts outside of.
                        arena_radius_m=(max(a.arena, 400.0)
                                        if a.suite.startswith("city") else a.arena))

    with SimClient(a.socket) as client:
        if a.sky:
            client.call("set_sky", sky=a.sky)
        info = client.info()
        print(f"sim: scene={info['scene_name']} {info['intrinsics']['width']}x"
              f"{info['intrinsics']['height']} dt={info['dt']} "
              f"render_ticks={info.get('render_ticks')} sync={info.get('sync')}")
        intr = Intrinsics.from_dict(info["intrinsics"])
        if info.get("ring"):
            print(f"     ring x{len(info['cameras'])} "
                  f"{[c['name'] for c in info['cameras']]} "
                  f"covering {info.get('coverage_deg')} deg, "
                  f"{info.get('seam_overlap_deg')} deg seam overlap")
            perception = build_ring_perception(a, info)
            # Pointing stops being a manoeuvre. See GuidanceConfig.
            gcfg = replace(gcfg, omnidirectional=True)
        else:
            perception = build_perception(a, intr)

        results: List[EpisodeResult] = []
        t0 = time.perf_counter()
        for i, sc in enumerate(scenarios):
            recorder = None
            wanted = (sc.name in want_video if want_video is not None
                      else i < a.video_limit)
            if a.video and wanted:
                if info.get("ring"):
                    from pursuit.city import map_overlay
                    from pursuit.ring import Ring
                    from pursuit.viz import RingRecorder
                    bs, aim = map_overlay(sc, info["origin_xy"])
                    recorder = RingRecorder(
                        out / f"{sc.name}.mp4", Ring.from_info(info),
                        fps=info["fps"],
                        arena_m=max(150.0, 0.80 * sc.start_range_m + 40.0),
                        buildings=bs, aim_xy=aim,
                        strike_radius_m=sc.strike_radius_m)
                else:
                    from pursuit.viz import PursuitRecorder
                    recorder = PursuitRecorder(out / f"{sc.name}.mp4",
                                               (intr.width, intr.height),
                                               fps=info["fps"],
                                               arena_m=a.arena + 20)
            ep = Episode(client, info, perception, gcfg, ecfg, recorder)
            r = ep.run(sc)
            # Telemetry first, and the video release guarded. A scenario matrix
            # is an hour of rendering; losing all of it because the *writer*
            # raised after the flight is the wrong failure, and it is exactly
            # what happened once.
            ep.save_telemetry(out / f"{sc.name}.telemetry.json")
            try:
                if recorder is not None:
                    recorder.close()
            except Exception as exc:                              # noqa: BLE001
                print(f"  (video close failed: {type(exc).__name__}: {exc})")
            results.append(r)
            flag = "HIT " if r.success else "MISS"
            print(f"[{i + 1}/{len(scenarios)}] {flag} {sc.name:<18} "
                  f"{r.outcome:<14} miss={r.miss_distance_m:6.2f}m "
                  f"t={r.time_to_intercept_s} det={r.detect_rate:.2f} "
                  f"trk={r.track_rate:.2f}", flush=True)

    print_table(results)
    payload = {
        "args": vars(a),
        "guidance": asdict(gcfg),
        "evader": asdict(ecfg),
        "sim": {k: info[k] for k in ("scene_name", "dt", "fps", "render_ticks")},
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "n": len(results),
        "hits": sum(1 for r in results if r.success),
        "results": [asdict(r) for r in results],
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}/results.json")
    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
