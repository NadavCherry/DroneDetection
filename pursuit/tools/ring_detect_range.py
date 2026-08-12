#!/usr/bin/env python3
"""How far out can the ring see an inbound intruder, and what is it worth?

The one measurement the city-defence mission turns on. An interceptor 1.5x
faster than its target defends a radius of ``0.6 * R_detect`` minus a second or
so of reaction (``pursuit/city.py`` derives it), so detection range is not one
figure of merit among several -- it *is* the defended radius, at a fixed 0.6
metres per metre. Nothing else in the system trades that steeply, which makes
guessing at it the most expensive thing anyone could do here.

So this flies the real approach and scores every frame: the interceptor holds
station over Rivermark exactly as it would in the mission, an intruder runs in
from 250 m on a real bearing at a real speed, and each detector says whether it
found it. Nothing is teleported -- the motion detector is temporal by
construction and a cold call on a jumped pose is meaningless (the same reason
this repository's dataset tool records whole flights).

    .venv/bin/python -m pursuit.tools.ring_detect_range --bearings 0,90,200
    .venv/bin/python -m pursuit.tools.ring_detect_range --weights work/runs/sim-n-p2/weights/best.pt

Reports, per 10 m range bin: what fraction of frames each detector produced a
box within the gate, and what defended radius that range implies.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pursuit.geometry import angle_between  # noqa: E402
from pursuit.ring import (MotionConfig, Ring,  # noqa: E402
                          RingMotionDetector)
from simulators.pegasus.pursuit_proto import SimClient, host_socket  # noqa: E402

HOST_SOCKET = host_socket()
SPEED_ADVANTAGE = 1.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--socket", default=HOST_SOCKET)
    ap.add_argument("--bearings", default="0,75,155,240,310",
                    help="world bearings the intruder runs in from")
    ap.add_argument("--start-range", type=float, default=250.0)
    ap.add_argument("--stop-range", type=float, default=25.0)
    ap.add_argument("--speed", type=float, default=12.0)
    ap.add_argument("--altitude", type=float, default=30.0)
    ap.add_argument("--intruder-agl", type=float, default=20.0)
    ap.add_argument("--chaser-yaw-deg", type=float, default=23.0)
    ap.add_argument("--weights", default=None,
                    help="also score an appearance detector, for the comparison")
    ap.add_argument("--imgsz", type=int, default=2048)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--gate-deg", type=float, default=1.2,
                    help="angular gate for calling a detection a hit")
    ap.add_argument("--k-static", type=float, default=None)
    ap.add_argument("--min-area", type=int, default=None)
    ap.add_argument("--sweep", default=None,
                    help="compare several motion configurations on the SAME "
                         "frames, e.g. 'scale=1,0.5;open_ksize=0,3'. One pass "
                         "of the simulator scores all of them, which matters "
                         "because a step here costs a second and the variants "
                         "have to be compared on identical pixels to mean "
                         "anything")
    ap.add_argument("--quiet-frames", type=int, default=0,
                    help="measure the false-blob rate with an empty sky first")
    ap.add_argument("--out", default="work/pursuit/ring_detect_range.json")
    a = ap.parse_args(argv)

    mcfg = MotionConfig()
    if a.k_static is not None:
        mcfg.k_static = a.k_static
    if a.min_area is not None:
        mcfg.min_area = a.min_area
    variants = _variants(mcfg, a.sweep)

    yolo = None
    if a.weights:
        from pursuit.perception import YoloDetector
        yolo = YoloDetector(a.weights, imgsz=a.imgsz, conf=a.conf)

    bearings = [float(x) for x in a.bearings.split(",")]
    rows = []
    with SimClient(a.socket) as c:
        info = c.info()
        if not info.get("ring"):
            raise SystemExit("the server is not running --cameras ring")
        ring = Ring.from_info(info)
        intr = ring.cameras[0].intr
        gz, (ox, oy) = float(info["ground_z"]), info["origin_xy"]
        dt = float(info["dt"])
        print(f"scene={info['scene_name']} ring={len(ring.cameras)} "
              f"{intr.width}x{intr.height} fx={intr.fx:.1f}")
        print(f"intruder {a.speed:.0f} m/s from {a.start_range:.0f} m, "
              f"chaser holding at {a.altitude:.0f} m AGL, yaw {a.chaser_yaw_deg:.0f}\n")

        cz = gz + a.altitude
        tz = gz + a.intruder_agl
        yaw = math.radians(a.chaser_yaw_deg)
        chaser = {"xyz": [ox, oy, cz], "yaw": yaw}

        if a.quiet_frames:
            # What the detector reports with nothing to report. Half the value
            # of a wide-recall stage is the price it charges, and a false-blob
            # rate is the only honest way to read a detection rate: 0.5 at 120 m
            # means something quite different at 0.1 false blobs a frame than at
            # 5. The target is parked 3 km away, which is out of every camera
            # and still inside the far clip plane.
            print("quiet frames (no target in the sky):")
            quiet = {k: RingMotionDetector(ring, v) for k, v in variants.items()}
            counts = {k: 0 for k in variants}
            far = {"xyz": [ox + 3000.0, oy, tz], "yaw": 0.0}
            c.reset(chaser, far)
            for _ in range(int(a.quiet_frames)):
                _h, frames = c.call_frames("step", chaser=chaser, target=far)
                for k, md in quiet.items():
                    counts[k] += len(md.detect(frames, yaw, 0.0))
            for k, n in counts.items():
                print(f"  {k:<26} {n / a.quiet_frames:6.2f} false blobs per frame "
                      f"across all four cameras")
            print()

        for b in bearings:
            motions = {k: RingMotionDetector(ring, v) for k, v in variants.items()}
            ang = math.radians(b)
            rng = a.start_range
            first = {k: None for k in list(variants) + ["yolo"]}
            c.reset(chaser, {"xyz": [ox + rng * math.cos(ang),
                                     oy + rng * math.sin(ang), tz],
                             "yaw": ang + math.pi})
            n = 0
            while rng > a.stop_range:
                rng -= a.speed * dt
                tgt = {"xyz": [ox + rng * math.cos(ang),
                               oy + rng * math.sin(ang), tz],
                       "yaw": ang + math.pi}
                header, frames = c.call_frames("step", chaser=chaser, target=tgt)
                gt = header["gt"]
                truth = _truth_los(ring, gt)
                slant = math.dist((ox, oy, cz), tuple(gt["target_xyz"]))
                span = intr.fx * 0.47 / slant

                got = {"yolo": False}
                for key, md in motions.items():
                    got[key] = _near(md.detect(frames, yaw, 0.0), truth,
                                     a.gate_deg)
                if yolo is not None:
                    cam = ring.owner(truth) if truth else None
                    if cam is not None and frames.get(cam.name) is not None:
                        boxes = yolo.detect(frames[cam.name], n, None)
                        got["yolo"] = _near(
                            [type("D", (), {"los": cam.to_body(bx.cx, bx.cy)})()
                             for bx in boxes], truth, a.gate_deg)
                for k, v in got.items():
                    if v and first[k] is None:
                        first[k] = slant
                rows.append({"bearing": b, "range_m": slant, "span_px": span,
                             "camera": gt.get("camera"), **got})
                n += 1
            print(f"  bearing {b:>5.0f} deg: "
                  + ", ".join(f"{k} {_fmt(first[k])} m" for k in variants)
                  + (f", yolo {_fmt(first['yolo'])} m" if yolo else ""))

    _report(rows, list(variants) + (["yolo"] if yolo else []))
    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1), encoding="utf-8")
        print(f"\nwrote {p}")
    return 0


def _fmt(v):
    return "never" if v is None else f"{v:.0f}"


def _truth_los(ring: Ring, gt: dict):
    """Body-frame direction of the target, from the simulator's own label."""
    cam = ring.get(gt.get("camera") or "")
    uv = gt.get("uv") or gt.get("analytic_uv")
    if cam is None or uv is None:
        return None
    return cam.to_body(uv[0], uv[1])


def _near(dets, truth, gate_deg: float) -> bool:
    if truth is None:
        return False
    g = math.radians(gate_deg)
    return any(angle_between(d.los, truth) <= g for d in dets)


def _variants(base: MotionConfig, spec) -> dict:
    """``{label: MotionConfig}`` from a ``field=v1,v2;field=v3`` grid."""
    from dataclasses import replace as _replace
    from itertools import product

    if not spec:
        return {"motion": base}
    grid = {}
    for part in spec.split(";"):
        k, _, v = part.partition("=")
        vals = []
        for raw in v.split(","):
            raw = raw.strip()
            vals.append(int(raw) if raw.lstrip("-").isdigit() else float(raw))
        grid[k.strip()] = vals
    out = {}
    for combo in product(*grid.values()):
        over = dict(zip(grid, combo))
        label = ",".join(f"{k}={v:g}" for k, v in over.items())
        out[label] = _replace(base, **over)
    return out


def _report(rows, keys) -> None:
    bins = defaultdict(list)
    for r in rows:
        bins[int(r["range_m"] // 20) * 20].append(r)
    head = f"{'range':>9}{'span px':>9}{'n':>6}" + "".join(
        f"{k[-14:]:>16}" for k in keys)
    print("\n" + head)
    print("-" * len(head))
    for lo in sorted(bins):
        rs = bins[lo]
        line = (f"{lo:>4d}-{lo + 20:<4d}"
                f"{sum(r['span_px'] for r in rs) / len(rs):>9.2f}{len(rs):>6d}")
        for k in keys:
            line += f"{sum(1 for r in rs if r.get(k)) / len(rs):>16.2f}"
        print(line)

    # The number the mission cares about: the furthest range at which detection
    # is reliable, and the radius that defends.
    print()
    for key in keys:
        good = [lo for lo in sorted(bins)
                if sum(1 for r in bins[lo] if r.get(key)) / len(bins[lo]) >= 0.5]
        rdet = (max(good) + 20) if good else 0
        print(f"{key:<26} reliable to {rdet:>4d} m -> defends "
              f"{max(0.0, 0.6 * rdet - 14):>4.0f} m radius at 1.5x speed")


if __name__ == "__main__":
    raise SystemExit(main())
