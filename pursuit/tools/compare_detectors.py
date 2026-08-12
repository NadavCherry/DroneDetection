"""Head-to-head detector comparison on live simulator sequences.

The pursuit loop had been running a 2.9 M-parameter nano trained only on this
simulator, while the project's own flagship is a 25 M-parameter RGB+motion model
trained on ARD-MAV, NPS-Drones and real footage. That is an argument for
switching, not evidence, and the two models fail in opposite ways: the nano
knows this renderer and nothing else, the fusion model knows real drones and has
never seen this renderer. Which gap is worse is an empirical question.

It has to be measured on *sequences*, not sampled poses. The fusion model's
fourth channel is an ego-registered frame difference over t-3 and t-6, so it is
undefined on a teleported frame and would score zero for a reason that has
nothing to do with its ability. So this flies short constant-velocity passes with
the camera tracking the target -- the geometry a real pursuit spends its time in,
including the ego-motion the registration has to solve for -- and scores every
frame of every pass under both models.

Reported per span bucket, since that is what decides whether a lock can start:
``recall`` at the tracker's maintenance floor and ``acquire`` at the score a
track needs to be born.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pursuit.perception import FusionDetector, YoloDetector  # noqa: E402
from simulators.pegasus.pursuit_proto import SimClient, host_socket  # noqa: E402

BUCKETS = ((4, 8), (8, 14), (14, 25), (25, 50), (50, 10 ** 6))
SKIES = ("clear", "partly_cloudy", "cloudy", "overcast", "sunrise", "evening",
         "noon_grass", "lakeside", "mealie_road")


def _bucket(span: float) -> str | None:
    for lo, hi in BUCKETS:
        if lo <= span < hi:
            return f"{lo}-{hi if hi < 10 ** 5 else '+'}"
    return None


def _passes(rng: random.Random, n: int, ox: float, oy: float, gz: float):
    """Short flights: chaser closing on a target that is itself translating."""
    out = []
    for _ in range(n):
        r0 = rng.uniform(18.0, 85.0)
        bearing = rng.uniform(-math.pi, math.pi)
        cz = gz + rng.uniform(12.0, 32.0)
        el = rng.uniform(-0.14, 0.16)
        tx = ox + r0 * math.cos(el) * math.cos(bearing)
        ty = oy + r0 * math.cos(el) * math.sin(bearing)
        tz = max(gz + 4.0, cz + r0 * math.sin(el))
        # target crosses the line of sight, which is what the motion channel is
        # meant to key on and what a stationary evader never provides
        cross = bearing + math.pi / 2
        tspeed = rng.uniform(4.0, 9.0)
        tvel = (tspeed * math.cos(cross) * rng.choice((1.0, -1.0)),
                tspeed * math.sin(cross) * rng.choice((1.0, -1.0)),
                rng.uniform(-2.0, 2.0))
        out.append(dict(cam=(ox, oy, cz), yaw=bearing, tgt=(tx, ty, tz),
                        tvel=tvel, closing=rng.uniform(8.0, 14.0),
                        bearing=bearing))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sock", default=host_socket())
    # "label=kind:weights", e.g. "ft=fusion:work/runs/sim-fusion-m-p2/weights/best.pt".
    # Free-form so a fine-tune can be scored against both of its parents in the
    # same pass over the same frames -- comparing across separate runs would
    # confound the model with the pose sequence and the sky.
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=KIND:PATH")
    ap.add_argument("--nano", default="work/runs/sim-n-p2/weights/best.pt")
    ap.add_argument("--fusion",
                    default="work/runs/combined-fusion-m-p2-2/weights/best.pt")
    ap.add_argument("--passes", type=int, default=26)
    ap.add_argument("--frames", type=int, default=26)
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--acquire", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=1440)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    c = SimClient(a.sock, timeout_s=900)
    info = c.info()
    gz, (ox, oy) = float(info["ground_z"]), info["origin_xy"]
    scene = info.get("scene", "?")

    if a.model:
        models = {}
        for spec in a.model:
            label, _, rest = spec.partition("=")
            kind, _, path = rest.partition(":")
            models[label] = (FusionDetector(path, tile=640, conf=a.conf)
                             if kind == "fusion"
                             else YoloDetector(path, imgsz=a.imgsz, conf=a.conf))
    else:
        models = {
            "nano-p2 (2.9M, sim-only)": YoloDetector(a.nano, imgsz=a.imgsz,
                                                     conf=a.conf),
            "fusion m-p2 (25M, real)": FusionDetector(a.fusion, tile=640,
                                                      conf=a.conf),
        }
    rng = random.Random(a.seed)
    stats = {k: defaultdict(lambda: {"n": 0, "rec": 0, "acq": 0}) for k in models}
    dt = 1.0 / 20.0

    for pi, p in enumerate(_passes(rng, a.passes, ox, oy, gz)):
        c.call("set_sky", sky=SKIES[pi % len(SKIES)])
        for m in models.values():
            if hasattr(m, "reset"):
                m.reset()
        cam = list(p["cam"])
        tgt = list(p["tgt"])
        for fi in range(a.frames):
            for k in range(3):
                tgt[k] += p["tvel"][k] * dt
            los = [tgt[k] - cam[k] for k in range(3)]
            n = math.sqrt(sum(v * v for v in los)) or 1.0
            for k in range(3):
                cam[k] += los[k] / n * p["closing"] * dt
            yaw = math.atan2(los[1], los[0])
            header, frame = c.step({"xyz": cam, "yaw": yaw},
                                   {"xyz": tgt, "yaw": p["bearing"] + math.pi})
            gt = header["gt"]
            if not gt.get("visible") or not gt.get("uv"):
                continue
            span = float(gt.get("span_px") or 0.0)
            key = _bucket(span)
            gx, gy = gt["uv"]
            tau = max(14.0, span)
            for name, model in models.items():
                boxes = model.detect(frame, pi * 1000 + fi, gt)
                hit = 0.0
                for b in boxes:
                    cxb, cyb = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
                    if math.hypot(cxb - gx, cyb - gy) <= tau:
                        hit = max(hit, b.score)
                if key:
                    s = stats[name][key]
                    s["n"] += 1
                    s["rec"] += int(hit >= 0.05)
                    s["acq"] += int(hit >= a.acquire)
        done = sum(v["n"] for v in stats[next(iter(models))].values())
        print(f"  pass {pi + 1}/{a.passes}  frames scored={done}", flush=True)
    c.close()

    print(f"\nscene={scene}   recall@0.05 / acquire@{a.acquire}\n")
    hdr = f"{'span px':<10}{'n':>6}"
    for name in models:
        hdr += f"{name:>28}"
    print(hdr)
    print("-" * len(hdr))
    keys = [f"{lo}-{hi if hi < 10 ** 5 else '+'}" for lo, hi in BUCKETS]
    for key in keys:
        ns = [stats[m][key]["n"] for m in models]
        if not any(n >= 6 for n in ns):
            continue
        row = f"{key:<10}{max(ns):>6}"
        for name in models:
            s = stats[name][key]
            if s["n"]:
                row += f"{s['rec'] / s['n']:>15.3f} /{s['acq'] / s['n']:>10.3f}"
            else:
                row += f"{'-':>28}"
        print(row)
    print("-" * len(hdr))
    tot = f"{'ALL':<10}"
    summary = {}
    for name in models:
        n = sum(v["n"] for v in stats[name].values())
        rec = sum(v["rec"] for v in stats[name].values())
        acq = sum(v["acq"] for v in stats[name].values())
        summary[name] = {"n": n, "recall": rec / max(1, n), "acquire": acq / max(1, n)}
        tot += f"{rec / max(1, n):>15.3f} /{acq / max(1, n):>10.3f}"
    print(f"{tot[:10]}{sum(v['n'] for v in stats[next(iter(models))].values()):>6}"
          f"{tot[10:]}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"scene": scene, "summary": summary,
             "buckets": {m: dict(stats[m]) for m in models}}, indent=2), encoding="utf-8")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
