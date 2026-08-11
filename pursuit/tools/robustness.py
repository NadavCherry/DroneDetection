#!/usr/bin/env python3
"""How bad can the camera get before the intercept stops working?

The sandbox proves the guidance law converges against a *perfect* sensor. That
is necessary and it is not interesting -- the detector will miss frames, wobble
the box, read the span wrong and arrive late, and the only question that matters
is which of those the law survives and at what magnitude.

So each row here degrades exactly one property and sweeps it until the intercept
rate falls over. The number that comes out is a **budget**: "tolerates 45 percent
dropout" is a specification the perception side can be held to, and a detector
that meets it needs no further defence. The last row combines everything at once
at a level the individual sweeps say should be survivable, because failures
compound in ways single-axis sweeps never show.

    .venv/bin/python -m pursuit.tools.robustness
    .venv/bin/python -m pursuit.tools.robustness --suite stress --out work/pursuit/robustness.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pursuit.episode import ScenarioConfig
from pursuit.evader import EvaderConfig
from pursuit.guidance import GuidanceConfig
from pursuit.sandbox import SIM_INTRINSICS, SyntheticCamera, build_suite, run_episode

# name -> the SyntheticCamera keyword it sweeps, and the values to try.
SWEEPS = [
    ("dropout          (fraction of frames the detector misses)",
     "dropout", [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]),
    ("box noise        (px, 1-sigma on the centre)",
     "noise_px", [0.0, 1.0, 3.0, 6.0, 10.0, 16.0]),
    ("span noise       (fractional 1-sigma -> monocular range wobble)",
     "span_noise", [0.0, 0.1, 0.2, 0.35, 0.5, 0.7]),
    ("span bias        (systematic; 0.92 is this rig's measured value)",
     "span_bias", [1.0, 0.92, 0.8, 0.65, 1.25, 1.5]),
    ("latency          (frames between geometry and report, 50 ms each)",
     "latency_frames", [0, 1, 2, 3, 5, 8]),
    ("detection floor  (px span below which nothing is reported)",
     "min_span_px", [4.0, 6.0, 8.0, 12.0, 18.0, 25.0]),
]

COMBINED = {"dropout": 0.3, "noise_px": 4.0, "span_noise": 0.25,
            "span_bias": 0.92, "latency_frames": 2, "min_span_px": 8.0}


def run(suite: str, base: ScenarioConfig, ecfg: EvaderConfig,
        gcfg: GuidanceConfig, **cam_kw) -> dict:
    scenarios = build_suite(suite, base)
    # Give guidance the latency its camera actually has for this row. A fixed
    # compensation across a latency sweep measures a pipeline that does not know
    # how stale its own bearings are -- which nobody would deploy, and which
    # made the published budget read far harsher than the law really is.
    lat = cam_kw.get("latency_frames", 0) * 0.05
    gcfg = replace(gcfg, sensor_latency_s=max(gcfg.sensor_latency_s, lat))
    results = []
    for sc in scenarios:
        cam = SyntheticCamera(SIM_INTRINSICS, seed=sc.seed, **cam_kw)
        results.append(run_episode(sc, gcfg, ecfg, cam))
    hits = [r for r in results if r.success]
    tti = [r.time_to_intercept_s for r in hits if r.time_to_intercept_s]
    return {"n": len(results), "hits": len(hits),
            "rate": len(hits) / max(1, len(results)),
            "median_tti_s": round(sorted(tti)[len(tti) // 2], 2) if tti else None,
            "failures": [f"{r.name}:{r.outcome}" for r in results if not r.success]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="full")
    ap.add_argument("--hit-radius", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    base = ScenarioConfig(hit_radius_m=a.hit_radius)
    ecfg = EvaderConfig()
    gcfg = replace(GuidanceConfig(), hit_range_m=a.hit_radius)


    report = {"suite": a.suite, "sweeps": {}, "combined": None}
    for label, key, values in SWEEPS:
        print(f"\n{label}")
        rows = []
        for v in values:
            r = run(a.suite, base, ecfg, gcfg, **{key: v})
            rows.append({key: v, **r})
            mark = "OK " if r["rate"] >= 1.0 else ("~  " if r["rate"] >= 0.9 else "XX ")
            print(f"  {mark}{key}={v:<6} {r['hits']}/{r['n']} "
                  f"({100 * r['rate']:5.1f}%)  median t={r['median_tti_s']}s"
                  + ("" if r["rate"] >= 1.0 else
                     "   " + ", ".join(r["failures"][:4])))
        report["sweeps"][key] = rows

    print(f"\ncombined ({', '.join(f'{k}={v}' for k, v in COMBINED.items())})")
    r = run(a.suite, base, ecfg, gcfg, **COMBINED)
    report["combined"] = {**COMBINED, **r}
    print(f"  {r['hits']}/{r['n']} ({100 * r['rate']:.1f}%) median t={r['median_tti_s']}s")
    if r["failures"]:
        print("   failures: " + ", ".join(r["failures"]))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
