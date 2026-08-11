"""Where does the chaser actually pass the target, and is it biased?

A hit radius of one metre hides a lot. Measured across 223 rendered intercepts
the aircraft passes a consistent few centimetres *under* its target -- small
against the radius, but a bias rather than scatter (positive in 85 percent of
runs), and a bias is a property of the law that will not shrink just because the
tolerance is generous.

This reports the signed pass geometry over a sandbox suite, so the question can
be asked against a *perfect* sensor. That separation is the point: whatever
survives here is guidance or airframe, not perception, and cannot be argued away
as a detector artefact.

    python -m pursuit.tools.aimbias --suite stress
    python -m pursuit.tools.aimbias --suite stress --sweep vertical_lead_s=0,0.1,0.2
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pursuit.episode import ScenarioConfig  # noqa: E402
from pursuit.evader import EvaderConfig  # noqa: E402
from pursuit.guidance import GuidanceConfig  # noqa: E402
from pursuit.sandbox import (RING_INTRINSICS, SIM_INTRINSICS,  # noqa: E402
                             SyntheticCamera, SyntheticRing, build_suite,
                             run_episode)


def _stats(xs):
    if not xs:
        return None
    n = len(xs)
    mean = st.fmean(xs)
    sd = st.pstdev(xs) if n > 1 else 0.0
    sem = sd / (n ** 0.5) if n else 0.0
    return {"n": n, "mean": mean, "sd": sd,
            "t": (mean / sem) if sem > 1e-12 else 0.0,
            "pos": sum(1 for x in xs if x > 0.0)}


def measure(suite: str, gcfg: GuidanceConfig, ecfg: EvaderConfig,
            realistic: bool = True, ring: bool = False) -> dict:
    """Fly ``suite`` and collect the signed pass geometry of every intercept."""
    cam_kw = dict(span_bias=0.92, noise_px=0.6, span_noise=0.10) if realistic else {}
    out = {"vertical": [], "lateral": [], "along": [], "cpa": [],
           "hits": 0, "n": 0, "by_policy": {}}
    for sc in build_suite(suite, ScenarioConfig()):
        if ring:
            from pursuit.ring import default_ring
            cam = SyntheticRing(default_ring(RING_INTRINSICS), seed=sc.seed,
                                min_span_px=3.0, **cam_kw)
        else:
            cam = SyntheticCamera(SIM_INTRINSICS, seed=sc.seed, **cam_kw)
        r = run_episode(sc, gcfg, ecfg, cam)
        out["n"] += 1
        if not r.success or r.pass_vertical_m is None:
            continue
        out["hits"] += 1
        out["vertical"].append(r.pass_vertical_m)
        out["lateral"].append(r.pass_lateral_m)
        out["along"].append(r.pass_along_m)
        out["cpa"].append(r.pass_cpa_m)
        out["by_policy"].setdefault(sc.policy, []).append(r.pass_vertical_m)
    return out


def report(tag: str, m: dict, by_policy: bool = False) -> None:
    v, la, al = _stats(m["vertical"]), _stats(m["lateral"]), _stats(m["along"])
    if not v:
        print(f"{tag}: no intercepts")
        return
    print(f"\n{tag}   {m['hits']}/{m['n']} intercepts   "
          f"mean CPA {st.fmean(m['cpa']):.3f} m")
    print(f"  {'axis':<10}{'mean m':>10}{'sd':>9}{'t':>8}{'sign+':>10}")
    for name, s in (("vertical", v), ("lateral", la), ("along", al)):
        print(f"  {name:<10}{s['mean']:>+10.4f}{s['sd']:>9.4f}{s['t']:>8.1f}"
              f"{s['pos']}/{s['n']:<9}")
    hi = "above" if v["mean"] > 0 else "below"
    print(f"  -> passes {abs(v['mean']) * 100:.1f} cm {hi} the target"
          f"{'  (SIGNIFICANT)' if abs(v['t']) > 3.0 else '  (not significant)'}")
    if by_policy:
        print("  by policy:")
        for p, xs in sorted(m["by_policy"].items()):
            s = _stats(xs)
            print(f"    {p:<12}{s['mean']:>+9.4f}  n={s['n']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="stress")
    ap.add_argument("--sweep", default="",
                    help="field=v1,v2,... on GuidanceConfig")
    ap.add_argument("--perfect", action="store_true",
                    help="noiseless camera (isolates guidance from perception)")
    ap.add_argument("--by-policy", action="store_true")
    ap.add_argument("--ring", action="store_true",
                    help="the four-camera ring and omnidirectional guidance")
    a = ap.parse_args(argv)

    ecfg = EvaderConfig()
    base = GuidanceConfig()
    if a.ring:
        from pursuit.city import city_guidance, city_top_speed
        scs = build_suite(a.suite, ScenarioConfig())
        base = (city_guidance(base, city_top_speed(scs))
                if a.suite.startswith("city")
                else replace(base, omnidirectional=True))
        ecfg = replace(ecfg, arena_radius_m=max(ecfg.arena_radius_m, 400.0))
    if not a.sweep:
        report(f"[{a.suite}]",
               measure(a.suite, base, ecfg, not a.perfect, a.ring),
               a.by_policy)
        return 0

    field, _, vals = a.sweep.partition("=")
    for raw in vals.split(","):
        val = float(raw)
        cfg = replace(base, **{field.strip(): val})
        report(f"[{a.suite}] {field}={val:g}",
               measure(a.suite, cfg, ecfg, not a.perfect, a.ring),
               a.by_policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
