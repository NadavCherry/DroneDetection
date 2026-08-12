"""List Rivermark's buildings and their footprints, for point-defence scenarios.

A defence scenario is only honest if the thing being defended is a thing. The
`defend` suite originally aimed its intruders at bearings picked off a compass,
which put some "buildings" in the middle of a road; this reads the loaded stage
and reports where the actual structures are, so the scenarios can aim at one.

Runs inside the Isaac container against an already-loaded scene:

    docker exec isaac-sim bash -c "cd /tmp/dev/dronedet && /isaac-sim/python.sh \\
        simulators/pegasus/scripts/find_buildings.py --scene rivermark"

Selection is by world-space bounding box rather than by prim name. Rivermark's
naming is inconsistent (``house_A``, ``SM_Building_02``, ``bldg_grp``, plus
instanced proxies), and a name filter silently misses whole streets. A footprint
of at least 6 m on both horizontal axes and at least 4 m tall is a building and
is not a lamp post, a car or a hedge, whatever it happens to be called.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="rivermark")
    ap.add_argument("--min-footprint-m", type=float, default=6.0)
    ap.add_argument("--max-footprint-m", type=float, default=150.0,
                    help="upper bound on the longer horizontal side. Without "
                         "one, the biggest 'buildings' in Rivermark are the "
                         "road network (776 x 702 x 5 m), the terrain subsets "
                         "and the scene root itself -- all of which pass a "
                         "min-size test comfortably and none of which is a "
                         "structure an intruder could fly into")
    ap.add_argument("--min-height-m", type=float, default=8.0)
    ap.add_argument("--max-range-m", type=float, default=120.0,
                    help="only report structures within this of the origin")
    ap.add_argument("--out", default="/tmp/dev/pursuit/buildings.json")
    a = ap.parse_args()

    from isaacsim.simulation_app import SimulationApp  # noqa: E402
    app = SimulationApp({"headless": True})

    from pxr import Usd, UsdGeom  # noqa: E402
    from simulators.pegasus.scenes.outdoor import (  # noqa: E402
        load_outdoor_scene, scene_ground_z, scene_origin_xy)

    load_outdoor_scene(app, a.scene, prim_path="/World/Scene",
                       progress=lambda m: print(m, flush=True))
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    ground = scene_ground_z(a.scene)
    ox, oy = scene_origin_xy(a.scene)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    found = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        try:
            box = cache.ComputeWorldBound(prim)
            rng = box.ComputeAlignedRange()
        except Exception:
            continue
        if rng.IsEmpty():
            continue
        lo, hi = rng.GetMin(), rng.GetMax()
        w, d, h = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
        if min(w, d) < a.min_footprint_m or h < a.min_height_m:
            continue
        if max(w, d) > a.max_footprint_m:
            continue
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        r = math.hypot(cx - ox, cy - oy)
        if r > a.max_range_m:
            continue
        found.append({"path": str(prim.GetPath()), "xy": [round(cx, 2), round(cy, 2)],
                      "range_m": round(r, 1),
                      "bearing_deg": round(math.degrees(math.atan2(cy - oy, cx - ox)), 1),
                      "size_m": [round(w, 1), round(d, 1), round(h, 1)],
                      "top_z": round(hi[2], 2),
                      "height_agl_m": round(hi[2] - ground, 2)})

    # A building made of sub-meshes reports once per level of its hierarchy;
    # keep the outermost, which is the one whose centre is the structure's.
    found.sort(key=lambda b: (-(b["size_m"][0] * b["size_m"][1])))
    kept = []
    for b in found:
        if all(math.dist(b["xy"], k["xy"]) > 8.0 for k in kept):
            kept.append(b)
    kept.sort(key=lambda b: b["range_m"])

    print(f"\n{len(kept)} structures within {a.max_range_m:.0f} m of the origin\n")
    print(f"{'range':>7}{'bearing':>9}{'w x d x h':>20}{'top AGL':>9}  path")
    for b in kept[:40]:
        w, d, h = b["size_m"]
        print(f"{b['range_m']:>7.1f}{b['bearing_deg']:>9.1f}"
              f"{f'{w:.0f} x {d:.0f} x {h:.0f}':>20}{b['height_agl_m']:>9.1f}  "
              f"{b['path'][-58:]}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"scene": a.scene, "ground_z": ground,
                                       "origin_xy": [ox, oy],
                                       "buildings": kept}, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
