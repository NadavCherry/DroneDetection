#!/usr/bin/env python3
"""Prove the camera ring actually sees 360 degrees, and measure what it costs.

Two questions, and neither can be answered by reading the intrinsics:

**Is there a hole?** Four 96-degree cameras *should* cover the circle with 6
degrees to spare at each seam, but "should" is a statement about a spreadsheet.
What is on the wire is a mount quaternion, a render product and a projection,
and a sign error in any of them produces a ring that is missing a quadrant while
every printed number still looks right. So this walks a target all the way
around the aircraft and asks the simulator, at every bearing, which cameras can
see it -- rendered box and analytic projection, separately, because a
disagreement between those two is itself the bug.

**What does it cost?** Four render products instead of one. The number that
matters is seconds per control tick, which decides whether a scenario matrix is
an afternoon or a week.

    .venv/bin/python -m pursuit.tools.ring_probe --range 60 --step-deg 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulators.pegasus.pursuit_proto import SimClient, host_socket  # noqa: E402

HOST_SOCKET = host_socket()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--socket", default=HOST_SOCKET)
    ap.add_argument("--range", type=float, default=60.0)
    ap.add_argument("--altitude", type=float, default=30.0)
    ap.add_argument("--step-deg", type=float, default=3.0)
    ap.add_argument("--chaser-yaw-deg", type=float, default=37.0,
                    help="deliberately not zero: a ring that only works when "
                         "the airframe is aligned with the world axes is a ring "
                         "with a missing rotation in it")
    ap.add_argument("--out", default="work/pursuit/ring_probe.json")
    a = ap.parse_args(argv)

    with SimClient(a.socket) as c:
        info = c.info()
        cams = info.get("cameras") or []
        intr = info["intrinsics"]
        print(f"scene={info['scene_name']} ring={info.get('ring')} "
              f"cameras={[m['name'] for m in cams]}")
        print(f"per camera {intr['width']}x{intr['height']} fx={intr['fx']:.1f} "
              f"HFOV={intr['hfov_deg']}  VFOV={intr['vfov_deg']}")
        print(f"coverage {info.get('coverage_deg')} deg, "
              f"seam overlap {info.get('seam_overlap_deg')} deg\n")

        gz = float(info["ground_z"])
        ox, oy = info["origin_xy"]
        z = gz + a.altitude
        yaw = math.radians(a.chaser_yaw_deg)
        chaser = {"xyz": [ox, oy, z], "yaw": yaw}

        bearings = [i * a.step_deg for i in range(int(round(360.0 / a.step_deg)))]
        rows = []
        c.reset(chaser, {"xyz": [ox + a.range, oy, z], "yaw": 0.0})
        t0 = time.perf_counter()
        for b in bearings:
            ang = math.radians(b)
            tgt = {"xyz": [ox + a.range * math.cos(ang),
                           oy + a.range * math.sin(ang), z],
                   "yaw": ang + math.pi}
            header, _payload = c.call_raw("step", chaser=chaser, target=tgt)
            gt = header["gt"]
            per = gt.get("per_camera") or {gt.get("camera") or "nose": gt}
            seen = [k for k, v in per.items() if v.get("visible")]
            proj = [k for k, v in per.items() if v.get("analytic_in_frame")]
            rows.append({"bearing_deg": b, "rendered": sorted(seen),
                         "analytic": sorted(proj), "owner": gt.get("camera"),
                         "span_px": gt.get("span_px"),
                         "label_gap_px": gt.get("label_gap_px")})
        elapsed = time.perf_counter() - t0

    # -- verdicts ------------------------------------------------------------
    blind_r = [r["bearing_deg"] for r in rows if not r["rendered"]]
    blind_a = [r["bearing_deg"] for r in rows if not r["analytic"]]
    overlaps = [r for r in rows if len(r["analytic"]) > 1]
    gaps = [r for r in rows
            if r["analytic"] and not r["rendered"]]

    print(f"{len(rows)} bearings at {a.range:.0f} m, "
          f"{1000 * elapsed / max(1, len(rows)):.0f} ms per step "
          f"({len(rows) / elapsed:.2f} steps/s)\n")
    print(f"{'bearings with NO camera (rendered)':<42}{len(blind_r)}")
    print(f"{'bearings with NO camera (projected)':<42}{len(blind_a)}")
    print(f"{'bearings seen by two cameras (seam)':<42}{len(overlaps)}"
          f"  -> {100.0 * len(overlaps) / max(1, len(rows)):.1f}% of the circle")
    print(f"{'projected in frame but no rendered box':<42}{len(gaps)}")
    if blind_a:
        print(f"\n  BLIND WEDGE at bearings {blind_a}")
    owners = {}
    for r in rows:
        owners.setdefault(r["owner"], []).append(r["bearing_deg"])
    print("\nwhich camera owns which arc:")
    for k, v in sorted(owners.items(), key=lambda kv: (kv[0] is None, kv[0] or "")):
        print(f"  {str(k):>8}: {len(v):>3} bearings, "
              f"{min(v):>5.0f}..{max(v):>5.0f} deg")

    spans = [r["span_px"] for r in rows if r["span_px"]]
    gapspx = [r["label_gap_px"] for r in rows if r["label_gap_px"] is not None]
    if spans:
        print(f"\nrendered span {min(spans):.1f}..{max(spans):.1f} px "
              f"(analytic {720.32 * 0.47 / a.range:.1f} px at this range)")
    if gapspx:
        print(f"label vs projection gap: mean {sum(gapspx) / len(gapspx):.2f} px, "
              f"max {max(gapspx):.2f} px")

    ok = not blind_a and not blind_r
    print(f"\n{'RING COVERS 360 DEGREES' if ok else 'RING HAS A HOLE'}")
    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"info": info, "range_m": a.range,
                                 "ms_per_step": 1000 * elapsed / max(1, len(rows)),
                                 "rows": rows}, indent=1), encoding="utf-8")
        print(f"wrote {p}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
