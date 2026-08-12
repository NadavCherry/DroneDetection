#!/usr/bin/env python3
"""Record the city-defence mission end to end: boot, fly, score, publish.

One command, because the run has to be reproducible by somebody who was not
here. It boots Rivermark with the four-camera ring, flies the whole arrival
circle, records a clip from ten directions spread around it, and writes the
scorecard that says how many intruders were stopped and how many buildings were
hit.

    .venv/bin/python -m pursuit.tools.record_city \\
        --weights work/runs/sim-n-p2/weights/best.pt

What lands in ``work/pursuit/city/``:

    METRICS.md      intercepted vs struck, per building, per bearing, timing
    headline.mp4    the clip for the README
    city-XXX.mp4    ten engagements, one every ~36 degrees of the compass
    results.json    every scenario's telemetry summary
    gallery/        all the clips on one page

Ten clips rather than the first ten scenarios, and that is not cosmetic: the
suite is ordered by arrival bearing, so "the first ten" would be ten
neighbouring directions attacking the same two buildings -- the least
informative ten clips the run could produce.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulators.pegasus.pursuit_proto import host_socket  # noqa: E402

VENV = str(ROOT / ".venv/bin/python")
SOCK = Path(host_socket())

# Every ~36 degrees of the compass, out of the 24 the suite flies.
VIDEO_BEARINGS = (0, 30, 75, 105, 150, 180, 210, 255, 285, 330)


def _sh(cmd: str, timeout: int = 180) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ""


def sim_is_up() -> bool:
    return SOCK.exists()


def boot_sim(scene: str, fps: int, wait_s: int = 600) -> bool:
    """Start the ring server, unless one is already answering.

    Reusing a live server is the difference between a two-minute iteration and
    a fifteen-minute one, and Rivermark's warm load is 26 s of that.
    """
    if sim_is_up():
        print("simulator already up; reusing it")
        return True
    _sh("docker exec isaac-sim bash -c 'pkill -f pursuit_server.py'", 60)
    time.sleep(10)
    _sh(f"docker exec -d isaac-sim bash -c \"cd /tmp/dev/dronedet && "
        f"/isaac-sim/python.sh simulators/pegasus/scripts/pursuit_server.py "
        f"--scene {scene} --cameras ring --fps {fps} "
        f"> /tmp/dev/pursuit/server_ring_{scene}.log 2>&1\"", 60)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if SOCK.exists():
            time.sleep(20)          # the socket binds before the first render
            return True
        time.sleep(5)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None,
                    help="appearance model; without one the run is motion-only")
    ap.add_argument("--detector", default=None,
                    choices=["yolo", "fusion", "motion", "oracle"],
                    help="default: yolo when --weights is given, else motion")
    ap.add_argument("--suite", default="city")
    ap.add_argument("--scene", default="rivermark")
    ap.add_argument("--out", default="work/pursuit/city")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--init-score", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=2048)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--extra", default="", help="passed through to run_pursuit")
    a = ap.parse_args(argv)

    detector = a.detector or ("yolo" if a.weights else "motion")
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    if not boot_sim(a.scene, a.fps):
        print("!! the simulator never came up")
        return 1

    names = ",".join(f"city-{b:03d}" for b in VIDEO_BEARINGS)
    cmd = (f"{VENV} -m pursuit.run_pursuit --detector {detector} "
           f"--suite {a.suite} --out {out} --conf {a.conf} "
           f"--init-score {a.init_score} --imgsz {a.imgsz} ")
    if a.weights:
        cmd += f"--weights {a.weights} "
    if a.limit:
        cmd += f"--limit {a.limit} "
    if not a.no_video:
        cmd += f"--video --video-names {names} "
    cmd += a.extra
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=str(ROOT))

    rj = out / "results.json"
    if not rj.exists():
        print("!! no results.json -- nothing to report")
        return 1
    payload = json.loads(rj.read_text(encoding="utf-8"))
    print(f"\n== {payload.get('hits')}/{payload.get('n')} intruders intercepted")

    # The headline clip: a save, and among saves the one that shows the most.
    # A long engagement means the intruder was caught far out with the whole
    # find-close-hit sequence visible; a two-second one shows only the endgame.
    best = None
    for r in payload["results"]:
        clip = out / f"{r['name']}.mp4"
        if not (r.get("success") and clip.exists()):
            continue
        score = r.get("time_to_intercept_s") or 0.0
        if best is None or score > best[0]:
            best = (score, clip, r)
    if best:
        shutil.copy2(best[1], out / "headline.mp4")
        print(f"headline: {best[1].name} ({best[2].get('time_to_intercept_s')} s, "
              f"CPA {best[2].get('pass_cpa_m')} m, "
              f"{best[2].get('strike_margin_s')} s to spare)")

    subprocess.run(f"{VENV} -m pursuit.tools.city_report --search {a.out} "
                   f"--out {a.out}/METRICS.md", shell=True, cwd=str(ROOT))
    subprocess.run(f"{VENV} -m pursuit.tools.gallery --search {a.out} "
                   f"--out {a.out}/gallery", shell=True, cwd=str(ROOT))
    print(f"\nDONE -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
