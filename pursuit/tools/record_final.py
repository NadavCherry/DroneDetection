"""Record the finished system: every environment, every arrival, one folder.

This is the deliverable run. It boots each scene in turn, flies the whole
``mission`` suite against it with the strongest detector, and leaves a single
tidy tree behind:

    work/pursuit/final/
      gallery/index.html  every clip, playable, labelled with its outcome
      METRICS.md          the end-to-end table, FPS included
      headline.mp4        the one clip worth putting in the README
      skydome/            clips + telemetry + results.json
      rivermark/

One scene at a time, and that is not a stylistic choice: this machine has 8 GB
of GPU and Isaac alone holds about 5.5 of it, so a second scene and a 25 M
detector do not fit alongside the first. Each scene is booted, flown, and torn
down before the next starts.

Readiness is the socket appearing plus a settle, never a tick count -- Isaac
reports a stage as loaded well before it is renderable, and a mistyped USD path
does not raise, it just quietly gives you an empty world.

    python -m pursuit.tools.record_final --weights work/runs/sim-fusion-m-p2/weights/best.pt
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
SCENES = ("skydome", "rivermark")


def _sh(cmd: str, timeout: int = 120) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ""


def stop_sim() -> None:
    _sh("docker exec isaac-sim bash -c 'pkill -f pursuit_server.py'", 60)
    time.sleep(12)


def boot_sim(scene: str, fps: int, wait_s: int = 480) -> bool:
    stop_sim()
    if SOCK.exists():
        try:
            SOCK.unlink()
        except OSError:
            pass
    _sh(f"docker exec -d isaac-sim bash -c \"cd /tmp/dev/dronedet && "
        f"/isaac-sim/python.sh simulators/pegasus/scripts/pursuit_server.py "
        f"--scene {scene} --fps {fps} > /tmp/dev/pursuit/server_{scene}.log 2>&1\"",
        60)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if SOCK.exists():
            time.sleep(25)          # the socket binds before the first render
            return True
        time.sleep(5)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--detector", default="fusion", choices=["fusion", "yolo", "oracle"])
    ap.add_argument("--suite", default="mission")
    ap.add_argument("--out", default="work/pursuit/final")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--init-score", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=1440)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--video-limit", type=int, default=40)
    ap.add_argument("--purge", action="store_true",
                    help="delete superseded run folders once this run succeeds")
    a = ap.parse_args(argv)

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    scenes = [s.strip() for s in a.scenes.split(",") if s.strip()]
    flown = []

    for scene in scenes:
        print(f"\n{'=' * 62}\n== booting {scene}\n{'=' * 62}", flush=True)
        if not boot_sim(scene, a.fps):
            print(f"!! {scene} never came up; skipping", flush=True)
            continue
        dest = out / scene
        cmd = (f"{VENV} -m pursuit.run_pursuit --detector {a.detector} "
               f"--weights {a.weights} --imgsz {a.imgsz} --conf {a.conf} "
               f"--init-score {a.init_score} --suite {a.suite} "
               f"--out {dest} --video --video-limit {a.video_limit}")
        print(f"$ {cmd}", flush=True)
        proc = subprocess.run(cmd, shell=True, cwd=str(ROOT))
        if (dest / "results.json").exists():
            flown.append(scene)
            payload = json.loads((dest / "results.json").read_text(encoding="utf-8"))
            print(f"== {scene}: {payload.get('hits')}/{payload.get('n')} "
                  f"intercepts", flush=True)
        else:
            print(f"!! {scene} produced no results.json (rc={proc.returncode})",
                  flush=True)

    if not flown:
        print("\nno scene completed; leaving existing artifacts alone")
        return 1

    # -- the one clip for the README ----------------------------------------
    # Rivermark, not skydome, and as a hard requirement rather than a
    # preference. The skydome's ground is an untextured plane under a
    # photographic sky, so the horizon looks real and the floor does not; it is
    # a testbed for "drone against bright sky", not a picture of the world.
    # Rivermark has real ground, real clutter and is the harder case, which
    # makes it both the honest and the better-looking choice.
    headline_scenes = [s for s in ("rivermark",) if s in flown] or flown
    best = None
    for scene in headline_scenes:
        payload = json.loads((out / scene / "results.json").read_text(encoding="utf-8"))
        for r in payload["results"]:
            clip = out / scene / f"{r['name']}.mp4"
            if not (r.get("success") and clip.exists()):
                continue
            # Among those, prefer a long engagement: an intruder that arrives
            # from off-frame and is chased down shows the whole system, where a
            # three-second point-blank intercept shows only the endgame.
            score = r.get("time_to_intercept_s") or 0.0
            if best is None or score > best[0]:
                best = (score, clip, scene, r)
    if best:
        shutil.copy2(best[1], out / "headline.mp4")
        print(f"\nheadline: {best[2]}/{best[1].name} "
              f"({best[3].get('time_to_intercept_s')} s, "
              f"miss {best[3].get('pass_cpa_m')} m)")

    subprocess.run(f"{VENV} -m pursuit.tools.report --search {a.out} "
                   f"--out {a.out}/METRICS.md", shell=True, cwd=str(ROOT))
    subprocess.run(f"{VENV} -m pursuit.tools.gallery --search {a.out} "
                   f"--out {a.out}/gallery", shell=True, cwd=str(ROOT))

    if a.purge:
        # Only ever after a successful run, and only directories this project
        # generated: these are all superseded by the run above (different
        # detector, different guidance, different scenarios).
        for stale in ("show-sky", "show-town", "ladder-town", "videos",
                      "yolo-final", "yolo-full", "yolo-full2", "yolo-sky",
                      "gallery", "oracle-sky", "oracle-full"):
            p = ROOT / "work/pursuit" / stale
            if p.is_dir():
                shutil.rmtree(p)
                print(f"purged {p.relative_to(ROOT)}")

    print(f"\nDONE -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
