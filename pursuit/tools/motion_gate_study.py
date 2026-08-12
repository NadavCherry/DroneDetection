"""Does "what actually moved" separate the drone from Rivermark's clutter?

The detector locks onto buildings, kerbs and road shadows in the town. The
obvious counter is the one this whole repository is built on: cancel the static
world and keep what moved. The fusion model already computes exactly that as its
fourth channel -- a grid-LK + RANSAC homography registers t-dt and t-2dt onto t
and the channel is the smaller of the two residuals -- but it feeds it to the
network as an input rather than using it to *veto* a box.

This measures whether it can veto. For each frame it takes every detection, asks
whether it is the drone (within the evaluation radius of ground truth) or
clutter, and scores several candidate gate statistics on the motion channel
inside the box:

``peak``        the strongest motion pixel in the box
``mean``        mean motion in the box
``contrast``    box mean minus the mean of a surrounding ring -- the important one,
                because a homography cannot register a *translating* camera in a
                scene with depth, so buildings light the channel up everywhere.
                An absolute threshold would then pass every rooftop; what a drone
                has and a rooftop does not is motion that is *locally distinct*
                from its own surroundings.
``compact``     the fraction of the box's motion energy inside its central third.
                A drone is a blob a few pixels across; a mis-registered building
                edge is a long thin structure that happens to pass through the box.

The output is the only thing that matters for a gate: at a threshold that keeps
95 percent of true detections, how many false ones survive.

    python -m pursuit.tools.motion_gate_study --weights work/runs/sim-fusion-m-p2/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulators.pegasus.pursuit_proto import host_socket  # noqa: E402

SKIES = ("clear", "partly_cloudy", "cloudy", "overcast", "sunrise", "evening",
         "noon_grass", "lakeside", "mealie_road")


def _stats(box, motion) -> dict:
    """Gate statistics for one box against the motion channel."""
    h, w = motion.shape
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
    inner = motion[y1:y2, x1:x2].astype(np.float32)
    if inner.size == 0:
        return {"peak": 0.0, "mean": 0.0, "contrast": 0.0, "compact": 0.0}

    # A ring three times the box, minus the box itself: local background.
    bw, bh = x2 - x1, y2 - y1
    rx1, ry1 = max(0, x1 - bw), max(0, y1 - bh)
    rx2, ry2 = min(w, x2 + bw), min(h, y2 + bh)
    ring = motion[ry1:ry2, rx1:rx2].astype(np.float32)
    ring_sum = float(ring.sum()) - float(inner.sum())
    ring_n = max(1, ring.size - inner.size)

    cx1, cy1 = x1 + bw // 3, y1 + bh // 3
    core = motion[cy1:cy1 + max(1, bh // 3), cx1:cx1 + max(1, bw // 3)]
    tot = float(inner.sum()) or 1.0
    return {
        "peak": float(inner.max()),
        "mean": float(inner.mean()),
        "contrast": float(inner.mean()) - ring_sum / ring_n,
        "compact": float(core.sum()) / tot,
    }


def _roc(true_vals, false_vals, keep=0.95) -> tuple:
    """Threshold retaining ``keep`` of the true set; fraction of false surviving."""
    if not true_vals or not false_vals:
        return (None, None)
    thr = sorted(true_vals)[max(0, int((1.0 - keep) * len(true_vals)) - 1)]
    survive = sum(1 for v in false_vals if v >= thr) / len(false_vals)
    return (thr, survive)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sock", default=host_socket())
    ap.add_argument("--weights",
                    default="work/runs/sim-fusion-m-p2/weights/best.pt")
    ap.add_argument("--passes", type=int, default=22)
    ap.add_argument("--frames", type=int, default=26)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="work/pursuit/motion_gate.json")
    a = ap.parse_args(argv)

    from pursuit.perception import FusionDetector
    from simulators.pegasus.pursuit_proto import SimClient

    c = SimClient(a.sock, timeout_s=900)
    info = c.info()
    gz, (ox, oy) = float(info["ground_z"]), info["origin_xy"]
    scene = (info.get("scene") or {})
    scene_name = scene.get("scene") if isinstance(scene, dict) else scene

    det = FusionDetector(a.weights, tile=640, conf=a.conf)
    impl = det._impl
    rng = random.Random(a.seed)
    true_s, false_s = {k: [] for k in ("peak", "mean", "contrast", "compact")}, \
                      {k: [] for k in ("peak", "mean", "contrast", "compact")}
    n_true = n_false = n_frames = 0

    for pi in range(a.passes):
        c.call("set_sky", sky=SKIES[pi % len(SKIES)])
        det.reset()
        r0 = rng.uniform(25.0, 85.0)
        bearing = rng.uniform(-math.pi, math.pi)
        cz = gz + rng.uniform(14.0, 32.0)
        el = rng.uniform(-0.12, 0.14)
        tgt = [ox + r0 * math.cos(el) * math.cos(bearing),
               oy + r0 * math.cos(el) * math.sin(bearing),
               max(gz + 5.0, cz + r0 * math.sin(el))]
        cam = [ox, oy, cz]
        cross = bearing + math.pi / 2
        sp = rng.uniform(4.0, 9.0)
        tvel = (sp * math.cos(cross) * rng.choice((1.0, -1.0)),
                sp * math.sin(cross) * rng.choice((1.0, -1.0)),
                rng.uniform(-1.5, 1.5))
        closing = rng.uniform(8.0, 13.0)

        for fi in range(a.frames):
            for k in range(3):
                tgt[k] += tvel[k] * 0.05
            los = [tgt[k] - cam[k] for k in range(3)]
            n = math.sqrt(sum(v * v for v in los)) or 1.0
            for k in range(3):
                cam[k] += los[k] / n * closing * 0.05
            yaw = math.atan2(los[1], los[0])
            header, frame = c.step({"xyz": cam, "yaw": yaw},
                                   {"xyz": tgt, "yaw": bearing + math.pi})
            gt = header["gt"]
            boxes = det.detect(frame, pi * 1000 + fi, gt)
            n_frames += 1
            if not boxes:
                continue
            # The motion channel the detector just built for this frame.
            idx = pi * 1000 + fi
            motion = impl._motion_map(idx)
            gx, gy = (gt.get("uv") or [None, None])
            span = float(gt.get("span_px") or 0.0)
            tau = max(14.0, span)
            for b in boxes:
                st = _stats((b.x1, b.y1, b.x2, b.y2), motion)
                is_true = (gt.get("visible") and gx is not None
                           and math.hypot((b.x1 + b.x2) / 2 - gx,
                                          (b.y1 + b.y2) / 2 - gy) <= tau)
                bucket = true_s if is_true else false_s
                for k, v in st.items():
                    bucket[k].append(v)
                n_true += int(is_true)
                n_false += int(not is_true)
        print(f"  pass {pi + 1}/{a.passes}  true={n_true} false={n_false}",
              flush=True)
    c.close()

    print(f"\nscene={scene_name}   {n_frames} frames   "
          f"{n_true} true detections, {n_false} false\n")
    if not n_true or not n_false:
        print("need both classes to say anything")
        return 1
    print(f"{'statistic':<12}{'true mean':>11}{'false mean':>12}"
          f"{'thr@95%TP':>12}{'false kept':>12}")
    print("-" * 59)
    summary = {}
    for k in ("peak", "mean", "contrast", "compact"):
        t, f = true_s[k], false_s[k]
        thr, surv = _roc(t, f, keep=0.95)
        summary[k] = {"true_mean": float(np.mean(t)), "false_mean": float(np.mean(f)),
                      "thr": thr, "false_survive": surv}
        print(f"{k:<12}{np.mean(t):>11.2f}{np.mean(f):>12.2f}"
              f"{(thr if thr is not None else float('nan')):>12.2f}"
              f"{100 * surv:>11.1f}%")
    print("-" * 59)
    best = min(summary, key=lambda k: summary[k]["false_survive"])
    print(f"best gate: '{best}' -- keeps 95% of true detections and removes "
          f"{100 * (1 - summary[best]['false_survive']):.1f}% of false ones")

    # The 95%-retention row alone is misleading here: a handful of true
    # detections carry no motion signal at all (a drone that happens to be
    # crossing slowly, or one seen against a surface the homography registered
    # perfectly), and they drag any high-retention threshold to zero. The
    # operating point is a choice, so print the curve and let it be chosen.
    print("\ntrade-off curve -- false detections surviving, by true-positive retention\n")
    hdr = f"{'keep TP':>9}" + "".join(f"{k:>12}" for k in
                                      ("peak", "mean", "contrast", "compact"))
    print(hdr)
    print("-" * len(hdr))
    curve = {}
    for keep in (0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60):
        row = f"{100 * keep:>8.0f}%"
        curve[keep] = {}
        for k in ("peak", "mean", "contrast", "compact"):
            thr, surv = _roc(true_s[k], false_s[k], keep=keep)
            curve[keep][k] = {"thr": thr, "false_survive": surv}
            row += f"{100 * surv:>11.1f}%"
        print(row)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scene": scene_name, "n_true": n_true,
                               "n_false": n_false, "stats": summary,
                               "curve": {str(k): v for k, v in curve.items()},
                               "raw": {"true": true_s, "false": false_s}}, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
