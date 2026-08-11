"""Mine the detector's own false positives in Rivermark as training negatives.

Measured on 480 live town frames, the fine-tuned detector produced 182 true
detections and 730 false ones -- four out of five things it reports are kerbs,
rooflines, parked cars and road shadows. No guidance law survives that, and the
motion channel can only halve it (a homography cannot register a translating
camera against a scene with depth, so buildings light it up legitimately).

The fix is to show the network the things it is getting wrong. The last training
round included background tiles, but they were sampled at *random*, and a random
640 tile of a town is mostly empty road -- easy, already correct, and worth
nothing. What teaches is a tile centred on the exact rooftop corner the detector
just called a drone.

So this flies the camera through the scene with **no drone anywhere in frame**,
which makes every single detection false by construction -- no labelling, no
ground-truth matching, no judgement call. Each one becomes a 4-channel tile with
an empty label file.

The frames are flown as continuous trajectories rather than teleported, because
the fourth channel is a difference against t-3 and t-6: a mined negative must
carry the same ego-motion residual it will have to be rejected against at
inference, or it teaches the network about an artefact it will never see.

    python -m pursuit.tools.mine_negatives --scene rivermark --out work/simdata_neg
"""
from __future__ import annotations

import argparse
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sock", default=host_socket())
    ap.add_argument("--weights",
                    default="work/runs/sim-fusion-m-p2/weights/best.pt")
    ap.add_argument("--out", default="work/simdata_neg")
    ap.add_argument("--tag", default="neg")
    ap.add_argument("--passes", type=int, default=40)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--max-per-frame", type=int, default=3,
                    help="cap tiles per frame so one busy rooftop cannot "
                         "dominate the whole mined set")
    ap.add_argument("--val-fraction", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args(argv)

    from pursuit.perception import FusionDetector
    from simulators.pegasus.pursuit_proto import SimClient

    out = ROOT / a.out
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    c = SimClient(a.sock, timeout_s=900)
    info = c.info()
    gz, (ox, oy) = float(info["ground_z"]), info["origin_xy"]
    det = FusionDetector(a.weights, tile=a.tile, conf=a.conf)
    impl = det._impl
    rng = random.Random(a.seed)

    n_tiles = n_det = n_frames = 0
    for pi in range(a.passes):
        c.call("set_sky", sky=SKIES[pi % len(SKIES)])
        det.reset()
        split = "val" if rng.random() < a.val_fraction else "train"

        # A course through the town at engagement speed and altitude.
        heading = rng.uniform(-math.pi, math.pi)
        speed = rng.uniform(8.0, 15.0)
        cam = [ox + rng.uniform(-40.0, 40.0), oy + rng.uniform(-40.0, 40.0),
               gz + rng.uniform(12.0, 34.0)]
        yaw = heading + rng.uniform(-0.4, 0.4)
        yaw_rate = rng.uniform(-0.5, 0.5)
        climb = rng.uniform(-1.5, 1.5)

        for fi in range(a.frames):
            cam[0] += speed * math.cos(heading) * 0.05
            cam[1] += speed * math.sin(heading) * 0.05
            cam[2] = max(gz + 8.0, cam[2] + climb * 0.05)
            yaw += yaw_rate * 0.05
            # The target is parked 4 km away and below the horizon: the
            # renderer still has two aircraft, but not in this camera.
            far = {"xyz": [ox + 4000.0, oy + 4000.0, gz + 2.0], "yaw": 0.0}
            header, frame = c.step({"xyz": cam, "yaw": yaw}, far)
            gt = header["gt"]
            if gt.get("visible"):
                # Should never happen at 4 km; if it does, the frame is not a
                # negative and must not be mined as one.
                continue
            n_frames += 1

            boxes = det.detect(frame, pi * 1000 + fi, gt)
            if not boxes:
                continue
            motion = impl._motion_map(pi * 1000 + fi)
            bgr = np.ascontiguousarray(frame[:, :, ::-1])
            stack = np.dstack([bgr, motion])
            h, w = stack.shape[:2]
            n_det += len(boxes)

            for bi, b in enumerate(sorted(boxes, key=lambda x: -x.score)[:a.max_per_frame]):
                cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
                jx = rng.uniform(-0.25, 0.25) * a.tile
                jy = rng.uniform(-0.25, 0.25) * a.tile
                x0 = int(min(max(cx + jx - a.tile / 2, 0), max(w - a.tile, 0)))
                y0 = int(min(max(cy + jy - a.tile / 2, 0), max(h - a.tile, 0)))
                crop = stack[y0:y0 + a.tile, x0:x0 + a.tile]
                if crop.shape[:2] != (a.tile, a.tile):
                    pad = np.zeros((a.tile, a.tile, 4), np.uint8)
                    pad[:crop.shape[0], :crop.shape[1]] = crop
                    crop = pad
                stem = f"{a.tag}_{pi:04d}_{fi:03d}_{bi}"
                np.save(str(out / "images" / split / f"{stem}.npy"), crop)
                (out / "labels" / split / f"{stem}.txt").write_text("")
                n_tiles += 1

        print(f"  pass {pi + 1}/{a.passes}  frames={n_frames} "
              f"false_dets={n_det} tiles={n_tiles}", flush=True)
    c.close()

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"channels: 4\nnc: 1\nnames:\n  0: drone\n")
    rate = n_det / max(1, n_frames)
    print(f"\n{n_frames} drone-free frames produced {n_det} false detections "
          f"({rate:.2f} per frame)")
    print(f"mined {n_tiles} hard-negative tiles -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
