#!/usr/bin/env python3
"""Build a labelled Isaac Sim air-to-air dataset by flying the simulator.

The repo's existing detectors do not transfer to this renderer. Measured on the
observer footage from ``run_two_drone.py``, against the rig's own ground truth:
the temporal edge model reaches recall 0.50 with eleven thousand false
positives, the appearance expert reaches **0.00**, and the round-7 fusion model
0.07. That is not a tuning problem -- they were trained on real video of real
drones, and asking them to find a rendered Iris is a domain transfer nobody
asked them to make.

Which is fine, because this domain labels itself. The simulator knows exactly
where the target is, and the ``bounding_box_2d_tight`` annotator measures the
box on the same rendered frame the detector will be handed -- so every frame
comes with a pixel-exact label for free, and the only real cost is render time.

Two samplers, because they cover different failure modes:

``flights``
    Whole pursuits, flown by the real guidance law against the real evader
    policies with an oracle sensor. This is the *deployment distribution*: the
    ranges the chaser actually sees, in the order it sees them, with the target
    swelling from a handful of pixels to filling the frame, against backgrounds
    swinging past as the aircraft yaws. A detector trained only on independent
    random poses is trained on a distribution its user does not have.

``poses``
    Independent random geometry -- span sampled log-uniformly so the 5-pixel and
    the 300-pixel cases get equal weight, aspect angle uniform, position uniform
    across the frame. Flights alone would leave the target near the centre of
    the image (the chaser is trying to point at it) at the ranges the guidance
    law happens to dwell at, and a detector that has only seen centred targets
    fails exactly when acquisition needs it most.

Negatives -- frames with the target out of view or occluded -- are kept rather
than dropped. A detector that has never been shown an empty sky has never been
given a reason not to fire at one, and in a closed loop a false positive is not
a scored error, it is a chaser flying at a cloud.

    # skydome pass (start the server with --scene skydome first)
    .venv/bin/python -m pursuit.tools.make_sim_dataset --out work/simdata --tag sky --flights 24 --poses 2500

    # then restart the server with --scene rivermark and append the town
    .venv/bin/python -m pursuit.tools.make_sim_dataset --out work/simdata --tag town --flights 24 --poses 2500 --append
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pursuit.episode import Episode, ScenarioConfig
from pursuit.evader import POLICIES, EvaderConfig
from pursuit.geometry import Intrinsics
from pursuit.guidance import GuidanceConfig
from pursuit.perception import OracleDetector, Perception
from simulators.pegasus.pursuit_proto import SimClient, host_socket

HOST_SOCKET = host_socket()
SKIES = ["clear", "noon_grass", "partly_cloudy", "cloudy", "overcast",
         "sunrise", "evening", "lakeside", "mealie_road"]


class DatasetWriter:
    """Writes a YOLO-format detection dataset, split by *episode* not by frame.

    A random frame split would be a lie here: consecutive frames of one pursuit
    are near-duplicates, so a validation frame drawn from a training episode has
    a training frame 50 ms either side of it. Splitting whole episodes (and whole
    pose batches) is what makes the validation number mean anything.
    """

    def __init__(self, root: Path, tag: str, val_fraction: float = 0.12,
                 quality: int = 96, neg_target: float = 0.2,
                 seed: int = 0) -> None:
        self.root = Path(root)
        self.tag = tag
        self.val_fraction = float(val_fraction)
        self.quality = int(quality)
        self.neg_target = float(neg_target)
        self.rng = random.Random(seed)
        for split in ("train", "val"):
            (self.root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.root / "labels" / split).mkdir(parents=True, exist_ok=True)
        self.counts = {"train": 0, "val": 0, "neg": 0, "pos": 0, "rejected": 0}
        self.spans: List[float] = []

    def _split_of(self, group: int) -> str:
        """Deterministic episode-level split.

        ``group % 100`` looked like a hash and is not one: groups are handed out
        in order, so a run with a few dozen episodes never produces a number
        above the validation threshold in the training direction -- the first
        attempt at this put every single frame in ``val``. Multiplying by a large
        odd constant first scatters consecutive group numbers across the range,
        which is what was wanted.
        """
        return ("val" if (group * 2654435761) % 100 < int(self.val_fraction * 100)
                else "train")

    def write(self, frame_rgb: np.ndarray, gt: dict, group: int, idx: int,
              min_span_px: float = 1.5) -> bool:
        name = f"{self.tag}_{group:05d}_{idx:05d}"
        bgr = frame_rgb[:, :, ::-1]
        h, w = bgr.shape[:2]

        if not self._label_trustworthy(gt):
            self.counts["rejected"] += 1
            return False

        lines = []
        bbox = gt.get("bbox")
        if bbox and gt.get("visible"):
            x1, y1, x2, y2 = (float(v) for v in bbox)
            bw, bh = x2 - x1, y2 - y1
            if max(bw, bh) >= min_span_px:
                # The annotator's tight box on a thin airframe can be a single
                # pixel high; a degenerate box trains a degenerate regression, so
                # give every box at least two pixels on each side.
                bw, bh = max(bw, 2.0), max(bh, 2.0)
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                lines.append(f"0 {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}")
                self.spans.append(max(bw, bh))

        if not lines and not self._keep_negative():
            # Frames with nothing in them are worth showing a detector -- they
            # are the only reason it has not to fire at an empty sky -- but a
            # pursuit spends half its frames searching, and a dataset that is
            # half background teaches reticence more strongly than detection.
            # Subsample down to a target share instead of taking all or none.
            return False

        split = self._split_of(group)
        cv2.imwrite(str(self.root / "images" / split / f"{name}.jpg"), bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        (self.root / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        self.counts[split] += 1
        self.counts["pos" if lines else "neg"] += 1
        return bool(lines)

    @staticmethod
    def _label_trustworthy(gt: dict) -> bool:
        """Drop a frame whose rendered label disagrees with the geometry.

        The simulator reports two independent things about the target: a box
        measured by the renderer's annotator, and the projection of the pose it
        was told to fly to. They are produced by entirely different machinery,
        so when they agree the label is trustworthy in a way that neither alone
        establishes -- and when they disagree, one of them is wrong and the
        frame is not worth training on.

        This is not hypothetical caution. Rivermark labels 469 objects in a
        typical frame (buildings, roads, kerbs, lane markings), and a version of
        the server that unioned every annotator row produced "ground truth"
        boxes drawn around car parks, in nearly six out of ten frames, while
        every summary statistic still looked plausible. The failure was found by
        plotting spans, which is luck. A frame-level cross-check makes it
        impossible to ship instead.

        The tolerance scales with the target: at 5 px a 20 px disagreement is
        enormous, and at 400 px it is nothing.
        """
        if not gt.get("bbox") or not gt.get("visible"):
            return True                       # negatives have nothing to check
        gap = gt.get("label_gap_px")
        ratio = gt.get("span_ratio")
        if gap is None or ratio is None:
            # No projection to compare against means the target is behind the
            # camera, and a box claiming to be it is then wrong by construction:
            # the frame is a negative that has been handed a positive label.
            # Reject rather than trust.
            return False
        span = float(gt.get("analytic_span_px") or 0.0)
        if gap > max(20.0, 0.75 * span):
            return False
        return 0.35 <= ratio <= 3.0

    def _keep_negative(self) -> bool:
        total = self.counts["pos"] + self.counts["neg"]
        if total < 40:
            return True
        share = self.counts["neg"] / total
        return share < self.neg_target or self.rng.random() < 0.15

    def finish(self, extra: Optional[dict] = None) -> dict:
        (self.root / "data.yaml").write_text(
            f"path: {self.root.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "nc: 1\n"
            "names:\n"
            "  0: drone\n", encoding="utf-8")
        stats = {
            **self.counts,
            "span_px": {
                "n": len(self.spans),
                "min": round(min(self.spans), 1) if self.spans else None,
                "p10": round(float(np.percentile(self.spans, 10)), 1) if self.spans else None,
                "median": round(float(np.median(self.spans)), 1) if self.spans else None,
                "p90": round(float(np.percentile(self.spans, 90)), 1) if self.spans else None,
                "max": round(max(self.spans), 1) if self.spans else None,
            },
            **(extra or {}),
        }
        path = self.root / f"stats_{self.tag}.json"
        path.write_text(json.dumps(stats, indent=1), encoding="utf-8")
        return stats


# ------------------------------------------------------------------- samplers

def sample_poses(client: SimClient, info: dict, writer: DatasetWriter, n: int,
                 rng: random.Random, group0: int, log) -> int:
    """Independent random geometry: uniform aspect, log-uniform pixel span."""
    intr = Intrinsics.from_dict(info["intrinsics"])
    ground_z = float(info["ground_z"])
    ox, oy = info["origin_xy"]
    span_m = float(info["target_span_m"])

    # Half-angles of the real field of view, so "just outside the frame" is a
    # meaningful category rather than a guess.
    az_max = math.atan2(intr.cx, intr.fx)
    az_min = -math.atan2(intr.width - intr.cx, intr.fx)
    el_max = math.atan2(intr.cy, intr.fy)
    el_min = -math.atan2(intr.height - intr.cy, intr.fy)

    group = group0
    done = 0
    t0 = time.perf_counter()
    while done < n:
        group += 1
        if done % 200 == 0:
            client.call("set_sky", sky=rng.choice(SKIES))
        for k in range(min(25, n - done)):
            # Span drives range, not the other way round: sampling range
            # uniformly would spend most of the dataset on targets of one
            # apparent size, and apparent size is what the detector sees.
            span_px = math.exp(rng.uniform(math.log(3.5), math.log(320.0)))
            r = max(1.2, min(140.0, intr.fx * span_m / span_px))
            # 12 percent of samples deliberately land outside the frame.
            outside = rng.random() < 0.12
            pad = math.radians(6.0)
            az = (rng.uniform(az_min - math.radians(25), az_max + math.radians(25))
                  if outside else rng.uniform(az_min + pad, az_max - pad))
            el = (rng.uniform(el_min - math.radians(15), el_max + math.radians(15))
                  if outside else rng.uniform(el_min + pad, el_max - pad))

            chaser_yaw = rng.uniform(-math.pi, math.pi)
            alt = rng.uniform(10.0, 45.0)
            cz = ground_z + alt
            cx = ox + rng.uniform(-40.0, 40.0)
            cy = oy + rng.uniform(-40.0, 40.0)

            world_az = chaser_yaw + az
            tx = cx + r * math.cos(el) * math.cos(world_az)
            ty = cy + r * math.cos(el) * math.sin(world_az)
            tz = max(ground_z + 1.0, cz + r * math.sin(el))

            header, frame = client.step(
                {"xyz": [cx, cy, cz], "yaw": chaser_yaw},
                {"xyz": [tx, ty, tz], "yaw": rng.uniform(-math.pi, math.pi)})
            writer.write(frame, header["gt"], group, k)
            done += 1
        if done % 500 < 25:
            log(f"  poses {done}/{n} ({done / max(1e-6, time.perf_counter() - t0):.1f} fps)")
    return group


def sample_flights(client: SimClient, info: dict, writer: DatasetWriter, n: int,
                   rng: random.Random, group0: int, log) -> int:
    """Whole oracle-guided pursuits -- the distribution the detector will meet."""
    intr = Intrinsics.from_dict(info["intrinsics"])
    perception = Perception(OracleDetector(), intr)
    group = group0
    for i in range(n):
        group += 1
        if i % 4 == 0:
            client.call("set_sky", sky=rng.choice(SKIES))
        sc = ScenarioConfig(
            name=f"flight{i}", policy=rng.choice(POLICIES), seed=rng.randint(1, 9999),
            start_range_m=rng.uniform(20.0, 75.0),
            start_bearing_deg=rng.uniform(-60.0, 60.0),
            start_elevation_deg=rng.uniform(-18.0, 18.0),
            altitude_m=rng.uniform(15.0, 35.0),
            evader_speed=rng.uniform(6.0, 12.0),
            speed_advantage=rng.uniform(1.3, 1.9),
            max_seconds=rng.uniform(12.0, 26.0))
        frames = {"n": 0}

        def record(frame, gt, est, gs, perc, chaser, target, _g=group, _c=frames):
            writer.write(frame, gt, _g, _c["n"])
            _c["n"] += 1

        ep = Episode(client, info, perception, GuidanceConfig(),
                     EvaderConfig(speed=sc.evader_speed), recorder=record)
        r = ep.run(sc)
        log(f"  flight {i + 1}/{n} {sc.policy:<11} {r.outcome:<12} "
            f"{frames['n']:4d} frames  miss={r.miss_distance_m:.2f}m")
    return group


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--socket", default=HOST_SOCKET)
    ap.add_argument("--out", default="work/simdata")
    ap.add_argument("--tag", default="sky", help="prefix for this scene's files")
    ap.add_argument("--flights", type=int, default=20)
    ap.add_argument("--poses", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-fraction", type=float, default=0.12)
    ap.add_argument("--neg-target", type=float, default=0.2,
                    help="target share of background-only frames")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing dataset (a second scene)")
    a = ap.parse_args(argv)

    t0 = time.perf_counter()

    def log(msg):
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)

    out = Path(a.out)
    if out.exists() and not a.append:
        log(f"note: {out} exists; pass --append to add to it")
    rng = random.Random(a.seed)
    writer = DatasetWriter(out, a.tag, val_fraction=a.val_fraction,
                           neg_target=a.neg_target, seed=a.seed)

    with SimClient(a.socket, timeout_s=600) as client:
        info = client.info()
        log(f"scene={info['scene_name']} {info['intrinsics']['width']}x"
            f"{info['intrinsics']['height']} render_ticks={info.get('render_ticks')}")
        group = 0
        if a.flights:
            log(f"flights: {a.flights}")
            group = sample_flights(client, info, writer, a.flights, rng, group, log)
        if a.poses:
            log(f"poses: {a.poses}")
            group = sample_poses(client, info, writer, a.poses, rng, group, log)

    stats = writer.finish({"scene": info["scene_name"],
                           "elapsed_s": round(time.perf_counter() - t0, 1)})
    log(f"done: {json.dumps(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
