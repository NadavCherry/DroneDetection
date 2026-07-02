"""REALTIME pipeline definitions (RT-A .. RT-F).

Every pipeline consumes frames sequentially and yields detections in
original-frame coordinates plus per-stage timings. The tracker runs
inline in the runner (it costs ~1 ms/frame).

    RT-A  classical only: lagged-EMA + 3-frame-diff proposals, no NN
    RT-B  proposals -> temporal nano verifier @256 (top-8 crops)
    RT-C  full-frame temporal nano @1280 (no proposal stage)
    RT-D  full-frame temporal nano @640
    RT-E  RT-B with verification every 2nd frame (decimation)
    RT-F  full-frame single-frame nano @1280 (reference: no temporal)

RT-B/E optionally run the PC 'near/big expert' once per second for
landed/hovering drones (amortized cost ~1 ms equivalent).
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from dronedet.detections import Detection, nms
from dronedet.stabilize import invert, apply_to_points

from dronedet.motion import MotionDetector

from .rt_motion import FrameDiff3
from .rt_stabilize import LiteStabilizer


def make_slow_channel():
    """The PROVEN lagged-median slow-mover detector from the PC pipeline,
    tuned for edge: background recomputed every 30 frames (the spike is
    ~30 ms on Orin-class CPU once a second; run it in a worker thread if
    jitter matters). The pure-EMA variant measurably lost slow drifters
    (val proposal recall 0.04 vs 0.57) -- see REALTIME/README."""
    return MotionDetector(backend="median", lag=90, window=240,
                          sample_stride=10, warmup=30, bg_update_every=10,
                          flicker_alpha=0.004, flicker_rate0=0.30,
                          flicker_boost=3.0, max_dets=80)


def rank_isolated(cands, radius=40.0, k=16):
    """Crowding-aware ranking: wind-blown foliage fires in clusters, a real
    aircraft is a lone mover. Score is divided by (1 + 0.8 * neighbours
    within ``radius``), then the top-k survive."""
    out = []
    for c in cands:
        nb = sum(1 for o in cands
                 if o is not c and (o.cx - c.cx) ** 2 + (o.cy - c.cy) ** 2 <= radius ** 2)
        out.append((c.score / (1.0 + 0.8 * nb), c))
    out.sort(key=lambda t: -t[0])
    return [c for _, c in out[:k]]

DT = 6


def _map_back(dets, m):
    minv = invert(m)
    out = []
    for d in dets:
        (x1, y1), (x2, y2) = apply_to_points(minv, [[d.x1, d.y1], [d.x2, d.y2]])
        out.append(Detection(x1, y1, x2, y2, d.score, d.label))
    return out


class RTBase:
    name = "rt-base"

    def __init__(self):
        self.stab = LiteStabilizer()
        self.grays: deque[np.ndarray] = deque(maxlen=2 * DT + 1)
        self.times: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def _tick(self, key, t0):
        self.times[key] = self.times.get(key, 0.0) + time.perf_counter() - t0
        self.counts[key] = self.counts.get(key, 0) + 1

    def pre(self, frame_bgr):
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        m = self.stab.update(gray)
        gray_stab = cv2.warpAffine(gray, m, (gray.shape[1], gray.shape[0]),
                                   borderMode=cv2.BORDER_REPLICATE)
        self.grays.append(gray_stab)
        self._tick("stabilize+warp", t0)
        return gray_stab, m

    def temporal_image(self):
        if len(self.grays) < 2 * DT + 1:
            return None
        return np.dstack([self.grays[0], self.grays[DT], self.grays[-1]])

    def stage_report(self, n_frames):
        rep = {k: 1000 * v / n_frames for k, v in self.times.items()}
        rep["TOTAL ms/frame"] = sum(rep.values())
        return rep


class ClassicPipeline(RTBase):
    """RT-A: no neural network at all."""

    name = "rt-a-classic"

    def __init__(self):
        super().__init__()
        self.slow = make_slow_channel()

    def process(self, idx, frame_bgr):
        gray_stab, m = self.pre(frame_bgr)
        t0 = time.perf_counter()
        cands = rank_isolated(nms(self.slow.process(gray_stab),
                                  dist_thresh=10.0), k=12)
        dets = [Detection(c.x1, c.y1, c.x2, c.y2,
                          min(1.0, c.score / 20.0) * 0.9, "motion")
                for c in cands]
        self._tick("motion", t0)
        return _map_back(dets, m)


class VerifiedPipeline(RTBase):
    """RT-B / RT-E: proposals -> temporal nano verifier on 256 crops."""

    def __init__(self, name, verifier, expert=None, expert_every=30,
                 verify_every=1, max_crops=16, crop=256, confirm_dist=14.0):
        super().__init__()
        self.name = name
        self.verifier = verifier
        self.expert = expert
        self.expert_every = expert_every
        self.verify_every = verify_every
        self.max_crops = max_crops
        self.crop = crop
        self.confirm_dist = confirm_dist
        self.slow = make_slow_channel()

    def process(self, idx, frame_bgr):
        gray_stab, m = self.pre(frame_bgr)
        h, w = gray_stab.shape

        t0 = time.perf_counter()
        cands = rank_isolated(nms(self.slow.process(gray_stab),
                                  dist_thresh=10.0), k=self.max_crops)
        self._tick("motion", t0)

        dets = []
        do_verify = (idx % self.verify_every == 0) and len(self.grays) == 2 * DT + 1
        verdicts = [None] * len(cands)
        if do_verify and cands:
            t0 = time.perf_counter()
            half = self.crop // 2
            crops = []
            for c in cands:
                x0 = int(np.clip(c.cx - half, 0, w - self.crop))
                y0 = int(np.clip(c.cy - half, 0, h - self.crop))
                crops.append((x0, y0, np.ascontiguousarray(np.dstack(
                    [g[y0:y0 + self.crop, x0:x0 + self.crop]
                     for g in (self.grays[0], self.grays[DT], self.grays[-1])]))))
            self._tick("crop build", t0)
            t0 = time.perf_counter()
            results = self.verifier([c[2] for c in crops])
            self._tick("verifier NN", t0)
            for i, (res, (x0, y0, _)) in enumerate(zip(results, crops)):
                best = {"drone": 0.0, "bird": 0.0}
                for cls_name, conf, bx, by in res:
                    if ((bx + x0 - cands[i].cx) ** 2 +
                            (by + y0 - cands[i].cy) ** 2 <= self.confirm_dist ** 2):
                        best[cls_name] = max(best[cls_name], conf)
                verdicts[i] = best

        for c, v in zip(cands, verdicts):
            s_m = min(1.0, c.score / 20.0)
            if v is None:
                dets.append(Detection(c.x1, c.y1, c.x2, c.y2, 0.5 * s_m, "motion"))
            elif v["drone"] >= max(0.5 * v["bird"], 0.10):
                dets.append(Detection(c.x1, c.y1, c.x2, c.y2,
                                      0.5 + 0.5 * max(v["drone"], s_m), "drone+motion"))
            elif v["bird"] >= 0.10:
                dets.append(Detection(c.x1, c.y1, c.x2, c.y2,
                                      0.12 + 0.08 * min(v["bird"], 1.0), "bird"))
            else:
                dets.append(Detection(c.x1, c.y1, c.x2, c.y2, 0.5 * s_m, "motion"))
        dets = _map_back(dets, m)

        if self.expert is not None and idx % self.expert_every == 0:
            t0 = time.perf_counter()
            dets.extend(self.expert(frame_bgr))
            self._tick("expert (amortized)", t0)
        return nms(dets, dist_thresh=10.0)


class FullFramePipeline(RTBase):
    """RT-C / RT-D / RT-F: one detector pass on the whole frame."""

    def __init__(self, name, detector, temporal=True):
        super().__init__()
        self.name = name
        self.detector = detector
        self.temporal = temporal

    def process(self, idx, frame_bgr):
        gray_stab, m = self.pre(frame_bgr)
        if self.temporal:
            img = self.temporal_image()
            if img is None:
                return []
        else:
            img = frame_bgr
        t0 = time.perf_counter()
        res = self.detector([img])[0]
        self._tick("detector NN", t0)
        dets = []
        for cls_name, conf, bx, by in res:
            if cls_name == "drone":
                dets.append(Detection(bx - 12, by - 12, bx + 12, by + 12,
                                      conf, "drone"))
            else:
                dets.append(Detection(bx - 12, by - 12, bx + 12, by + 12,
                                      0.12 + 0.08 * min(conf, 1.0), "bird"))
        return _map_back(dets, m) if self.temporal else dets
