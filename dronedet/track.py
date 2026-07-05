"""Camera-motion-compensated multi-target tracker for tiny objects.

Operates in *stabilized* coordinates (per-frame global shift comes with the
detection JSON), so the Kalman constant-velocity model sees true target
motion, not camera drift. Association is Hungarian on center distance
(appearance is meaningless at 4 px). Confirmed tracks coast through misses
and a local re-acquisition stage searches a small window around the
prediction in a short-term median background difference -- recovering
targets that dip below the global detector's threshold.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .detections import Detection, DetectionSet
from .video import frames

# lifecycle
CONFIRM_HITS = 3
TENTATIVE_MAX_MISS = 3
CONFIRMED_MAX_MISS = 45
GATE_BASE = 16.0
GATE_PER_MISS = 1.5
GATE_MAX = 48.0

# re-acquisition: a *strict* local search -- it recovers a briefly-faded
# target near the prediction, and must never latch onto neighbouring clutter
REACQ_AFTER = 3          # start local search after this many consecutive misses
REACQ_HALF = 18          # search half-window around the prediction (px)
REACQ_SNR = 3.8
REACQ_UNIQUE = 0.75      # 2nd peak must be < this fraction of the 1st
REACQ_MAX_CONSEC = 12    # then the real detector must re-feed the track
REACQ_SCORE = 0.30       # synthesized detection score

# track post-filters
MIN_TRACK_FRAMES = 10    # too short = gust
APPEARANCE_KEEP = 0.55   # tracks with this mean detector confidence are kept
#                          regardless of kinematics (a hovering/landed drone
#                          confirmed by the detector must not be filtered)
# Foliage jitter is zero-mean; a flying object sustains direction. A track
# passes if some window of its *real-detection* positions moves >= NET px
# with straightness net/path >= DIRECTEDNESS. (A drone that only ever hovers
# must be caught by the detector, not kinematics.) Birds pass too -- bird
# vs drone is an appearance/classifier problem, not a kinematic one.
DIRECT_WINDOW = 40
DIRECT_MIN = 0.55
DIRECT_NET_MIN = 25.0
MERGE_DIST = 6.0         # median distance over common frames below this =
#                          duplicates riding the same detections (~1 px);
#                          kept tight so objects flying in formation
#                          (e.g. a bird 10-20 px from the drone) never merge


class Kalman:
    """Constant-velocity [cx, cy, vx, vy] filter with position measurement."""

    def __init__(self, cx: float, cy: float, q: float = 0.35, r: float = 1.8):
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([4.0, 4.0, 4.0, 4.0])
        self.F = np.eye(4)
        self.F[0, 2] = self.F[1, 3] = 1.0
        self.Q = np.diag([0.25, 0.25, 1.0, 1.0]) * q
        self.H = np.zeros((2, 4))
        self.H[0, 0] = self.H[1, 1] = 1.0
        self.R = np.eye(2) * r * r

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z) -> None:
        y = np.asarray(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


@dataclass
class Track:
    tid: int
    kf: Kalman
    w: float
    h: float
    score: float
    hits: int = 1
    misses: int = 0
    consec_reacq: int = 0
    confirmed: bool = False
    frames: dict[int, tuple] = field(default_factory=dict)  # stabilized coords

    @property
    def gate(self) -> float:
        return min(GATE_BASE + GATE_PER_MISS * self.misses, GATE_MAX)


class Reacquirer:
    """Short-term median background over stabilized grays for local search."""

    def __init__(self, maxlen: int = 12, stride: int = 4):
        self.samples: deque[np.ndarray] = deque(maxlen=maxlen)
        self.stride = stride
        self.n = 0

    def push(self, gray_stab: np.ndarray) -> None:
        self.n += 1
        if self.n % self.stride == 1:
            self.samples.append(gray_stab.astype(np.float32))

    def search(self, gray_stab: np.ndarray, cx: float, cy: float):
        if len(self.samples) < 4:
            return None
        h, w = gray_stab.shape
        x0, y0 = int(max(0, cx - REACQ_HALF)), int(max(0, cy - REACQ_HALF))
        x1, y1 = int(min(w, cx + REACQ_HALF)), int(min(h, cy + REACQ_HALF))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        stack = np.stack([s[y0:y1, x0:x1] for s in self.samples])
        bg = np.median(stack, axis=0)
        mad = np.median(np.abs(stack - bg), axis=0)
        sigma = np.maximum(1.4826 * mad, 1.5)
        snr = np.abs(gray_stab[y0:y1, x0:x1].astype(np.float32) - bg) / sigma
        _, peak, _, loc = cv2.minMaxLoc(snr)
        if peak < REACQ_SNR:
            return None
        # uniqueness: in clutter there is never a single clean peak
        masked = snr.copy()
        my0, my1 = max(0, loc[1] - 4), min(masked.shape[0], loc[1] + 5)
        mx0, mx1 = max(0, loc[0] - 4), min(masked.shape[1], loc[0] + 5)
        masked[my0:my1, mx0:mx1] = 0
        _, peak2, _, _ = cv2.minMaxLoc(masked)
        if peak2 >= REACQ_UNIQUE * peak:
            return None
        return float(x0 + loc[0]), float(y0 + loc[1]), float(peak)


class Tracker:
    def __init__(self, min_score: float = 0.25, new_track_score: float | None = None):
        self.min_score = min_score
        self.new_track_score = new_track_score if new_track_score is not None else min_score
        self.tracks: list[Track] = []
        self.done: list[Track] = []
        self._next_id = 1

    def step(self, frame_idx: int, dets_stab: list[Detection],
             reacq: Reacquirer | None = None,
             gray_stab: np.ndarray | None = None) -> None:
        dets = [d for d in dets_stab if d.score >= self.min_score]
        preds = [t.kf.predict() for t in self.tracks]

        # Hungarian association on gated center distance
        matches, un_t, un_d = [], set(range(len(self.tracks))), set(range(len(dets)))
        if self.tracks and dets:
            cost = np.full((len(self.tracks), len(dets)), 1e6)
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(dets):
                    dist = float(np.hypot(preds[i][0] - d.cx, preds[i][1] - d.cy))
                    if dist <= t.gate:
                        cost[i, j] = dist
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] < 1e6:
                    matches.append((i, j))
                    un_t.discard(i)
                    un_d.discard(j)

        for i, j in matches:
            t, d = self.tracks[i], dets[j]
            t.kf.update([d.cx, d.cy])
            t.w = 0.7 * t.w + 0.3 * d.w
            t.h = 0.7 * t.h + 0.3 * d.h
            t.score = 0.85 * t.score + 0.15 * d.score
            t.hits += 1
            t.misses = 0
            t.consec_reacq = 0
            if t.hits >= CONFIRM_HITS:
                t.confirmed = True
            t.frames[frame_idx] = (*t.kf.x[:2], t.w, t.h, "tracked")

        for i in un_t:
            t = self.tracks[i]
            t.misses += 1
            # local re-acquisition around the prediction
            if (reacq is not None and gray_stab is not None and t.confirmed
                    and t.misses >= REACQ_AFTER
                    and t.consec_reacq < REACQ_MAX_CONSEC):
                found = reacq.search(gray_stab, *t.kf.x[:2])
                if found is not None:
                    x, y, peak = found
                    t.kf.update([x, y])
                    t.hits += 1
                    t.misses = 0
                    t.consec_reacq += 1
                    t.score = 0.9 * t.score + 0.1 * REACQ_SCORE
                    t.frames[frame_idx] = (*t.kf.x[:2], t.w, t.h, "reacq")
                    continue
            if t.confirmed and t.misses <= CONFIRMED_MAX_MISS:
                t.frames[frame_idx] = (*t.kf.x[:2], t.w, t.h, "coast")

        for j in un_d:
            d = dets[j]
            if d.score < self.new_track_score:
                continue
            t = Track(self._next_id, Kalman(d.cx, d.cy), d.w, d.h, d.score)
            t.frames[frame_idx] = (d.cx, d.cy, d.w, d.h, "tracked")
            self.tracks.append(t)
            self._next_id += 1

        keep = []
        for t in self.tracks:
            dead = (t.misses > CONFIRMED_MAX_MISS if t.confirmed
                    else t.misses > TENTATIVE_MAX_MISS)
            (self.done if dead else keep).append(t)
        self.tracks = keep

    def finish(self) -> list[Track]:
        self.done.extend(self.tracks)
        self.tracks = []
        return postprocess([t for t in self.done if t.confirmed])


def directedness(t: Track) -> tuple[float, float]:
    """Best (net/path, net) over sliding windows of consecutive
    real-detection positions. Coast/reacq frames follow the prediction and
    would be artificially straight, so they are excluded."""
    items = sorted((f, v) for f, v in t.frames.items() if v[4] == "tracked")
    runs, cur = [], []
    for f, v in items:
        if cur and f - cur[-1][0] > 5:
            runs.append(cur)
            cur = []
        cur.append((f, v))
    if cur:
        runs.append(cur)
    best, best_net = 0.0, 0.0
    for run in runs:
        pts = np.array([(v[0], v[1]) for _, v in run])
        n = len(pts)
        if n < 15:
            continue
        w = min(DIRECT_WINDOW, n)
        for i in range(0, n - w + 1, max(1, w // 4)):
            seg = pts[i:i + w]
            steps = np.hypot(*np.diff(seg, axis=0).T)
            path = float(steps.sum())
            net = float(np.hypot(*(seg[-1] - seg[0])))
            if path > 1e-6 and net >= DIRECT_NET_MIN and net / path > best:
                best, best_net = net / path, net
    return best, best_net


def postprocess(tracks: list[Track]) -> list[Track]:
    """Clutter rejection (kinematic, unless appearance-confirmed) + merging."""
    kept = [t for t in tracks
            if len(t.frames) >= MIN_TRACK_FRAMES
            and (t.score >= APPEARANCE_KEEP or directedness(t)[0] >= DIRECT_MIN)]

    # merge concurrent duplicates. Median distance over common frames: a
    # track that spawned on clutter and then latched onto another track's
    # target rides it for most of its life -- the median ignores the short
    # pre-capture segment that would drag a mean above threshold.
    kept.sort(key=lambda t: -len(t.frames))
    out: list[Track] = []
    for t in kept:
        dup = False
        for k in out:
            common = set(t.frames) & set(k.frames)
            if len(common) < 5:
                continue
            d = np.median([math_hypot(t.frames[f], k.frames[f]) for f in common])
            if d < MERGE_DIST:
                for f, v in t.frames.items():  # absorb non-overlapping part
                    k.frames.setdefault(f, v)
                dup = True
                break
        if not dup:
            out.append(t)
    return out


def math_hypot(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def run_tracker_file(video: str, dets_path: str, out: str,
                     video_out: str | None = None,
                     min_score: float = 0.25) -> None:
    ds = DetectionSet.load(dets_path)
    shifts = {int(k): v for k, v in ds.meta.get("shifts", {}).items()}
    # full 2x3 current->reference transforms (affine-aware). When present they
    # compensate camera rotation/zoom, not just translation; else fall back to
    # the translation-only shifts so old detection JSONs still track.
    transforms = {int(k): np.asarray(v, dtype=np.float64).reshape(2, 3)
                  for k, v in ds.meta.get("transforms", {}).items()}

    def frame_M(i: int) -> np.ndarray:
        M = transforms.get(i)
        if M is not None:
            return M
        dx, dy = shifts.get(i, (0.0, 0.0))
        return np.float64([[1, 0, dx], [0, 1, dy]])

    tracker = Tracker(min_score=min_score)
    reacq = Reacquirer()
    t0 = time.perf_counter()
    n = 0
    for idx, frame in frames(video):
        M = frame_M(idx)                          # full 2x3 current->reference
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_stab = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]),
                                   borderMode=cv2.BORDER_REPLICATE)
        scale = float(np.sqrt(abs(np.linalg.det(M[:, :2])))) or 1.0   # box-size compensation
        dets_stab = []
        for d in ds.frames.get(idx, []):
            sx = M[0, 0] * d.cx + M[0, 1] * d.cy + M[0, 2]            # -> reference coords
            sy = M[1, 0] * d.cx + M[1, 1] * d.cy + M[1, 2]
            hw, hh = d.w * scale / 2, d.h * scale / 2
            dets_stab.append(Detection(sx - hw, sy - hh, sx + hw, sy + hh, d.score, d.label))
        tracker.step(idx, dets_stab, reacq=reacq, gray_stab=gray_stab)
        reacq.push(gray_stab)
        n += 1
    confirmed = tracker.finish()
    elapsed = time.perf_counter() - t0

    payload = {
        "video": video,
        "dets": ds.method,
        "min_score": min_score,
        "meta": {"fps": round(n / elapsed, 2), "n_confirmed_tracks": len(confirmed)},
        "tracks": [],
    }
    for t in confirmed:
        tf = {}
        for f, (cx, cy, w, h, status) in sorted(t.frames.items()):
            Minv = cv2.invertAffineTransform(frame_M(f))   # reference -> original coords
            ox = Minv[0, 0] * cx + Minv[0, 1] * cy + Minv[0, 2]
            oy = Minv[1, 0] * cx + Minv[1, 1] * cy + Minv[1, 2]
            iscale = float(np.sqrt(abs(np.linalg.det(Minv[:, :2])))) or 1.0
            tf[str(f)] = [round(ox, 2), round(oy, 2),
                          round(w * iscale, 2), round(h * iscale, 2), status]
        payload["tracks"].append({"id": t.tid, "n": len(tf), "score": round(t.score, 3),
                                  "frames": tf})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload))
    print(f"{len(confirmed)} confirmed tracks -> {out} ({payload['meta']['fps']} fps)")

    if video_out:
        from .render import render_tracks

        render_tracks(video, out, video_out)
