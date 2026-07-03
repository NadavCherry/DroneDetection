"""Semi-automatic ground-truth builder for data/videos/07_05.mp4.

Tracks a seeded target frame-by-frame with a *local* search in a
non-causal windowed-median difference image (a GT-only luxury: the
background at frame t is the median of frames t±2.5s, so a briefly
hovering target still stands out). Continuity from the seed resolves
object identity; results are verified visually afterwards.

Outputs work/gt.json plus diagnostic contact sheets in work/gt_verify/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.gt import GroundTruth, GTObject
from dronedet.stabilize import Stabilizer, shift_of, warp_to_reference
from dronedet.video import frames

VIDEO = "data/videos/07_05.mp4"
SAMPLE = 5          # background sample stride (frames)
HALF_WIN = 75       # background window half-width (frames)
SEARCH = 28         # local search half-window (px)
SEARCH_LOST = 55    # expanded search when coasting
SNR_ACCEPT = 3.5
MAX_COAST = 90


def load_stabilized():
    stab = Stabilizer("translation")
    grays, shifts = [], []
    for idx, frame in frames(VIDEO):
        m = stab.update(frame)
        g = warp_to_reference(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), m)
        grays.append(g)
        shifts.append(shift_of(m))
    return np.stack(grays), shifts  # uint8 (N,H,W)


def diff_image(grays, t):
    """|frame_t - median(window around t)| normalized by per-pixel MAD sigma."""
    lo, hi = max(0, t - HALF_WIN), min(len(grays), t + HALF_WIN + 1)
    idx = [i for i in range(lo, hi, SAMPLE) if abs(i - t) >= 2]
    stack = grays[idx].astype(np.float32)
    bg = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - bg), axis=0)
    sigma = np.maximum(1.4826 * mad, 1.5)
    return np.abs(grays[t].astype(np.float32) - bg) / sigma


def refine_box(snr, cx, cy, peak):
    """Blob extent at the peak: connected region above 0.35*peak."""
    h, w = snr.shape
    x0, y0 = max(0, int(cx) - 20), max(0, int(cy) - 20)
    x1, y1 = min(w, int(cx) + 21), min(h, int(cy) + 21)
    win = snr[y0:y1, x0:x1]
    mask = (win >= max(2.5, 0.35 * peak)).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return cx, cy, 4.0, 4.0
    # component containing/nearest the peak position
    px, py = int(cx) - x0, int(cy) - y0
    best, bestd = 1, 1e9
    for i in range(1, n):
        d = (cents[i][0] - px) ** 2 + (cents[i][1] - py) ** 2
        if d < bestd:
            best, bestd = i, d
    x, y, ww, hh, _ = stats[best]
    ccx, ccy = cents[best]
    return x0 + float(ccx), y0 + float(ccy), float(max(ww, 3)), float(max(hh, 3))


def track(grays, seed_xy, seed_frame, v0=(0.0, 0.0), reverse=False, name="",
          max_coast=MAX_COAST, snr_accept=SNR_ACCEPT):
    """Track one target from a seed; returns {frame: (cx,cy,w,h,status)}."""
    n, h, w = grays.shape
    order = range(seed_frame, -1, -1) if reverse else range(seed_frame, n)
    pos = np.array(seed_xy, dtype=np.float64)
    vel = np.array(v0, dtype=np.float64)
    out = {}
    coast = 0
    cached_t, cached_diff = -10 ** 9, None
    for t in order:
        # windowed background changes slowly; recompute every 2 frames
        if cached_diff is None or abs(t - cached_t) >= 2:
            cached_diff, cached_t = diff_image(grays, t), t
        snr = cached_diff
        pred = pos + (vel if not reverse else -vel)
        r = SEARCH if coast == 0 else SEARCH_LOST
        # clamp the prediction so the search window stays inside the frame;
        # only give up when the target is confidently outside
        pred[0] = np.clip(pred[0], -20, w + 20)
        pred[1] = np.clip(pred[1], -20, h + 20)
        x0, y0 = int(max(0, pred[0] - r)), int(max(0, pred[1] - r))
        x1, y1 = int(min(w, pred[0] + r + 1)), int(min(h, pred[1] + r + 1))
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            break  # fully left the frame
        win = snr[y0:y1, x0:x1]
        _, peak, _, loc = cv2.minMaxLoc(win)
        if peak >= snr_accept:
            cx, cy = x0 + loc[0], y0 + loc[1]
            cx, cy, bw, bh = refine_box(snr, cx, cy, peak)
            step = np.array([cx, cy]) - pos
            if coast == 0:
                vel = 0.7 * vel + 0.3 * (step if not reverse else -step)
            else:
                vel = (step if not reverse else -step) / max(1, coast + 1)
            pos = np.array([cx, cy])
            out[t] = (cx, cy, bw, bh, "tracked")
            coast = 0
        else:
            coast += 1
            pos = pred
            vel *= 0.9  # damp during coast so the prediction doesn't fly away
            out[t] = (float(pos[0]), float(pos[1]), 5.0, 5.0, "coast")
            if coast > max_coast:
                break
        if t % 100 == 0:
            print(f"  [{name}] frame {t}: pos=({pos[0]:.0f},{pos[1]:.0f}) "
                  f"peak={peak:.1f} coast={coast}", flush=True)
    return out


def main():
    print("loading + stabilizing...", flush=True)
    grays, shifts = load_stabilized()
    n = len(grays)
    print(f"{n} frames", flush=True)

    def trim(traj):
        items = sorted(traj.items())
        while items and items[-1][1][4] == "coast":
            items.pop()
        while items and items[0][1][4] == "coast":
            items.pop(0)
        return dict(items)

    # The far drone's flight has three phases; the middle one (a fast
    # right-and-up dash across low-contrast haze/trees, ~frames 291..335)
    # could not be tracked reliably and is EXCLUDED from evaluation.
    A_END = 290       # descent + low leftward skim: reliable up to here
    B_START = 336     # slow leftward cruise after the U-turn near the right edge

    print("tracking far drone segment A from (1005,117)@0 ...", flush=True)
    far_a = trim(track(grays, (1005.0, 117.0), 0, v0=(-1.0, 4.0), name="farA"))
    far_a = {t: v for t, v in far_a.items() if t <= A_END}

    print("tracking far drone segment B (return leg) from (1050,356)@400 ...", flush=True)
    b_fwd = track(grays, (1050.0, 356.0), 400, v0=(-2.4, -0.3), name="farB-fwd")
    b_bwd = track(grays, (1050.0, 356.0), 400, v0=(-2.4, -0.3), reverse=True,
                  name="farB-bwd", max_coast=25, snr_accept=4.5)
    far_b = trim({**b_bwd, **b_fwd})
    far_b = {t: v for t, v in far_b.items() if t >= B_START}

    far = {**far_a, **far_b}

    gt = GroundTruth(video=VIDEO)
    gt.meta["shifts"] = {str(i): [round(s[0], 3), round(s[1], 3)]
                         for i, s in enumerate(shifts)}
    gt.meta["exclude_frames"] = list(range(A_END + 1, B_START))
    gt.meta["notes"] = (
        "far drone: descent+low skim (0..290) and post-U-turn leftward cruise "
        "(336..570) tracked reliably; the fast dash between them is excluded "
        "from evaluation (uncertain GT)."
    )

    def to_obj(name, traj, ignore=False):
        obj = GTObject(name=name, ignore=ignore)
        statuses = {}
        for t, (cx, cy, bw, bh, status) in sorted(traj.items()):
            dx, dy = shifts[t]
            obj.frames[t] = (cx - dx, cy - dy, bw, bh)  # back to original coords
            statuses[str(t)] = status
        gt.meta.setdefault("status", {})[name] = statuses
        return obj

    gt.objects["far"] = to_obj("far", far)

    # near drone: static, landed. Box measured on frame 0 and mapped
    # per-frame by the (tiny) camera shift.
    near = GTObject(name="near", ignore=False)
    ncx, ncy, nw, nh = 683.0, 664.0, 180.0, 105.0
    for t in range(n):
        dx, dy = shifts[t]
        near.frames[t] = (ncx - dx, ncy - dy, nw, nh)
    gt.objects["near"] = near

    gt.save("work/gt.json")
    fartr = [t for t, v in far.items() if v[4] == "tracked"]
    print(f"far: {len(far)} frames ({len(fartr)} tracked), "
          f"span {min(far)}..{max(far)}")
    coasts = [t for t, v in sorted(far.items()) if v[4] == "coast"]
    print("coast frames:", coasts[:40], "..." if len(coasts) > 40 else "")
    print("saved work/gt.json")


if __name__ == "__main__":
    main()
