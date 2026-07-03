"""Harden the 10_06 test reference (v2).

The v1 reference (build_gt_1006.py) is the PC pipeline's verified track,
measured frames only: 250 GT frames, 111 excluded -- misses inside the
excluded frames are invisible to scoring, and the reference is circular
for the PC pipeline. v2 makes it dense and independent:

  1. candidate position for EVERY frame in the flight span: measured
     position where the track measured, linear interpolation through the
     coast gaps (plus a short extrapolated apron at both ends);
  2. position refinement with a NON-CAUSAL windowed-median diff (offline
     is legal for ground truth): snap to the local motion peak within
     10 px of the candidate when the peak is strong and unique;
  3. verification sheets (raw zoom + diff zoom side by side, position
     circled) for a human pass: every frame gets looked at, frames where
     nothing is visible go to exclude_frames.

Run: .venv/bin/python realtime/tools/harden_gt_1006.py          # sheets
     ... --finalize bad1,bad2,...                               # write GT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from dronedet.gt import GroundTruth, GTObject
from dronedet.stabilize import Stabilizer, shift_of, warp_to_reference
from dronedet.video import frames

VIDEO = "data/videos/10_06.mp4"
OUT_DIR = Path("realtime/work/gt1006_verify")
APRON = 20          # extrapolated frames past both track ends
SEARCH_R = 10.0     # refine: search radius around candidate
SNR_MIN = 3.5
UNIQUE = 0.75
BG_OFFS = (-24, -16, -8, 8, 16, 24)   # non-causal background samples


def load_frames():
    stab = Stabilizer("translation")
    grays, shifts, colors = [], [], []
    for idx, frame in frames(VIDEO):
        m = stab.update(frame)
        grays.append(warp_to_reference(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), m))
        shifts.append(shift_of(m))
        colors.append(frame)
    return np.stack(grays).astype(np.float32), shifts, colors


def candidates(n, shifts):
    """Per-frame candidate position in STABILIZED coords + source tag."""
    pc = json.loads(Path("work/infer/10_06/tracks_all.json").read_text())
    id1 = {int(f): v for tr in pc["tracks"] if tr["id"] == 1
           for f, v in tr["frames"].items()}
    meas = {t: (v[0] + shifts[t][0], v[1] + shifts[t][1])
            for t, v in id1.items() if v[4] != "coast"}
    ks = sorted(meas)
    out = {}
    for t in range(max(0, ks[0] - APRON), min(n, ks[-1] + APRON + 1)):
        if t in meas:
            out[t] = (*meas[t], "meas")
            continue
        lo = max((k for k in ks if k < t), default=None)
        hi = min((k for k in ks if k > t), default=None)
        if lo is not None and hi is not None:
            a = (t - lo) / (hi - lo)
            out[t] = (meas[lo][0] * (1 - a) + meas[hi][0] * a,
                      meas[lo][1] * (1 - a) + meas[hi][1] * a, "interp")
        elif lo is not None:                      # forward apron
            out[t] = (*meas[lo], "extrap")
        elif hi is not None:                      # backward apron
            out[t] = (*meas[hi], "extrap")
    return out


def refine(grays, t, cx, cy, search_r=SEARCH_R, snr_min=SNR_MIN):
    n, h, w = grays.shape
    half = 22
    x0 = int(np.clip(cx - half, 0, w - 2 * half))
    y0 = int(np.clip(cy - half, 0, h - 2 * half))
    idxs = [t + o for o in BG_OFFS if 0 <= t + o < n]
    if len(idxs) < 3:      # video ends: 3 one-sided samples still work
        return None
    stack = grays[idxs, y0:y0 + 2 * half, x0:x0 + 2 * half]
    bg = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - bg), axis=0)
    sigma = np.maximum(1.4826 * mad, 1.5)
    snr = np.abs(grays[t, y0:y0 + 2 * half, x0:x0 + 2 * half] - bg) / sigma
    # only look near the candidate
    yy, xx = np.mgrid[0:2 * half, 0:2 * half]
    mask = (xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2 <= search_r ** 2
    snr_m = np.where(mask, snr, 0)
    _, peak, _, loc = cv2.minMaxLoc(snr_m)
    if peak < snr_min:
        return None
    masked = snr_m.copy()
    masked[max(0, loc[1] - 4):loc[1] + 5, max(0, loc[0] - 4):loc[0] + 5] = 0
    _, peak2, _, _ = cv2.minMaxLoc(masked)
    if peak2 >= UNIQUE * peak:
        return None
    # centroid around the peak for sub-pixel position
    py0, py1 = max(0, loc[1] - 2), min(2 * half, loc[1] + 3)
    px0, px1 = max(0, loc[0] - 2), min(2 * half, loc[0] + 3)
    win = snr[py0:py1, px0:px1]
    ys, xs = np.mgrid[py0:py1, px0:px1]
    m = win.sum()
    return (float(x0 + (xs * win).sum() / m), float(y0 + (ys * win).sum() / m),
            float(peak), snr, (x0, y0))


def sheets(grays, colors, shifts, cand, refined):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = []
    Z, HALF = 3, 26
    for t in sorted(cand):
        cx, cy, src = cand[t]
        r = refined.get(t)
        if r is not None:
            cx, cy = r[0], r[1]
        # raw crop (original frame coords)
        ox, oy = cx - shifts[t][0], cy - shifts[t][1]
        img = colors[t]
        x0 = int(np.clip(ox - HALF, 0, img.shape[1] - 2 * HALF))
        y0 = int(np.clip(oy - HALF, 0, img.shape[0] - 2 * HALF))
        raw = cv2.resize(img[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF], None,
                         fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
        cv2.circle(raw, (int((ox - x0) * Z), int((oy - y0) * Z)), 12,
                   (0, 0, 255), 1)
        # diff crop (stabilized coords)
        if r is not None:
            snr = r[3]
            d8 = np.clip(snr * 32, 0, 255).astype(np.uint8)
            heat = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
            hx0, hy0 = r[4]
            sc = raw.shape[1] / heat.shape[1]
            heat = cv2.resize(heat, (raw.shape[1], raw.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(heat, (int((cx - hx0) * sc), int((cy - hy0) * sc)),
                           (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
        else:
            g = grays[t]
            gx0 = int(np.clip(cx - HALF, 0, g.shape[1] - 2 * HALF))
            gy0 = int(np.clip(cy - HALF, 0, g.shape[0] - 2 * HALF))
            prev = grays[max(0, t - 6)]
            d = np.abs(g[gy0:gy0 + 2 * HALF, gx0:gx0 + 2 * HALF]
                       - prev[gy0:gy0 + 2 * HALF, gx0:gx0 + 2 * HALF])
            d8 = np.clip(d * 8, 0, 255).astype(np.uint8)
            heat = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
            heat = cv2.resize(heat, (raw.shape[1], raw.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        tag = f"{t} {src[0]}{'R' if r else ''}"
        cell = np.hstack([raw, heat])
        cv2.putText(cell, tag, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)
    per_row, per_sheet = 4, 40
    for s in range(0, len(cells), per_sheet):
        chunk = cells[s:s + per_sheet]
        rows = [np.hstack(chunk[i:i + per_row]) for i in range(0, len(chunk), per_row)
                if len(chunk[i:i + per_row]) == per_row]
        rest = len(chunk) % per_row
        if rest:
            pad = [np.zeros_like(chunk[0])] * (per_row - rest)
            rows.append(np.hstack(chunk[-rest:] + pad))
        cv2.imwrite(str(OUT_DIR / f"sheet_{s // per_sheet:02d}.png"), np.vstack(rows))
    print(f"{(len(cells) + per_sheet - 1) // per_sheet} sheets -> {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize", default=None,
                    help="comma-separated frames verified NOT visible -> excluded")
    a = ap.parse_args()

    grays, shifts, colors = load_frames()
    n = len(grays)
    print(f"{n} frames")
    cand = candidates(n, shifts)
    refined = {}
    for t, (cx, cy, src) in sorted(cand.items()):
        r = refine(grays, t, cx, cy)
        if r is not None:
            refined[t] = r
    n_r = len(refined)
    print(f"candidates {len(cand)} (span {min(cand)}..{max(cand)}), "
          f"refined pass1 {n_r} ({n_r / len(cand):.0%})")

    # pass 2+: unrefined frames re-anchor on their refined neighbours
    # (linear position propagation), wider search, stricter SNR. Fixes the
    # end-of-track frames where the PC-measured candidate lags the true
    # position by >10 px, and the apron frames past the last measurement.
    for _sweep in range(4):
        changed = 0
        for t in sorted(cand):
            if t in refined:
                continue
            ks = sorted(refined)
            lo = max((k for k in ks if k < t), default=None)
            hi = min((k for k in ks if k > t), default=None)
            if lo is not None and hi is not None and hi - lo <= 12:
                al = (t - lo) / (hi - lo)
                cx = refined[lo][0] * (1 - al) + refined[hi][0] * al
                cy = refined[lo][1] * (1 - al) + refined[hi][1] * al
            elif lo is not None and t - lo <= 6 and lo - 3 in refined:
                vx = (refined[lo][0] - refined[lo - 3][0]) / 3
                vy = (refined[lo][1] - refined[lo - 3][1]) / 3
                cx = refined[lo][0] + vx * (t - lo)
                cy = refined[lo][1] + vy * (t - lo)
            elif hi is not None and hi - t <= 6 and hi + 3 in refined:
                vx = (refined[hi + 3][0] - refined[hi][0]) / 3
                vy = (refined[hi + 3][1] - refined[hi][1]) / 3
                cx = refined[hi][0] + vx * (t - hi)
                cy = refined[hi][1] + vy * (t - hi)
            else:
                continue
            r = refine(grays, t, cx, cy, search_r=14.0, snr_min=4.0)
            if r is not None:
                refined[t] = r
                cand[t] = (cx, cy, cand[t][2] + "+")
                changed += 1
        if not changed:
            break
    n_r = len(refined)
    print(f"refined after propagation {n_r} ({n_r / len(cand):.0%}); "
          f"still unrefined: {sorted(set(cand) - set(refined))}")

    if a.finalize is None:
        sheets(grays, colors, shifts, cand, refined)
        payload = {str(t): {"pos": [round(refined[t][0] if t in refined else cand[t][0], 2),
                                    round(refined[t][1] if t in refined else cand[t][1], 2)],
                            "src": cand[t][2], "refined": t in refined,
                            "snr": round(refined[t][2], 1) if t in refined else None}
                   for t in sorted(cand)}
        (OUT_DIR / "candidates.json").write_text(json.dumps(payload))
        return

    bad = set(int(x) for x in a.finalize.split(",") if x.strip()) if a.finalize else set()
    cjson = json.loads((OUT_DIR / "candidates.json").read_text())
    gt_old = GroundTruth.load("realtime/work/gt_1006.json")
    gt = GroundTruth(video=VIDEO)
    gt.meta["shifts"] = {str(i): [round(s[0], 3), round(s[1], 3)]
                         for i, s in enumerate(shifts)}
    gt.meta["note"] = ("TEST reference v2: dense flight-span coverage; every "
                       "frame human-verified on raw+diff sheets; unverifiable "
                       "frames excluded; unconfirmed movers = ignore")
    obj = GTObject("far", ignore=False)
    for t_str, c in cjson.items():
        t = int(t_str)
        if t in bad:
            continue
        cx, cy = c["pos"]
        obj.frames[t] = (cx - shifts[t][0], cy - shifts[t][1], 8.0, 8.0)
    gt.objects["far"] = obj
    for name, o in gt_old.objects.items():
        if name != "far":
            gt.objects[name] = o
    excl = sorted(set(range(n)) - set(obj.frames))
    gt.meta["exclude_frames"] = excl
    gt.save("realtime/work/gt_1006_v2.json")
    print(f"gt_1006_v2: {len(obj.frames)} far frames, {len(excl)} excluded "
          f"(was {len(gt_old.objects['far'].frames)} / "
          f"{len(gt_old.meta.get('exclude_frames', []))})")


if __name__ == "__main__":
    main()
