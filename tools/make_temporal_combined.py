#!/usr/bin/env python3
"""Build a TEMPORAL-stack version of the combined dataset (Dogfight/YOLOMG idea:
motion encoded in the input). Each sample is a 3-channel image =
[ego-aligned gray(t-2dt), ego-aligned gray(t-dt), gray(t)]. A moving drone leaves
a 3-position trail across the channels; static background registers to a uniform
gray. Same per-dataset whole-video splits as the combined dataset.

Reuses the parsers/splits from make_dataset_external so labels/splits stay identical.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_dataset_external as M

OUT = M.OUT_ROOT / "combined_temporal"
DT = 3


def register(src, dst, gx=30, gy=17):
    """Homography mapping src gray onto dst gray (grid LK + RANSAC); None if weak."""
    h, w = dst.shape
    xs = np.linspace(w * 0.05, w * 0.95, gx)
    ys = np.linspace(h * 0.05, h * 0.95, gy)
    pts = np.array([[x, y] for y in ys for x in xs], np.float32).reshape(-1, 1, 2)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(src, dst, pts, None, winSize=(21, 21), maxLevel=3,
                                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    good = st.ravel() == 1
    if good.sum() < 12:
        return None
    H, inl = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 3.0)
    return H


def build_stack(buf, idx, dt):
    """3-channel stack aligned to frame idx from a {frame:gray} buffer."""
    g = buf[idx]
    h, w = g.shape
    chans = []
    for k in (2 * dt, dt, 0):
        gp = buf.get(idx - k)
        if gp is None or k == 0:
            chans.append(g)
            continue
        H = register(gp, g)
        chans.append(cv2.warpPerspective(gp, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
                     if H is not None else gp)
    return cv2.merge(chans)   # [t-2dt, t-dt, t]


def extract(video, boxes, frame_ids, split, prefix, tile=640, dt=DT, min_side=12, jitter=0.35):
    import random
    rng = random.Random(99)
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    want = set(frame_ids)
    cap = cv2.VideoCapture(str(video))
    buf, idx = {}, 0
    n_img = n_box = 0
    last = max(want) if want else -1
    while idx <= last + 1:
        ok, fr = cap.read()
        if not ok:
            break
        buf[idx] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        buf.pop(idx - 2 * dt - 2, None)                 # drop old
        if idx in want and idx >= 2 * dt:
            stack = build_stack(buf, idx, dt)
            H, W = stack.shape[:2]
            for (x1, y1, x2, y2) in boxes.get(idx, []):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                jx = rng.uniform(-jitter, jitter) * tile
                jy = rng.uniform(-jitter, jitter) * tile
                x0 = int(min(max(cx + jx - tile / 2, 0), max(W - tile, 0)))
                y0 = int(min(max(cy + jy - tile / 2, 0), max(H - tile, 0)))
                crop = stack[y0:y0 + tile, x0:x0 + tile]
                if crop.shape[:2] != (tile, tile):
                    pad = np.zeros((tile, tile, 3), np.uint8)
                    pad[:crop.shape[0], :crop.shape[1]] = crop
                    crop = pad
                lines = []
                for (bx1, by1, bx2, by2) in boxes.get(idx, []):
                    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    if not (x0 <= bcx <= x0 + tile and y0 <= bcy <= y0 + tile):
                        continue
                    lcx, lcy = bcx - x0, bcy - y0
                    bw, bh = max(bx2 - bx1, min_side), max(by2 - by1, min_side)
                    lines.append(f"0 {lcx/tile:.6f} {lcy/tile:.6f} {bw/tile:.6f} {bh/tile:.6f}")
                    n_box += 1
                stem = f"{prefix}_{idx:05d}"
                cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
                n_img += 1
        idx += 1
    cap.release()
    return n_img, n_box


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride-train", type=int, default=6)
    ap.add_argument("--stride-val", type=int, default=12)
    a = ap.parse_args()
    sp = M.combined_splits()
    stats = {"train": [0, 0], "val": [0, 0]}
    for ds in ("ardmav", "nps"):
        for split, stride in (("train", a.stride_train), ("val", a.stride_val)):
            for vid in sp[ds][split]:
                vpath, boxes = M._sources_for(ds, vid)
                if vpath is None or not Path(vpath).exists():
                    continue
                pos = [f for f, b in boxes.items() if b]
                ni, nb = extract(vpath, boxes, set(pos[::stride]), split, f"{ds}__{vid}")
                stats[split][0] += ni
                stats[split][1] += nb
                print(f"  [{split}] {ds}/{vid}: {ni} temporal tiles", flush=True)
    for split, fr, stride in (("train", (0, M.USER_SPLIT_AT), a.stride_train),
                              ("val", (M.USER_SPLIT_AT, 10 ** 9), a.stride_val)):
        boxes = M.parse_repo_gt("work/gt_user.json", frange=fr)
        pos = [f for f, b in boxes.items() if b]
        ni, nb = extract("data/videos/07_05.mp4", boxes, set(pos[::stride]), split, "user__07_05")
        stats[split][0] += ni
        stats[split][1] += nb
    M.write_data_yaml(OUT)
    print(f"\nCOMBINED TEMPORAL -> {OUT}")
    print(f"  train {stats['train'][0]} tiles / {stats['train'][1]} boxes | "
          f"val {stats['val'][0]} / {stats['val'][1]}")


if __name__ == "__main__":
    main()
