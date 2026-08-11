#!/usr/bin/env python3
"""Build a 4-channel RGB+motion fusion dataset (YOLOMG idea done for a *colour*
generalist).

Our shipped temporal expert stacks 3 ego-aligned GRAYS -> it throws colour away,
which is fine for the black drone but wastes the appearance signal that carries
the white/near drones. YOLOMG instead fuses an explicit ego-motion-compensated
motion-difference MAP with the RGB frame *inside one network*, and it is the
actual SOTA-tying idea on the hardest tiny set (ARD100 0.33->0.78). This builder
produces that input: each tile is a 4-channel array [B, G, R, motion(t)] where

    motion(t) = min(|gray_t - warp(gray_{t-dt})|, |gray_t - warp(gray_{t-2dt})|)

(the same 3-frame compensated differencing as mc_hybrid/_MCMotion, one channel).
A moving drone lights up in channel 4 regardless of colour; registered static
background cancels to ~0. The RGB channels keep the appearance the motion map
lacks. Stored as .npy (ultralytics loads >3-channel images only from .npy) with
`channels: 4` in data.yaml, so the ch=4 model sees both modalities and learns the
fusion end-to-end -- replacing our late NMS union of two separate experts.

Same per-dataset whole-video splits / labels as the combined + temporal datasets
(reuses make_dataset_external), so results stay comparable.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_dataset_external as M
from make_temporal_combined import register   # grid-LK + RANSAC homography

OUT = M.OUT_ROOT / "combined_fusion"
DT = 3


def motion_map(color_buf, idx, dt):
    """Single-channel ego-compensated motion-difference at frame idx (uint8)."""
    g = cv2.cvtColor(color_buf[idx], cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    diffs = []
    for k in (dt, 2 * dt):
        gp = color_buf.get(idx - k)
        if gp is None:
            continue
        gp = cv2.cvtColor(gp, cv2.COLOR_BGR2GRAY)
        H = register(gp, g)
        warp = (cv2.warpPerspective(gp, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
                if H is not None else gp)
        diffs.append(cv2.absdiff(g, warp))
    if not diffs:
        return np.zeros((h, w), np.uint8)
    m = diffs[0] if len(diffs) == 1 else np.minimum(diffs[0], diffs[1])
    return m


def build_fusion(color_buf, idx, dt):
    """4-channel [B, G, R, motion] uint8 array aligned to frame idx."""
    bgr = color_buf[idx]
    mot = motion_map(color_buf, idx, dt)
    return np.dstack([bgr, mot])   # H,W,4  (B,G,R,motion)


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
        buf[idx] = fr                                   # keep COLOR (BGR)
        buf.pop(idx - 2 * dt - 2, None)
        if idx in want and idx >= 2 * dt:
            stack = build_fusion(buf, idx, dt)
            H, W = stack.shape[:2]
            for (x1, y1, x2, y2) in boxes.get(idx, []):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                jx = rng.uniform(-jitter, jitter) * tile
                jy = rng.uniform(-jitter, jitter) * tile
                x0 = int(min(max(cx + jx - tile / 2, 0), max(W - tile, 0)))
                y0 = int(min(max(cy + jy - tile / 2, 0), max(H - tile, 0)))
                crop = stack[y0:y0 + tile, x0:x0 + tile]
                if crop.shape[:2] != (tile, tile):
                    pad = np.zeros((tile, tile, 4), np.uint8)
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
                np.save(str(OUT / "images" / split / f"{stem}.npy"), crop)
                (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
                n_img += 1
        idx += 1
    cap.release()
    return n_img, n_box


def write_data_yaml(out):
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "channels: 4\n"
        "names:\n  0: drone\n")


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
                print(f"  [{split}] {ds}/{vid}: {ni} fusion tiles", flush=True)
    for split, fr, stride in (("train", (0, M.USER_SPLIT_AT), a.stride_train),
                              ("val", (M.USER_SPLIT_AT, 10 ** 9), a.stride_val)):
        boxes = M.parse_repo_gt("work/gt_user.json", frange=fr)
        pos = [f for f, b in boxes.items() if b]
        ni, nb = extract("data/videos/07_05.mp4", boxes, set(pos[::stride]), split, "user__07_05")
        stats[split][0] += ni
        stats[split][1] += nb
    write_data_yaml(OUT)
    print(f"\nCOMBINED FUSION (4ch RGB+motion) -> {OUT}")
    print(f"  train {stats['train'][0]} tiles / {stats['train'][1]} boxes | "
          f"val {stats['val'][0]} / {stats['val'][1]}")


if __name__ == "__main__":
    main()
