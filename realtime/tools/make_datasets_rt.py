"""Build the three realtime training sets from 07_05 + gt_user.json.

    rt_crop256        temporal 256px slices  -> crop verifier (yolov8n-p2 @256)
    rt_full_temporal  full-frame stabilized temporal images -> RT-C/RT-D
    rt_full_single    full-frame raw RGB, same labels        -> RT-F reference

All: train = frames < 342, val = frames >= 342 (internal), 2 classes
(drone / bird), 24 px inflated labels, near drone erased (it is handled by
the low-rate expert at inference). data/videos/10_06.mp4 is never touched.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from dronedet.gt import GroundTruth
from dronedet.stabilize import Stabilizer, warp_to_reference
from dronedet.video import frames

VIDEO = "data/videos/07_05.mp4"
SPLIT_AT = 342
DT = 6
LABEL = 24.0
CLS = {"drone": 0, "bird": 1}
rng = random.Random(20260707)


def erase_near_gray(g, near_box):
    cx, cy, w, h = near_box
    pad = 16
    x1, y1 = max(0, int(cx - w / 2 - pad)), max(0, int(cy - h / 2 - pad))
    x2, y2 = min(g.shape[1], int(cx + w / 2 + pad)), min(g.shape[0], int(cy + h / 2 + pad))
    bw, bh = x2 - x1, y2 - y1
    sx = max(0, x1 - 265)
    src = g[y1:y2, sx:sx + bw]
    if src.shape[1] != bw:
        return g
    out = g.copy()
    rx = np.minimum(np.arange(bw), np.arange(bw)[::-1]).astype(np.float32)
    ry = np.minimum(np.arange(bh), np.arange(bh)[::-1]).astype(np.float32)
    a = np.minimum(np.minimum(rx[None, :], ry[:, None]) / 12.0, 1.0)
    if src.ndim == 3:
        a = a[..., None]
    out[y1:y2, x1:x2] = (a * src.astype(np.float32) +
                         (1 - a) * out[y1:y2, x1:x2].astype(np.float32)).astype(np.uint8)
    return out


def feather_paste_gray(dst, patch, cx, cy, haze=0.0):
    ph, pw = patch.shape[:2]
    x0, y0 = int(cx) - pw // 2, int(cy) - ph // 2
    if x0 < 0 or y0 < 0 or x0 + pw > dst.shape[1] or y0 + ph > dst.shape[0]:
        return
    roi = dst[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
    p = patch.astype(np.float32)
    if haze > 0.01:
        p = (1 - haze) * p + haze * float(roi.mean())
    yy, xx = np.mgrid[0:ph, 0:pw]
    r = np.hypot((xx - pw / 2) / (pw / 2), (yy - ph / 2) / (ph / 2))
    a = np.clip(1.6 - 1.6 * r, 0, 1)
    dst[y0:y0 + ph, x0:x0 + pw] = (a * p + (1 - a) * roi).astype(np.uint8)


def cut_patch(img, box, margin=3):
    cx, cy, w, h = box
    r = int(max(w, h) / 2 + margin)
    x0, y0 = int(cx - r), int(cy - r)
    if x0 < 0 or y0 < 0 or x0 + 2 * r > img.shape[1] or y0 + 2 * r > img.shape[0]:
        return None
    return img[y0:y0 + 2 * r, x0:x0 + 2 * r].copy()


def load_stabilized():
    gt = GroundTruth.load("work/gt_user.json")
    near = GroundTruth.load("work/gt.json").objects["near"]
    stab = Stabilizer("translation")
    grays, shifts = [], []
    for idx, frame in frames(VIDEO):
        m = stab.update(frame)
        g = warp_to_reference(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), m)
        nb = near.box(idx)
        if nb is not None:
            g = erase_near_gray(g, (nb[0] + m[0, 2], nb[1] + m[1, 2], nb[2], nb[3]))
        grays.append(g)
        shifts.append((m[0, 2], m[1, 2]))
    return gt, grays, shifts


def build_banks(gt, grays, shifts):
    banks = {"drone": [], "bird": []}
    far = gt.objects["far"]
    for idx in range(2 * DT, 141, 2):
        b = far.box(idx)
        if b is not None:
            p = cut_patch(grays[idx], (b[0] + shifts[idx][0], b[1] + shifts[idx][1], b[2], b[3]))
            if p is not None:
                banks["drone"].append(p)
    for name, o in gt.objects.items():
        if not name.startswith("bird"):
            continue
        for idx in sorted(o.frames)[::3]:
            if idx < 2 * DT:
                continue
            b = o.frames[idx]
            p = cut_patch(grays[idx], (b[0] + shifts[idx][0], b[1] + shifts[idx][1], b[2], b[3]))
            if p is not None and min(p.shape[:2]) >= 8:
                banks["bird"].append(p)
    return banks


def objs_at(gt, shifts, t):
    out = []
    fb = gt.objects["far"].box(t)
    if fb is not None:
        out.append(("drone", (fb[0] + shifts[t][0], fb[1] + shifts[t][1])))
    for name, o in gt.objects.items():
        if name.startswith("bird"):
            b = o.box(t)
            if b is not None:
                out.append(("bird", (b[0] + shifts[t][0], b[1] + shifts[t][1])))
    return out


def write_yolo(root, split, name, img, lines):
    cv2.imwrite(str(root / f"images/{split}/{name}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    (root / f"labels/{split}/{name}.txt").write_text("\n".join(lines) + "\n")


def make_dirs(root):
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: drone\n  1: bird\n")


def build_crop256(gt, grays, shifts, banks):
    TILE = 256
    root = Path("realtime/work/dataset_crop256")
    make_dirs(root)
    counts = {"train": 0, "val": 0}
    excl = set(gt.meta.get("exclude_frames", []))
    h, w = grays[0].shape
    for t in range(2 * DT, len(grays)):
        if t in excl:
            continue
        split = "train" if t < SPLIT_AT else "val"
        chans = [grays[t - 2 * DT], grays[t - DT], grays[t]]
        objs = objs_at(gt, shifts, t)
        slices = []
        for cls_name, (cx, cy) in objs[:2]:  # far slice + one bird slice
            margin = 40
            x0 = int(np.clip(cx - rng.uniform(margin, TILE - margin), 0, w - TILE))
            y0 = int(np.clip(cy - rng.uniform(margin, TILE - margin), 0, h - TILE))
            slices.append((cls_name, x0, y0, False))
        if split == "train":
            for _try in range(30):
                x0 = rng.randint(0, w - TILE)
                y0 = rng.randint(0, h - TILE)
                if all(not (x0 - 20 < ox < x0 + TILE + 20 and
                            y0 - 20 < oy < y0 + TILE + 20) for _, (ox, oy) in objs):
                    slices.append(("paste", x0, y0, True))
                    break
        for tag, x0, y0, is_paste in slices:
            crop_ch = [c[y0:y0 + TILE, x0:x0 + TILE].copy() for c in chans]
            lines, taken = [], []
            if not is_paste:
                for cls_name, (cx, cy) in objs:
                    if x0 <= cx < x0 + TILE and y0 <= cy < y0 + TILE:
                        lines.append(f"{CLS[cls_name]} {(cx-x0)/TILE:.6f} "
                                     f"{(cy-y0)/TILE:.6f} {LABEL/TILE:.6f} {LABEL/TILE:.6f}")
                        taken.append((cx - x0, cy - y0))
            else:
                for cls_name, k in (("drone", 2), ("bird", 2)):
                    bank = banks[cls_name]
                    for _ in range(k):
                        p0 = bank[rng.randrange(len(bank))].copy()
                        if rng.random() < 0.5:
                            p0 = p0[:, ::-1]
                        s = rng.uniform(0.4, 1.2)
                        p0 = cv2.resize(p0, None, fx=s, fy=s,
                                        interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
                        if min(p0.shape[:2]) < 4:
                            continue
                        haze = rng.uniform(0.0, 0.5)
                        speed = rng.uniform(0.0, 6.0) if cls_name == "drone" else rng.uniform(1.0, 8.0)
                        ang = rng.uniform(0, 2 * np.pi)
                        vx, vy = speed * np.cos(ang), speed * np.sin(ang)
                        cx = rng.randint(40, TILE - 40)
                        cy = rng.randint(40, TILE - 40)
                        if any((cx-tx)**2 + (cy-ty)**2 <= 40**2 for tx, ty in taken):
                            continue
                        for k_ch, ch in enumerate(crop_ch):
                            back = (2 - k_ch) * DT
                            p = p0
                            if cls_name == "bird":
                                js = rng.uniform(0.75, 1.3)
                                p = cv2.resize(p0, None, fx=js, fy=js,
                                               interpolation=cv2.INTER_LINEAR)
                            feather_paste_gray(ch, p, cx - vx * back, cy - vy * back, haze)
                        taken.append((cx, cy))
                        lines.append(f"{CLS[cls_name]} {cx/TILE:.6f} {cy/TILE:.6f} "
                                     f"{LABEL/TILE:.6f} {LABEL/TILE:.6f}")
            write_yolo(root, split, f"f{t:05d}_{tag}", np.dstack(crop_ch), lines)
            counts[split] += 1
    print("rt_crop256:", counts)


def build_full(gt, grays, shifts, temporal: bool):
    root = Path(f"realtime/work/dataset_full_{'temporal' if temporal else 'single'}")
    make_dirs(root)
    counts = {"train": 0, "val": 0}
    excl = set(gt.meta.get("exclude_frames", []))
    cache_color = {}
    if not temporal:
        near = GroundTruth.load("work/gt.json").objects["near"]
        for idx, frame in frames(VIDEO):
            nb = near.box(idx)
            if nb is not None:
                frame = erase_near_gray(frame, nb)  # works channel-wise via broadcast
            cache_color[idx] = frame
    h, w = grays[0].shape
    banks = build_banks(gt, grays, shifts) if temporal else None
    for t in range(2 * DT, len(grays)):
        if t in excl:
            continue
        split = "train" if t < SPLIT_AT else "val"
        lines = []
        objs = objs_at(gt, shifts, t)
        if temporal:
            chans = [grays[t - 2 * DT].copy(), grays[t - DT].copy(), grays[t].copy()]
            taken = [(cx, cy) for _, (cx, cy) in objs]
            if split == "train":  # paste extra instances away from real GT
                for cls_name, k in (("drone", 4), ("bird", 3)):
                    bank = banks[cls_name]
                    for _ in range(k):
                        p0 = bank[rng.randrange(len(bank))].copy()
                        if rng.random() < 0.5:
                            p0 = p0[:, ::-1]
                        sc = rng.uniform(0.4, 1.2)
                        p0 = cv2.resize(p0, None, fx=sc, fy=sc,
                                        interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
                        if min(p0.shape[:2]) < 4:
                            continue
                        haze = rng.uniform(0.0, 0.5)
                        speed = rng.uniform(0.0, 6.0) if cls_name == "drone" else rng.uniform(1.0, 8.0)
                        ang = rng.uniform(0, 2 * np.pi)
                        vx, vy = speed * np.cos(ang), speed * np.sin(ang)
                        cx = rng.randint(60, w - 60)
                        cy = rng.randint(60, h - 60)
                        if any((cx - tx) ** 2 + (cy - ty) ** 2 <= 60 ** 2 for tx, ty in taken):
                            continue
                        for k_ch, ch in enumerate(chans):
                            back = (2 - k_ch) * DT
                            pp = p0
                            if cls_name == "bird":
                                js = rng.uniform(0.75, 1.3)
                                pp = cv2.resize(p0, None, fx=js, fy=js,
                                                interpolation=cv2.INTER_LINEAR)
                            feather_paste_gray(ch, pp, cx - vx * back, cy - vy * back, haze)
                        taken.append((cx, cy))
                        lines.append(f"{CLS[cls_name]} {cx/w:.6f} {cy/h:.6f} "
                                     f"{LABEL/w:.6f} {LABEL/h:.6f}")
            img = np.dstack(chans)
        else:
            img = cache_color[t]
        for cls_name, (cx, cy) in objs:
            if not temporal:  # single-frame labels in ORIGINAL coords
                cx, cy = cx - shifts[t][0], cy - shifts[t][1]
            lines.append(f"{CLS[cls_name]} {cx/w:.6f} {cy/h:.6f} "
                         f"{LABEL/w:.6f} {LABEL/h:.6f}")
        write_yolo(root, split, f"f{t:05d}", img, lines)
        counts[split] += 1
    print(f"rt_full_{'temporal' if temporal else 'single'}:", counts)


def main():
    gt, grays, shifts = load_stabilized()
    banks = build_banks(gt, grays, shifts)
    print(f"banks: drone {len(banks['drone'])}, bird {len(banks['bird'])}")
    build_crop256(gt, grays, shifts, banks)
    build_full(gt, grays, shifts, temporal=True)
    build_full(gt, grays, shifts, temporal=False)


if __name__ == "__main__":
    main()
