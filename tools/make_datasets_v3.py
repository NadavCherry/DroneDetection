"""Dataset v3 -- the round-3 training sets from 07_05 + gt_user.json.

Supersedes make_dataset_ft7.py (PC verifier) and the dataset half of
realtime/tools/make_datasets_rt.py (edge nets). One stabilization pass,
three outputs:

    work/dsv3_full_temporal   full-frame 3-ch stabilized stacks @1280 (RT-C/D + PC full pass)
    work/dsv3_crop640         640px temporal slices (PC verifier, ft7-v3)
    work/dsv3_crop256         256px temporal slices (edge verifier)

v3 changes over v2 (each targets a measured failure):
  * drone patch bank from ALL train frames, not just the sky phase 24..140
    -- adds the dim low-contrast bush-phase appearance that dominates val
       and the 10_06 foliage-crossing misses
  * sub-pixel trail placement (fractional per-channel offsets) -- the real
    val target drifts 0.5 px/frame; integer trails quantize that signal
  * patch photometric jitter (brightness/contrast/blur) + haze up to 0.55
  * drone paste speed 60% U(0,2.5) + 40% U(2.5,9) px/frame (was 0..2/0..6)
  * warmup frames t<2*DT with CLAMPED stacks (channels repeat frame 0) so
    the model learns the short-trail/no-trail start-of-stream regime;
    inference clamps identically -> detections from frame 0
  * hard-negative oversample: train frames where the v2 edge net false-
    alarmed get an extra no-paste copy (real labels only)
  * 2 paste variants per train frame for the full-frame set

--split-at 548 rebuilds everything on ALL labeled frames (the final
models; 10_06 stays untouched as the test video).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.gt import GroundTruth
from dronedet.stabilize import Stabilizer, warp_to_reference
from dronedet.video import frames

VIDEO = "data/videos/07_05.mp4"
DT = 6      # overridable via --dt (channel spacing ablation)
LABEL = 24.0
CLS = {"drone": 0, "bird": 1}


def erase_near_gray(g, near_box):
    cx, cy, w, h = near_box
    pad = 16
    x1, y1 = max(0, int(cx - w / 2 - pad)), max(0, int(cy - h / 2 - pad))
    x2 = min(g.shape[1], int(cx + w / 2 + pad))
    y2 = min(g.shape[0], int(cy + h / 2 + pad))
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


def cut_patch(img, box, margin=3):
    cx, cy, w, h = box
    r = int(max(w, h) / 2 + margin)
    x0, y0 = int(cx - r), int(cy - r)
    if x0 < 0 or y0 < 0 or x0 + 2 * r > img.shape[1] or y0 + 2 * r > img.shape[0]:
        return None
    return img[y0:y0 + 2 * r, x0:x0 + 2 * r].copy()


def feather_paste_subpx(dst, patch, cx, cy, haze=0.0):
    """Feathered paste with sub-pixel position: the fractional part of
    (cx, cy) is baked into the patch via warpAffine before the integer
    blit, so slow-drift trails are not quantized to whole pixels."""
    ph, pw = patch.shape[:2]
    fx = cx - math.floor(cx)
    fy = cy - math.floor(cy)
    if fx > 1e-3 or fy > 1e-3:
        m = np.float32([[1, 0, fx], [0, 1, fy]])
        patch = cv2.warpAffine(patch, m, (pw, ph), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    x0, y0 = int(math.floor(cx)) - pw // 2, int(math.floor(cy)) - ph // 2
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


def jitter_patch(rng, p0):
    """Photometric + geometric jitter of a bank patch."""
    if rng.random() < 0.5:
        p0 = p0[:, ::-1]
    s = rng.uniform(0.4, 1.2)
    p0 = cv2.resize(p0, None, fx=s, fy=s,
                    interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if min(p0.shape[:2]) < 4:
        return None
    gain = rng.uniform(0.8, 1.25)
    bias = rng.uniform(-10, 10)
    p0 = np.clip(p0.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if s < 0.65 or rng.random() < 0.15:
        p0 = cv2.GaussianBlur(p0, (3, 3), 0)
    return p0


def draw_speed(rng, cls_name):
    if cls_name == "drone":
        return rng.uniform(0.0, 2.5) if rng.random() < 0.6 else rng.uniform(2.5, 9.0)
    return rng.uniform(1.0, 8.0)


def paste_group(rng, chans, banks, taken, w, h, want, min_gap=60,
                border=60):
    """Paste `want` = [(cls, count), ...] trajectories into the channel
    stack. Returns label lines (image coords == stabilized coords)."""
    lines = []
    for cls_name, k in want:
        bank = banks[cls_name]
        if not bank:
            continue
        for _ in range(k):
            p0 = jitter_patch(rng, bank[rng.randrange(len(bank))].copy())
            if p0 is None:
                continue
            haze = rng.uniform(0.0, 0.55)
            speed = draw_speed(rng, cls_name)
            ang = rng.uniform(0, 2 * np.pi)
            vx, vy = speed * np.cos(ang), speed * np.sin(ang)
            cx = rng.uniform(border, w - border)
            cy = rng.uniform(border, h - border)
            if any((cx - tx) ** 2 + (cy - ty) ** 2 <= min_gap ** 2
                   for tx, ty in taken):
                continue
            for k_ch, ch in enumerate(chans):
                back = (2 - k_ch) * DT
                pp = p0
                if cls_name == "bird":  # wing flap: per-channel shape change
                    js = rng.uniform(0.75, 1.3)
                    pp = cv2.resize(p0, None, fx=js, fy=js,
                                    interpolation=cv2.INTER_LINEAR)
                feather_paste_subpx(ch, pp, cx - vx * back, cy - vy * back, haze)
            taken.append((cx, cy))
            lines.append((cls_name, cx, cy))
    return lines


def make_dirs(root):
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: drone\n  1: bird\n")


def write_yolo(root, split, name, img, lines):
    cv2.imwrite(str(root / f"images/{split}/{name}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    (root / f"labels/{split}/{name}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""))


def clamp_chans(grays, t):
    return [grays[max(0, t - 2 * DT)], grays[max(0, t - DT)], grays[t]]


def load_stabilized():
    gt = GroundTruth.load("work/gt_user.json")
    near = gt.objects["near"]
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


def build_banks(gt, grays, shifts, split_at):
    """v3: drone patches from EVERY train frame (step 2) -- sky descent,
    low skim AND the dim bush-phase drift; bird patches step 3."""
    banks = {"drone": [], "bird": []}
    far = gt.objects["far"]
    for idx in range(0, split_at, 2):
        if idx >= len(grays):
            break
        b = far.box(idx)
        if b is not None:
            p = cut_patch(grays[idx], (b[0] + shifts[idx][0],
                                       b[1] + shifts[idx][1], b[2], b[3]))
            if p is not None:
                banks["drone"].append(p)
    for name, o in gt.objects.items():
        if not name.startswith("bird"):
            continue
        for idx in sorted(o.frames)[::3]:
            if idx >= split_at or idx >= len(grays):
                continue
            b = o.frames[idx]
            p = cut_patch(grays[idx], (b[0] + shifts[idx][0],
                                       b[1] + shifts[idx][1], b[2], b[3]))
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


def hard_negative_frames(gt, split_at,
                         det_path="realtime/work/out/0705/rt-c-full1280/dets.json",
                         min_score=0.30, tau=12.0):
    """Train frames where the v2 edge net produced a false positive."""
    p = Path(det_path)
    if not p.exists():
        return set()
    ds = json.loads(p.read_text())
    out = set()
    for f_str, dets in ds["frames"].items():
        f = int(f_str)
        if f >= split_at:
            continue
        gts = []
        for name, obj in gt.objects.items():
            b = obj.box(f)
            if b is not None:
                r = max(tau, 0.5 * math.sqrt(max(b[2] * b[3], 1.0)))
                gts.append((b[0], b[1], r))
        for d in dets:
            if d[4] < min_score:
                continue
            cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
            if all((cx - gx) ** 2 + (cy - gy) ** 2 > r * r for gx, gy, r in gts):
                out.add(f)
                break
    return out


def build_full(gt, grays, shifts, banks, split_at, variants, hard_neg, root):
    make_dirs(root)
    counts = {"train": 0, "val": 0}
    excl = set(gt.meta.get("exclude_frames", []))
    h, w = grays[0].shape
    for t in range(0, len(grays)):
        if t in excl:
            continue
        split = "train" if t < split_at else "val"
        if split == "val" and t < 2 * DT:
            continue  # val stays comparable to v2 (no warmup frames)
        objs = objs_at(gt, shifts, t)
        real_lines = [f"{CLS[c]} {cx/w:.6f} {cy/h:.6f} {LABEL/w:.6f} {LABEL/h:.6f}"
                      for c, (cx, cy) in objs]
        if split == "val":
            img = np.dstack(clamp_chans(grays, t))
            write_yolo(root, split, f"f{t:05d}", img, real_lines)
            counts[split] += 1
            continue
        n_var = variants if t >= 2 * DT else 1  # warmup frames: one copy
        for v in range(n_var):
            rng = random.Random(hash((20260703, t, v)) & 0xFFFFFFFF)
            chans = [c.copy() for c in clamp_chans(grays, t)]
            lines = list(real_lines)
            if t >= 2 * DT:  # no pastes into clamped warmup stacks
                taken = [(cx, cy) for _, (cx, cy) in objs]
                pasted = paste_group(rng, chans, banks, taken, w, h,
                                     [("drone", 4 + v), ("bird", 3)])
                lines += [f"{CLS[c]} {cx/w:.6f} {cy/h:.6f} "
                          f"{LABEL/w:.6f} {LABEL/h:.6f}" for c, cx, cy in pasted]
            write_yolo(root, split, f"f{t:05d}_v{v}", np.dstack(chans), lines)
            counts[split] += 1
        if t in hard_neg:  # extra clean copy: emphasize the false-alarm frame
            img = np.dstack(clamp_chans(grays, t))
            write_yolo(root, split, f"f{t:05d}_hn", img, real_lines)
            counts[split] += 1
    print(f"{root.name}: {counts}")


def build_crops(gt, grays, shifts, banks, split_at, root, tile, n_paste_slices,
                paste_want):
    make_dirs(root)
    counts = {"train": 0, "val": 0}
    excl = set(gt.meta.get("exclude_frames", []))
    h, w = grays[0].shape
    margin = min(70, tile // 4)

    def origin(rng, cx, cy):
        x0 = int(cx - rng.uniform(margin, tile - margin))
        y0 = int(cy - rng.uniform(margin, tile - margin))
        return max(0, min(w - tile, x0)), max(0, min(h - tile, y0))

    for t in range(0, len(grays)):
        if t in excl:
            continue
        split = "train" if t < split_at else "val"
        if split == "val" and t < 2 * DT:
            continue
        rng = random.Random(hash((20260704, t, tile)) & 0xFFFFFFFF)
        chans = clamp_chans(grays, t)
        objs = objs_at(gt, shifts, t)

        slices = []
        drones = [(c, p) for c, p in objs if c == "drone"]
        birds = [(c, p) for c, p in objs if c == "bird"]
        if drones:
            slices.append(("far", origin(rng, *drones[0][1]), False))
        if birds:
            c, p = birds[rng.randrange(len(birds))]
            slices.append(("bird", origin(rng, *p), False))
        if split == "train" and t >= 2 * DT:
            for k in range(n_paste_slices):
                for _try in range(40):
                    x0 = rng.randint(0, w - tile)
                    y0 = rng.randint(0, h - tile)
                    if all(not (x0 - 30 < cx < x0 + tile + 30 and
                                y0 - 30 < cy < y0 + tile + 30)
                           for _, (cx, cy) in objs):
                        slices.append((f"paste{k}", (x0, y0), True))
                        break

        for tag, (x0, y0), is_paste in slices:
            crop_ch = [c[y0:y0 + tile, x0:x0 + tile].copy() for c in chans]
            lines = []
            if not is_paste:
                taken = []
                for cls_name, (cx, cy) in objs:
                    if x0 <= cx < x0 + tile and y0 <= cy < y0 + tile:
                        lines.append(f"{CLS[cls_name]} {(cx-x0)/tile:.6f} "
                                     f"{(cy-y0)/tile:.6f} {LABEL/tile:.6f} "
                                     f"{LABEL/tile:.6f}")
                        taken.append((cx - x0, cy - y0))
            else:
                taken = []
                pasted = paste_group(rng, crop_ch, banks, taken, tile, tile,
                                     paste_want, min_gap=48,
                                     border=min(48, tile // 5))
                lines = [f"{CLS[c]} {cx/tile:.6f} {cy/tile:.6f} "
                         f"{LABEL/tile:.6f} {LABEL/tile:.6f}"
                         for c, cx, cy in pasted]
            write_yolo(root, split, f"f{t:05d}_{tag}", np.dstack(crop_ch), lines)
            counts[split] += 1
    print(f"{root.name}: {counts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-at", type=int, default=342)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--only", choices=["full", "crop640", "crop256"], default=None)
    ap.add_argument("--dt", type=int, default=None,
                    help="temporal channel spacing override (default 6)")
    a = ap.parse_args()
    if a.dt:
        global DT
        DT = a.dt

    gt, grays, shifts = load_stabilized()
    banks = build_banks(gt, grays, shifts, a.split_at)
    print(f"banks: drone {len(banks['drone'])}, bird {len(banks['bird'])}")
    hard_neg = hard_negative_frames(gt, a.split_at)
    print(f"hard-negative train frames: {len(hard_neg)}")

    if a.only in (None, "full"):
        build_full(gt, grays, shifts, banks, a.split_at, a.variants, hard_neg,
                   Path(f"work/dsv3_full_temporal{a.suffix}"))
    if a.only in (None, "crop640"):
        build_crops(gt, grays, shifts, banks, a.split_at,
                    Path(f"work/dsv3_crop640{a.suffix}"), 640, 2,
                    [("drone", 3), ("bird", 3)])
    if a.only in (None, "crop256"):
        build_crops(gt, grays, shifts, banks, a.split_at,
                    Path(f"work/dsv3_crop256{a.suffix}"), 256, 2,
                    [("drone", 2), ("bird", 2)])


if __name__ == "__main__":
    main()
