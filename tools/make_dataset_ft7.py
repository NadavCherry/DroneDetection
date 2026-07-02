"""Multi-frame (3-frame-stacked) 2-class dataset -- the user's labeling
technique, formalized: channels are STABILIZED grays at t-12 / t-6 / t,
so a mover leaves a color-fringed trail while static scenery stays gray.

Pastes simulate temporal signatures: the same patch is pasted into each
channel at velocity-offset positions (drone: rigid, 0..2 px/frame,
including hover) while birds additionally get per-channel scale jitter
(wing flap). Real instances use the user's labels at frame t with the
true motion baked into the earlier channels.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.gt import GroundTruth
from dronedet.stabilize import Stabilizer, warp_to_reference
from dronedet.video import frames

VIDEO = "07_05.mp4"
SPLIT_AT = 342
TILE = 640
LABEL = 24.0
DT = 6                 # channel spacing (frames)
CLS = {"drone": 0, "bird": 1}
ROOT = Path("work/dataset_ft7")
rng = random.Random(20260706)


def cut_patch_gray(img, box, margin=3):
    cx, cy, w, h = box
    r = int(max(w, h) / 2 + margin)
    x0, y0 = int(cx - r), int(cy - r)
    if x0 < 0 or y0 < 0 or x0 + 2 * r > img.shape[1] or y0 + 2 * r > img.shape[0]:
        return None
    return img[y0:y0 + 2 * r, x0:x0 + 2 * r].copy()


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


def erase_near_gray(g, near_box):
    cx, cy, w, h = near_box
    pad = 16
    x1 = max(0, int(cx - w / 2 - pad))
    y1 = max(0, int(cy - h / 2 - pad))
    x2 = min(g.shape[1], int(cx + w / 2 + pad))
    y2 = min(g.shape[0], int(cy + h / 2 + pad))
    bw, bh = x2 - x1, y2 - y1
    sx = max(0, x1 - 265)
    src = g[y1:y2, sx:sx + bw]
    if src.shape[1] != bw:
        return g
    out = g.copy()
    ramp = np.minimum(np.arange(bw), np.arange(bw)[::-1]).astype(np.float32)
    rampy = np.minimum(np.arange(bh), np.arange(bh)[::-1]).astype(np.float32)
    a = np.minimum(np.minimum(ramp[None, :], rampy[:, None]) / 12.0, 1.0)
    out[y1:y2, x1:x2] = (a * src.astype(np.float32) +
                         (1 - a) * out[y1:y2, x1:x2].astype(np.float32)).astype(np.uint8)
    return out


def slice_origin(cx, cy, w, h):
    margin = 70
    x0 = int(cx - rng.uniform(margin, TILE - margin))
    y0 = int(cy - rng.uniform(margin, TILE - margin))
    return max(0, min(w - TILE, x0)), max(0, min(h - TILE, y0))


def window_avoiding(w, h, boxes, tries=60):
    for _ in range(tries):
        x0 = rng.randint(0, w - TILE)
        y0 = rng.randint(0, h - TILE)
        if all(not (x0 - 30 < cx + bw / 2 and cx - bw / 2 < x0 + TILE + 30 and
                    y0 - 30 < cy + bh / 2 and cy - bh / 2 < y0 + TILE + 30)
               for (cx, cy, bw, bh) in boxes):
            return x0, y0
    return None


def main() -> None:
    gt = GroundTruth.load("work/gt_user.json")
    excl = set(gt.meta.get("exclude_frames", []))

    print("stabilizing all frames...", flush=True)
    stab = Stabilizer("translation")
    grays = []
    near = gt.objects["near"]
    for idx, frame in frames(VIDEO):
        m = stab.update(frame)
        g = warp_to_reference(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), m)
        nb = near.box(idx)
        if nb is not None:
            dx, dy = m[0, 2], m[1, 2]
            g = erase_near_gray(g, (nb[0] + dx, nb[1] + dy, nb[2], nb[3]))
        grays.append(g)
    shifts = {int(k): v for k, v in gt.meta.get("shifts", {}).items()}
    n = len(grays)

    far = gt.objects["far"]
    birds = [o for name, o in gt.objects.items() if name.startswith("bird")]

    banks = {"drone": [], "bird": []}
    for idx in range(2 * DT, 141, 2):
        b = far.box(idx)
        if b is not None:
            dx, dy = shifts.get(idx, (0, 0))
            p = cut_patch_gray(grays[idx], (b[0] + dx, b[1] + dy, b[2], b[3]))
            if p is not None:
                banks["drone"].append(p)
    for bo in birds:
        for idx in sorted(bo.frames)[::3]:
            if idx < 2 * DT:
                continue
            dx, dy = shifts.get(idx, (0, 0))
            b = bo.frames[idx]
            p = cut_patch_gray(grays[idx], (b[0] + dx, b[1] + dy, b[2], b[3]))
            if p is not None and min(p.shape[:2]) >= 8:
                banks["bird"].append(p)
    print(f"banks: drone {len(banks['drone'])}, bird {len(banks['bird'])}")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    def stab_box(idx, box):
        dx, dy = shifts.get(idx, (0, 0))
        return (box[0] + dx, box[1] + dy, box[2], box[3])

    counts = {"train": 0, "val": 0}
    h, w = grays[0].shape
    for t in range(2 * DT, n):
        if t in excl:
            continue
        split = "train" if t < SPLIT_AT else "val"
        objs = []
        fb = far.box(t)
        if fb is not None:
            objs.append(("drone", stab_box(t, fb)))
        for bo in birds:
            bb = bo.box(t)
            if bb is not None:
                objs.append(("bird", stab_box(t, bb)))
        boxes_only = [b for _, b in objs]
        chans = [grays[t - 2 * DT], grays[t - DT], grays[t]]

        emitted = []
        if fb is not None:
            sb = stab_box(t, fb)
            emitted.append(("far", slice_origin(sb[0], sb[1], w, h), False))
        bird_boxes = [b for c, b in objs if c == "bird"]
        if bird_boxes:
            bb = bird_boxes[rng.randrange(len(bird_boxes))]
            emitted.append(("bird", slice_origin(bb[0], bb[1], w, h), False))
        if split == "train":
            win = window_avoiding(w, h, boxes_only)
            if win is not None:
                emitted.append(("paste", win, True))

        for tag, (x0, y0), is_paste in emitted:
            crop_ch = [c[y0:y0 + TILE, x0:x0 + TILE].copy() for c in chans]
            lines, taken = [], []
            if not is_paste:
                for cls_name, (cx, cy, bw, bh) in objs:
                    if x0 <= cx < x0 + TILE and y0 <= cy < y0 + TILE:
                        lines.append(f"{CLS[cls_name]} {(cx - x0) / TILE:.6f} "
                                     f"{(cy - y0) / TILE:.6f} {LABEL / TILE:.6f} "
                                     f"{LABEL / TILE:.6f}")
                        taken.append((cx - x0, cy - y0))
            else:
                for cls_name, k in (("drone", 3), ("bird", 3)):
                    bank = banks[cls_name]
                    if not bank:
                        continue
                    for _ in range(k):
                        p0 = bank[rng.randrange(len(bank))].copy()
                        if rng.random() < 0.5:
                            p0 = p0[:, ::-1]
                        s = rng.uniform(0.4, 1.2)
                        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
                        p0 = cv2.resize(p0, None, fx=s, fy=s, interpolation=interp)
                        if min(p0.shape[:2]) < 4:
                            continue
                        p0 = np.clip(p0.astype(np.float32) * rng.uniform(0.85, 1.15),
                                     0, 255).astype(np.uint8)
                        if s < 0.65:
                            p0 = cv2.GaussianBlur(p0, (3, 3), 0)
                        haze = rng.uniform(0.0, 0.5)
                        if cls_name == "drone":
                            speed = rng.uniform(0.0, 2.0)     # rigid, may hover
                        else:
                            speed = rng.uniform(1.0, 4.0)     # birds transit
                        ang = rng.uniform(0, 2 * np.pi)
                        vx, vy = speed * np.cos(ang), speed * np.sin(ang)
                        cx = rng.randint(60, TILE - 60)
                        cy = rng.randint(60, TILE - 60)
                        if any((cx - tx) ** 2 + (cy - ty) ** 2 <= 52 ** 2
                               for tx, ty in taken):
                            continue
                        for k_ch, ch in enumerate(crop_ch):
                            back = (2 - k_ch) * DT
                            px = cx - vx * back
                            py = cy - vy * back
                            p = p0
                            if cls_name == "bird":  # wing flap: shape changes
                                js = rng.uniform(0.75, 1.3)
                                p = cv2.resize(p0, None, fx=js, fy=js,
                                               interpolation=cv2.INTER_LINEAR)
                            feather_paste_gray(ch, p, px, py, haze=haze)
                        taken.append((cx, cy))
                        lines.append(f"{CLS[cls_name]} {cx / TILE:.6f} {cy / TILE:.6f} "
                                     f"{LABEL / TILE:.6f} {LABEL / TILE:.6f}")
            img = np.dstack(crop_ch)  # ch0=t-12, ch1=t-6, ch2=t
            name = f"f{t:05d}_{tag}"
            cv2.imwrite(str(ROOT / f"images/{split}/{name}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (ROOT / f"labels/{split}/{name}.txt").write_text("\n".join(lines) + "\n")
            counts[split] += 1

    (ROOT / "data.yaml").write_text(
        f"path: {ROOT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: drone\n  1: bird\n"
    )
    print(f"ft7 dataset: {counts['train']} train / {counts['val']} val -> {ROOT}")


if __name__ == "__main__":
    main()
