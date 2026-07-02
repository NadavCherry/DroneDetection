"""2-class (drone / bird) tiny-specialist dataset from the USER's labels.

Per frame, up to three 640 slices with the near drone erased:
  * a "real-far" slice window around the user-labeled drone,
  * a "real-bird" slice around a random labeled bird (when present),
  * a "paste" slice avoiding all real GT, with pasted drone+bird patches
    (multi-scale + haze, as in ft5) for abundant clean positives.
Real slices label every GT object inside the window (both classes) with
24 px inflated boxes -- no unlabeled-object poison. Val slices are
real-label only (no pastes), so internal metrics reflect real instances.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.gt import GroundTruth
from dronedet.video import frames

VIDEO = "07_05.mp4"
SPLIT_AT = 342
TILE = 640
LABEL = 24.0
CLS = {"drone": 0, "bird": 1}
ROOT = Path("work/dataset_ft6")
rng = random.Random(20260705)


def feather_paste(dst, patch, cx, cy, haze=0.0):
    ph, pw = patch.shape[:2]
    x0, y0 = cx - pw // 2, cy - ph // 2
    roi = dst[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
    p = patch.astype(np.float32)
    if haze > 0.01:
        p = (1 - haze) * p + haze * roi.reshape(-1, 3).mean(0)
    yy, xx = np.mgrid[0:ph, 0:pw]
    r = np.hypot((xx - pw / 2) / (pw / 2), (yy - ph / 2) / (ph / 2))
    a = np.clip(1.6 - 1.6 * r, 0, 1)[..., None]
    dst[y0:y0 + ph, x0:x0 + pw] = (a * p + (1 - a) * roi).astype(np.uint8)


def erase_near(frame, near_box):
    cx, cy, w, h = near_box
    pad = 16
    x1 = max(0, int(cx - w / 2 - pad))
    y1 = max(0, int(cy - h / 2 - pad))
    x2 = min(frame.shape[1], int(cx + w / 2 + pad))
    y2 = min(frame.shape[0], int(cy + h / 2 + pad))
    bw, bh = x2 - x1, y2 - y1
    sx = max(0, x1 - 265)
    src = frame[y1:y2, sx:sx + bw]
    if src.shape[1] != bw:
        return frame
    out = frame.copy()
    ramp = np.minimum(np.arange(bw), np.arange(bw)[::-1]).astype(np.float32)
    rampy = np.minimum(np.arange(bh), np.arange(bh)[::-1]).astype(np.float32)
    a = np.minimum(np.minimum(ramp[None, :], rampy[:, None]) / 12.0, 1.0)[..., None]
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


def slice_origin(cx, cy, w, h):
    margin = 70
    x0 = int(cx - rng.uniform(margin, TILE - margin))
    y0 = int(cy - rng.uniform(margin, TILE - margin))
    return max(0, min(w - TILE, x0)), max(0, min(h - TILE, y0))


def real_labels(objs, x0, y0):
    lines, taken = [], []
    for cls_name, box in objs:
        cx, cy = box[0], box[1]
        if not (x0 <= cx < x0 + TILE and y0 <= cy < y0 + TILE):
            continue
        lines.append(f"{CLS[cls_name]} {(cx - x0) / TILE:.6f} {(cy - y0) / TILE:.6f} "
                     f"{LABEL / TILE:.6f} {LABEL / TILE:.6f}")
        taken.append((cx - x0, cy - y0))
    return lines, taken


def window_avoiding(w, h, boxes, tries=60):
    for _ in range(tries):
        x0 = rng.randint(0, w - TILE)
        y0 = rng.randint(0, h - TILE)
        if all(not (x0 - 30 < cx + bw / 2 and cx - bw / 2 < x0 + TILE + 30 and
                    y0 - 30 < cy + bh / 2 and cy - bh / 2 < y0 + TILE + 30)
               for (cx, cy, bw, bh) in boxes):
            return x0, y0
    return None


def paste_many(crop, banks, taken):
    lines = []
    for cls_name, bank, k in (("drone", banks["drone"], 3), ("bird", banks["bird"], 3)):
        if not bank:
            continue
        for _ in range(k):
            p = bank[rng.randrange(len(bank))].copy()
            if rng.random() < 0.5:
                p = p[:, ::-1]
            s = rng.uniform(0.4, 1.2)
            interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
            p = cv2.resize(p, None, fx=s, fy=s, interpolation=interp)
            if min(p.shape[:2]) < 4:
                continue
            p = np.clip(p.astype(np.float32) * rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
            if s < 0.65:
                p = cv2.GaussianBlur(p, (3, 3), 0)
            haze = rng.uniform(0.0, 0.5)
            ph, pw = p.shape[:2]
            for _try in range(20):
                cx = rng.randint(pw // 2 + 22, TILE - pw // 2 - 22)
                cy = rng.randint(ph // 2 + 22, TILE - ph // 2 - 22)
                if all((cx - tx) ** 2 + (cy - ty) ** 2 > 44 ** 2 for tx, ty in taken):
                    break
            else:
                continue
            feather_paste(crop, p, cx, cy, haze=haze)
            taken.append((cx, cy))
            lines.append(f"{CLS[cls_name]} {cx / TILE:.6f} {cy / TILE:.6f} "
                         f"{LABEL / TILE:.6f} {LABEL / TILE:.6f}")
    return lines


def main() -> None:
    gt = GroundTruth.load("work/gt_user.json")
    excl = set(gt.meta.get("exclude_frames", []))
    cache = {}
    for idx, frame in frames(VIDEO):
        cache[idx] = frame

    far = gt.objects["far"]
    birds = [o for name, o in gt.objects.items() if name.startswith("bird")]
    near = gt.objects["near"]

    banks = {"drone": [], "bird": []}
    for idx in range(0, 141, 2):  # clear descent phase
        b = far.box(idx)
        if b is not None:
            p = cut_patch(cache[idx], b)
            if p is not None:
                banks["drone"].append(p)
    for bo in birds:
        for idx in sorted(bo.frames)[::3]:
            p = cut_patch(cache[idx], bo.frames[idx])
            if p is not None and min(p.shape[:2]) >= 8:
                banks["bird"].append(p)
    print(f"banks: drone {len(banks['drone'])}, bird {len(banks['bird'])}")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for idx, frame in sorted(cache.items()):
        if idx in excl:
            continue
        split = "train" if idx < SPLIT_AT else "val"
        h, w = frame.shape[:2]
        nb = near.box(idx)
        if nb is not None:
            frame = erase_near(frame, nb)
        objs = []
        fb = far.box(idx)
        if fb is not None:
            objs.append(("drone", fb))
        for bo in birds:
            bb = bo.box(idx)
            if bb is not None:
                objs.append(("bird", bb))
        boxes_only = [b for _, b in objs]

        emitted = []
        if fb is not None:  # real-far slice
            emitted.append(("far", slice_origin(fb[0], fb[1], w, h), False))
        bird_boxes = [b for c, b in objs if c == "bird"]
        if bird_boxes:  # real-bird slice
            bb = bird_boxes[rng.randrange(len(bird_boxes))]
            emitted.append(("bird", slice_origin(bb[0], bb[1], w, h), False))
        if split == "train":  # paste slice (train only)
            win = window_avoiding(w, h, boxes_only)
            if win is not None:
                emitted.append(("paste", win, True))

        for tag, (x0, y0), is_paste in emitted:
            crop = frame[y0:y0 + TILE, x0:x0 + TILE].copy()
            lines, taken = real_labels(objs, x0, y0)
            if is_paste:
                lines = paste_many(crop, banks, taken)
            name = f"f{idx:05d}_{tag}"
            cv2.imwrite(str(ROOT / f"images/{split}/{name}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (ROOT / f"labels/{split}/{name}.txt").write_text("\n".join(lines) + "\n")
            counts[split] += 1

    (ROOT / "data.yaml").write_text(
        f"path: {ROOT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: drone\n  1: bird\n"
    )
    print(f"ft6 dataset: {counts['train']} train / {counts['val']} val -> {ROOT}")


if __name__ == "__main__":
    main()
