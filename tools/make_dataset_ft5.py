"""Tiny-specialist dataset (ft5): paste-only training on real backgrounds.

Diagnosis that led here: identical trainer + tiny pasted targets learn
perfectly on sky-only synthetic images (P/R ~0.99) but not at all when the
180 px near drone shares the images -- its alignment scores dominate the
normalized loss and tiny targets' gradients are diluted to nothing.

So the tiny specialist trains on slices that contain NO real drone at all
(windows avoid both GT objects); every positive is a crisp multi-scale
pasted patch with a fixed 24 px inflated label. Foliage / treeline / sky
backgrounds provide hard negatives. The near drone is handled by the
separate full-frame expert (ft1) at inference.
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

VIDEO = "data/videos/07_05.mp4"
SPLIT_AT = 342
TILE = 640
LABEL = 24.0
K_PASTE = 5
SLICES_PER_FRAME = 3
PATCH_FRAMES = range(0, 141, 2)
ROOT = Path("work/dataset_ft5")
rng = random.Random(20260704)


def feather_paste(dst, patch, cx, cy, haze=0.0):
    """Radial-feathered paste; ``haze`` blends the patch toward the local
    background mean first (atmospheric contrast loss with distance)."""
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


def erase_near(frame: np.ndarray, near_box) -> np.ndarray:
    """Blend a dirt patch from beside the (static) near drone over it.

    A 640 window in a 1280x720 frame cannot geometrically avoid the
    centered near drone, so it is erased instead of avoided -- otherwise
    it would be an unlabeled object in the tiny-specialist's backgrounds.
    """
    cx, cy, w, h = near_box
    pad = 16
    x1 = max(0, int(cx - w / 2 - pad))
    y1 = max(0, int(cy - h / 2 - pad))
    x2 = min(frame.shape[1], int(cx + w / 2 + pad))
    y2 = min(frame.shape[0], int(cy + h / 2 + pad))
    bw, bh = x2 - x1, y2 - y1
    sx = max(0, x1 - 265)  # plain dirt road to the left, same lighting band
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


def window_avoiding(w, h, boxes, tries=40):
    """Random TILE window not intersecting any (cx, cy, bw, bh) box."""
    for _ in range(tries):
        x0 = rng.randint(0, w - TILE)
        y0 = rng.randint(0, h - TILE)
        ok = True
        for (cx, cy, bw, bh) in boxes:
            pad = 30
            if (x0 - pad < cx + bw / 2 and cx - bw / 2 < x0 + TILE + pad and
                    y0 - pad < cy + bh / 2 and cy - bh / 2 < y0 + TILE + pad):
                ok = False
                break
        if ok:
            return x0, y0
    return None


def main() -> None:
    gt = GroundTruth.load("work/gt.json")
    cache = {}
    for idx, frame in frames(VIDEO):
        cache[idx] = frame

    far = gt.objects["far"]
    bank = []
    for idx in PATCH_FRAMES:
        b = far.box(idx)
        if b is None:
            continue
        cx, cy, w, h = b
        r = int(max(w, h) / 2 + 3)
        x0, y0 = int(cx - r), int(cy - r)
        img = cache[idx]
        if 0 <= x0 and 0 <= y0 and x0 + 2 * r <= img.shape[1] and y0 + 2 * r <= img.shape[0]:
            bank.append(img[y0:y0 + 2 * r, x0:x0 + 2 * r].copy())
    print(f"patch bank: {len(bank)}")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for idx, frame in sorted(cache.items()):
        split = "train" if idx < SPLIT_AT else "val"
        h, w = frame.shape[:2]
        near_b = gt.objects["near"].box(idx)
        if near_b is not None:
            frame = erase_near(frame, near_b)
        avoid = []
        far_b = gt.objects["far"].box(idx)
        if far_b is not None:
            avoid.append(far_b)
        for k in range(SLICES_PER_FRAME):
            win = window_avoiding(w, h, avoid)
            if win is None:
                continue
            x0, y0 = win
            crop = frame[y0:y0 + TILE, x0:x0 + TILE].copy()
            lines, taken = [], []
            for _ in range(K_PASTE):
                p = bank[rng.randrange(len(bank))].copy()
                if rng.random() < 0.5:
                    p = p[:, ::-1]
                s = rng.uniform(0.35, 1.2)
                interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
                p = cv2.resize(p, None, fx=s, fy=s, interpolation=interp)
                if min(p.shape[:2]) < 5:
                    continue
                p = np.clip(p.astype(np.float32) * rng.uniform(0.85, 1.15),
                            0, 255).astype(np.uint8)
                if s < 0.65:  # optics: a sub-8 px target is never crisp
                    p = cv2.GaussianBlur(p, (3, 3), 0)
                haze = rng.uniform(0.0, 0.55)
                ph, pw = p.shape[:2]
                for _try in range(20):
                    cx = rng.randint(pw // 2 + 22, TILE - pw // 2 - 22)
                    cy = rng.randint(ph // 2 + 22, TILE - ph // 2 - 22)
                    if all((cx - tx) ** 2 + (cy - ty) ** 2 > 48 ** 2 for tx, ty in taken):
                        break
                else:
                    continue
                feather_paste(crop, p, cx, cy, haze=haze)
                taken.append((cx, cy))
                lines.append(f"0 {cx / TILE:.6f} {cy / TILE:.6f} "
                             f"{LABEL / TILE:.6f} {LABEL / TILE:.6f}")
            name = f"f{idx:05d}_{k}"
            cv2.imwrite(str(ROOT / f"images/{split}/{name}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (ROOT / f"labels/{split}/{name}.txt").write_text("\n".join(lines) + "\n")
            counts[split] += 1

    (ROOT / "data.yaml").write_text(
        f"path: {ROOT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: drone\n"
    )
    print(f"ft5 dataset: {counts['train']} train / {counts['val']} val -> {ROOT}")


if __name__ == "__main__":
    main()
