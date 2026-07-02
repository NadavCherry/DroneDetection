"""Sliced dataset + copy-paste augmentation (Kisantal-style) for ft3.

The sliced fine-tune (ft2) still failed to learn the far drone even on its
training slices: most real far-drone instances are near-zero-contrast
(field-skim phase) and tiny boxes get few positive assignments. This
builder multiplies the *clearly visible* drone examples: a patch bank is
cut from train-segment frames where the target is distinct (early descent
against sky/treeline), and K feathered copies are pasted into every train
slice at random positions with flip/brightness jitter. Val slices remain
untouched (honest evaluation).
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
# Label inflation: tiny targets are annotated as fixed MIN_BOX boxes around
# their centers. With true-size 8 px labels, YOLO's task-aligned assignment
# never bootstraps (initial IoU ~0.016 -> cls target ~0.016 -> box gradient
# down-weighted by the same factor -> the model learns "no object"); verified
# empirically on ft2/ft3. 24 px starts ~9x higher. Box size is irrelevant
# downstream -- the pipeline scores by center distance.
MIN_BOX = 24.0
K_PASTE = 4
PATCH_FRAMES = range(0, 141, 2)   # early descent: target clearly visible
ROOT = Path("work/dataset_cp")
rng = random.Random(20260703)


def build_patch_bank(gt: GroundTruth, cache: dict[int, np.ndarray]) -> list[np.ndarray]:
    bank = []
    far = gt.objects["far"]
    for idx in PATCH_FRAMES:
        b = far.box(idx)
        if b is None or idx not in cache:
            continue
        cx, cy, w, h = b
        r = int(max(w, h) / 2 + 3)
        x0, y0 = int(cx - r), int(cy - r)
        img = cache[idx]
        if x0 < 0 or y0 < 0 or x0 + 2 * r > img.shape[1] or y0 + 2 * r > img.shape[0]:
            continue
        bank.append(img[y0:y0 + 2 * r, x0:x0 + 2 * r].copy())
    return bank


def feather_paste(dst: np.ndarray, patch: np.ndarray, cx: int, cy: int) -> None:
    """Alpha-blend the patch with a radial falloff (no rectangular seams)."""
    ph, pw = patch.shape[:2]
    x0, y0 = cx - pw // 2, cy - ph // 2
    yy, xx = np.mgrid[0:ph, 0:pw]
    r = np.hypot((xx - pw / 2) / (pw / 2), (yy - ph / 2) / (ph / 2))
    alpha = np.clip(1.6 - 1.6 * r, 0.0, 1.0)[..., None]  # 1 at center, 0 at edge
    roi = dst[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
    dst[y0:y0 + ph, x0:x0 + pw] = (
        alpha * patch.astype(np.float32) + (1 - alpha) * roi).astype(np.uint8)


def paste_augment(slice_img: np.ndarray, bank: list[np.ndarray],
                  taken: list[tuple[float, float]]) -> list[str]:
    """Paste K jittered patches; returns extra YOLO label lines."""
    lines = []
    for _ in range(K_PASTE):
        patch = bank[rng.randrange(len(bank))].copy()
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        # multi-scale: downscaling a 12 px drone simulates greater distance
        # (the held-out cruise phase is 3-5 px)
        s = rng.uniform(0.35, 1.1)
        if s < 0.999:
            patch = cv2.resize(patch, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        elif s > 1.001:
            patch = cv2.resize(patch, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        if min(patch.shape[:2]) < 5:
            continue
        patch = np.clip(patch.astype(np.float32) * rng.uniform(0.85, 1.15),
                        0, 255).astype(np.uint8)
        ph, pw = patch.shape[:2]
        for _try in range(20):
            cx = rng.randint(pw // 2 + 24, TILE - pw // 2 - 24)
            cy = rng.randint(ph // 2 + 24, TILE - ph // 2 - 24)
            if all((cx - tx) ** 2 + (cy - ty) ** 2 > 40 ** 2 for tx, ty in taken):
                break
        else:
            continue
        feather_paste(slice_img, patch, cx, cy)
        taken.append((cx, cy))
        bw = max(pw - 6.0, MIN_BOX)   # label the drone, not the feather margin
        bh = max(ph - 6.0, MIN_BOX)
        lines.append(f"0 {cx / TILE:.6f} {cy / TILE:.6f} {bw / TILE:.6f} {bh / TILE:.6f}")
    return lines


def slice_origin(cx, cy, w, h):
    margin = 70
    x0 = int(cx - rng.uniform(margin, TILE - margin))
    y0 = int(cy - rng.uniform(margin, TILE - margin))
    return max(0, min(w - TILE, x0)), max(0, min(h - TILE, y0))


def labels_in(gt, idx, x0, y0):
    out, taken = [], []
    for obj in gt.objects.values():
        b = obj.box(idx)
        if b is None or obj.ignore:
            continue
        cx, cy, bw, bh = b
        if not (x0 <= cx < x0 + TILE and y0 <= cy < y0 + TILE):
            continue
        bw, bh = max(bw, MIN_BOX), max(bh, MIN_BOX)
        out.append(f"0 {(cx - x0) / TILE:.6f} {(cy - y0) / TILE:.6f} "
                   f"{bw / TILE:.6f} {bh / TILE:.6f}")
        taken.append((cx - x0, cy - y0))
    return out, taken


def main() -> None:
    gt = GroundTruth.load("work/gt.json")
    excl = set(gt.meta.get("exclude_frames", []))
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    cache = {}
    for idx, frame in frames(VIDEO):
        cache[idx] = frame
    bank = build_patch_bank(gt, cache)
    print(f"patch bank: {len(bank)} clear drone patches "
          f"({bank[0].shape[0]}..{bank[-1].shape[0]} px)")

    counts = {"train": 0, "val": 0}
    for idx, frame in sorted(cache.items()):
        if idx in excl:
            continue
        h, w = frame.shape[:2]
        split = "train" if idx < SPLIT_AT else "val"
        far = gt.objects["far"].box(idx)
        near = gt.objects["near"].box(idx)
        slices = []
        if far is not None:
            slices.append(("far", *slice_origin(far[0], far[1], w, h)))
        if near is not None:
            slices.append(("near", *slice_origin(near[0], near[1], w, h)))
        slices.append(("bg", rng.randint(0, w - TILE), rng.randint(0, h - TILE)))
        for tag, x0, y0 in slices:
            name = f"f{idx:05d}_{tag}"
            crop = frame[y0:y0 + TILE, x0:x0 + TILE].copy()
            lines, taken = labels_in(gt, idx, x0, y0)
            if split == "train":
                lines += paste_augment(crop, bank, taken)
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
    print(f"copy-paste dataset: {counts['train']} train / {counts['val']} val -> {ROOT}")


if __name__ == "__main__":
    main()
