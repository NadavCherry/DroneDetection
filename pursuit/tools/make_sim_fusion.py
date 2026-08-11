"""Rebuild the simulator dataset as 4-channel RGB+motion tiles.

The head-to-head on live Rivermark sequences said something useful: this
project's 25 M-parameter fusion model acquires 4-8 px targets about five times
better than the simulator-trained nano (0.255 vs 0.055), and then collapses to
near zero above 14 px where the nano is strong. That is two different problems,
not one -- long range is where a lock has to *start*, so the fusion model's
advantage is the valuable one, and its close-range failure is a domain gap
against a renderer it has never seen, which is fixable by showing it one.

Fine-tuning is therefore the right move rather than picking a winner: it keeps
the weights learned from ARD-MAV, NPS-Drones and real footage -- a far wider
dataset than any simulator run -- and adapts them to this renderer.

The existing on-disk simulator dataset can be reused as-is. Every one of its
10,373 frames was captured inside an oracle-guided pursuit and written as
``{tag}_{group}_{frame}.jpg``, so frames are already temporally contiguous
within a group, which is exactly what the motion channel needs. Frames are
regrouped here, differenced, and cut into 640 tiles.

The motion channel is imported from :mod:`tools.make_fusion_combined` rather
than reimplemented. Training and inference must agree bit-for-bit on how that
channel is built, and the surest way to guarantee that is to have one
implementation.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.make_fusion_combined import motion_map  # noqa: E402

STEM_RE = re.compile(r"^(.*)_(\d+)$")


def _sequences(img_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Group frames by their capture flight, ordered within each."""
    seqs: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in sorted(img_dir.glob("*.jpg")):
        m = STEM_RE.match(p.stem)
        if m:
            seqs[m.group(1)].append((int(m.group(2)), p))
    for k in seqs:
        seqs[k].sort()
    return seqs


def _read_label(path: Path, w: int, h: int) -> list[tuple[float, float, float, float]]:
    """YOLO-normalised label file -> absolute xyxy boxes."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    out = []
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in f[1:5])
        out.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def _emit(stack: np.ndarray, boxes, x0: int, y0: int, tile: int,
          out_img: Path, out_lbl: Path, min_side: float) -> int:
    crop = stack[y0:y0 + tile, x0:x0 + tile]
    if crop.shape[:2] != (tile, tile):
        pad = np.zeros((tile, tile, 4), np.uint8)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad
    lines, n = [], 0
    for (bx1, by1, bx2, by2) in boxes:
        bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if not (x0 <= bcx <= x0 + tile and y0 <= bcy <= y0 + tile):
            continue
        bw = max(bx2 - bx1, min_side)
        bh = max(by2 - by1, min_side)
        lines.append(f"0 {(bcx - x0) / tile:.6f} {(bcy - y0) / tile:.6f} "
                     f"{bw / tile:.6f} {bh / tile:.6f}")
        n += 1
    np.save(str(out_img), crop)
    out_lbl.write_text("\n".join(lines))
    return n


def build_split(src: Path, dst: Path, split: str, tile: int, dt: int,
                jitter: float, min_side: float, neg_per_seq: int,
                rng: random.Random) -> tuple[int, int]:
    img_dir, lbl_dir = src / "images" / split, src / "labels" / split
    (dst / "images" / split).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    seqs = _sequences(img_dir)
    for si, (name, frames) in enumerate(sorted(seqs.items())):
        buf: dict[int, np.ndarray] = {}
        negatives = 0
        for idx, path in frames:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            buf[idx] = bgr
            buf.pop(idx - 2 * dt - 2, None)
            # Frames before 2*dt have no history to difference against; using
            # them would train the model on an all-zero motion channel and teach
            # it to ignore the channel entirely.
            if idx < 2 * dt or (idx - 2 * dt) not in buf:
                continue
            h, w = bgr.shape[:2]
            stack = np.dstack([bgr, motion_map(buf, idx, dt)])
            boxes = _read_label(lbl_dir / f"{path.stem}.txt", w, h)
            stem = f"{name}_{idx:05d}"
            if boxes:
                bx1, by1, bx2, by2 = boxes[0]
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                x0 = int(min(max(cx + rng.uniform(-jitter, jitter) * tile - tile / 2,
                                 0), max(w - tile, 0)))
                y0 = int(min(max(cy + rng.uniform(-jitter, jitter) * tile - tile / 2,
                                 0), max(h - tile, 0)))
                n_box += _emit(stack, boxes, x0, y0, tile,
                               dst / "images" / split / f"{stem}.npy",
                               dst / "labels" / split / f"{stem}.txt", min_side)
                n_img += 1
            # Background tiles matter more here than usual: the fusion model's
            # motion channel lights up on anything the ego-registration failed to
            # cancel, and in Rivermark that is a great deal of parallaxing
            # rooftop. Without negatives it learns "bright in channel 4 => drone".
            if negatives < neg_per_seq:
                x0 = rng.randint(0, max(0, w - tile))
                y0 = rng.randint(0, max(0, h - tile))
                if not any(x0 <= (b[0] + b[2]) / 2 <= x0 + tile
                           and y0 <= (b[1] + b[3]) / 2 <= y0 + tile for b in boxes):
                    _emit(stack, [], x0, y0, tile,
                          dst / "images" / split / f"{stem}_neg.npy",
                          dst / "labels" / split / f"{stem}_neg.txt", min_side)
                    n_img += 1
                    negatives += 1
        print(f"  [{split}] seq {si + 1}/{len(seqs)} {name}: {n_img} tiles",
              flush=True)
    return n_img, n_box


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="work/simdata")
    ap.add_argument("--out", default="work/simdata_fusion")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--dt", type=int, default=3)
    ap.add_argument("--jitter", type=float, default=0.32)
    ap.add_argument("--min-side", type=float, default=8.0)
    ap.add_argument("--neg-per-seq", type=int, default=6)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args(argv)

    src, dst = Path(a.src), Path(a.out)
    rng = random.Random(a.seed)
    totals = {}
    for split in ("train", "val"):
        if not (src / "images" / split).exists():
            continue
        totals[split] = build_split(src, dst, split, a.tile, a.dt, a.jitter,
                                    a.min_side, a.neg_per_seq, rng)

    (dst / "data.yaml").write_text(
        f"path: {dst.resolve()}\ntrain: images/train\nval: images/val\n"
        f"channels: 4\nnc: 1\nnames:\n  0: drone\n")
    for split, (ni, nb) in totals.items():
        print(f"{split}: {ni} tiles, {nb} boxes")
    print(f"wrote {dst}/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
