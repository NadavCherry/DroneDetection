"""Build YOLOMG's two-stream dataset from OUR splits, so the comparison is paired.

WHAT MAKES A COMPARISON FAIR HERE
---------------------------------
The competitor has to be trained the way its authors train it, on the data we train on,
and scored by the evaluator that scores us. Those three pull in different directions and
this file is where the trade-off is resolved explicitly:

  same VIDEOS, same SPLIT      -- taken from `tools/make_dataset_external`, not re-derived,
                                  so the two arms cannot drift apart silently.
  same FRAMES                  -- the same stride over the same clips, so neither arm sees
                                  more supervision than the other.
  same LABELS                  -- the same parsers, so ARD-MAV's XML and Dogfight's NPS
                                  re-annotation are read identically for both arms.
  their INPUT REPRESENTATION   -- full frames plus a motion mask, at their image size, not
                                  our 640 px tiles. This is the deliberate asymmetry: we
                                  tile because our detector is designed around tiles, they
                                  do not because theirs is not. Forcing YOLOMG onto our
                                  tiling would be handicapping it with our design decision.
  same EVALUATION              -- both arms are scored full-frame against the same GT json
                                  by `tools/evaluate.py`. Training-time geometry differs;
                                  the measurement does not.

WHAT WE GIVE UP BY DOING IT THIS WAY
------------------------------------
Because the two arms train on differently-shaped inputs, a difference between them is a
difference between two *systems*, not between two loss functions. That is the honest unit
of comparison for "is our detector better than theirs", and it is NOT a controlled ablation
of the temporal representation -- `singleframe_* vs temporal_*` remains the only thing in
this repo that isolates one variable. The tables must keep the two kinds of claim apart.

LAYOUT PRODUCED (what YOLOMG's data yaml expects)
-------------------------------------------------
    <root>/images/<split>/<video>_<frame>.jpg     RGB, full frame
    <root>/images2/<split>/<video>_<frame>.jpg    mask32, full frame, JPEG as upstream
    <root>/labels/<split>/<video>_<frame>.txt     YOLO, class cx cy w h, normalised
    <root>/{train,val,test}.txt                   image paths
    <root>/{train2,val2,test2}.txt                mask paths, LINE-ALIGNED with the above
    <root>/BUILD.json

The paired `.txt`/`2.txt` lists are upstream's convention and they are matched by LINE
NUMBER, not by filename. A sort mismatch between the two would pair every image with some
other frame's mask -- catastrophic and completely silent -- so they are written from one
ordered list in one pass and then verified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.make_dataset_external import (  # noqa: E402  (path set above)
    ARD_ROOT, ARD_TEST_IDS, ARD_VAL_IDS, LOCAL_TEST_GT, LOCAL_TEST_VIDEO,
    LOCAL_TRAIN_GT, LOCAL_TRAIN_VIDEO, NPS_TEST, NPS_TRAIN, NPS_VAL, OUT_ROOT,
    _ard_all, _nps_video, _utcnow, parse_ardmav, parse_nps_dogfight,
    parse_repo_gt)
from tools.sota.motion_mask import YOLOMG_MASK32_DT, fd5_mask  # noqa: E402

#: JPEG quality for both streams. Upstream calls bare `cv2.imwrite`, which is OpenCV's
#: default of 95; we state it rather than inherit it, so a future OpenCV cannot move it.
JPEG_Q = [cv2.IMWRITE_JPEG_QUALITY, 95]

#: Source resolutions, measured on the cluster rather than assumed -- an earlier comment
#: here guessed "NPS is 4K" and was wrong. These decide whether the coordinate-space
#: correction in `motion_mask` (deviation 4) is reachable, so they are recorded:
#:
#:   ARD-MAV   60/60 videos at 1920x1080      -> correction is a no-op throughout
#:   NPS       30/50 at 1920x1080, 20/50 at 1280x960
#:             affected: train 14,15,17,18,19,20,26,27,28,32,33,34
#:                       val   37,38,39,40   <- ALL of them
#:                       test  43,44,45,46
#:
#: The val split mattering most: without the correction the competitor would select its
#: best checkpoint on a validation set whose motion stream was entirely corrupt.
SOURCE_RESOLUTIONS = {"ardmav": {(1920, 1080): 60},
                      "nps": {(1920, 1080): 30, (1280, 960): 20}}

#: The mask stream MUST live in a directory called `images2`, not `mask`. YOLOMG finds a
#: mask's labels with `utils.datasets.img2label_paths2`, which rewrites the substring
#: `/images2/` to `/labels/` and nothing else. Under any other name the rewrite is a no-op,
#: the loader looks for labels next to the masks, finds none, and trains the motion branch
#: against an empty label set -- no error, just a crippled baseline. Naming it is the fix.
MASK_DIR = "images2"


def _frames_needed(indices: list[int], dt: int) -> set[int]:
    """Every decoded frame the masks require: each target plus its two taps."""
    need: set[int] = set()
    for i in indices:
        need.update((i - dt, i, i + dt))
    return need


def _write_pair(root: Path, split: str, stem: str, frame, mask, boxes, w, h) -> bool:
    """Write one image/mask/label triple. Returns False if the image failed to encode."""
    img_p = root / "images" / split / f"{stem}.jpg"
    msk_p = root / MASK_DIR / split / f"{stem}.jpg"
    lbl_p = root / "labels" / split / f"{stem}.txt"
    if not cv2.imwrite(str(img_p), frame, JPEG_Q):
        return False
    # Upstream writes the raw float average; imwrite clips and rounds it to uint8.
    cv2.imwrite(str(msk_p), np.clip(mask, 0, 255).astype(np.uint8), JPEG_Q)

    lines = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
        bw, bh = abs(x2 - x1) / w, abs(y2 - y1) / h
        if bw <= 0 or bh <= 0 or not (0 <= cx <= 1 and 0 <= cy <= 1):
            continue
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    lbl_p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return True


def _emit_video(root: Path, split: str, vid: str, boxes_by_frame: dict,
                cap, n_frames: int, stride: int, dt: int) -> tuple[int, int]:
    """Emit every `stride`-th annotated frame of one video. -> (n_written, n_boxes).

    ONE sequential pass with a ring buffer of the last 2*dt+1 frames, rather than seeking
    to each tap. Both properties matter at this scale:

      memory  -- the obvious implementation collects every frame the masks need into a
                 dict first. On a 1920x1080 ARD-MAV video with ~250 targets that is 750
                 frames, about 4.6 GB. A ring buffer holds five.
      speed   -- `cap.set(CAP_PROP_POS_FRAMES, i)` is not a cheap operation: it seeks to
                 the preceding keyframe and re-decodes forward. Three of those per target,
                 against one linear decode for the whole video.

    Frames are read in order and a target is emitted when it reaches the middle of the
    buffer, which is exactly the alignment `fd5_mask` documents.
    """
    targets = {f for f in boxes_by_frame if f % stride == 0 and dt <= f < n_frames - dt}
    if not targets:
        return 0, 0

    span = 2 * dt + 1
    buf: list = []                      # holds frames [i-span+1 .. i], newest last
    written = n_box = 0
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        buf.append(frame)
        if len(buf) > span:
            buf.pop(0)
        if len(buf) < span:
            continue
        mid = i - dt                    # the frame now sitting at the buffer's centre
        if mid not in targets:
            continue
        taps = (buf[0], buf[dt], buf[2 * dt])
        h, w = taps[1].shape[:2]
        boxes = boxes_by_frame[mid]
        if _write_pair(root, split, f"{vid}_{mid:06d}", taps[1],
                       fd5_mask(*taps), boxes, w, h):
            written += 1
            n_box += len(boxes)
    return written, n_box


def _video_reader(path: Path):
    """Sequential reader: the capture itself, plus its frame count."""
    cap = cv2.VideoCapture(str(path))
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap


def build_nps(root: Path, stride: int, dt: int = YOLOMG_MASK32_DT):
    splits = {"train": NPS_TRAIN, "val": NPS_VAL, "test": NPS_TEST}
    stats = {}
    for split, clips in splits.items():
        tot_f = tot_b = 0
        for clip in clips:
            vp = _nps_video(clip)
            if vp is None:
                print(f"  [{split}] {clip}: NO VIDEO", flush=True)
                continue
            cap, n, _ = _video_reader(vp)
            try:
                nf, nb = _emit_video(root, split, clip, parse_nps_dogfight(clip),
                                     cap, n, stride if split == "train" else stride * 3, dt)
            finally:
                cap.release()
            tot_f += nf
            tot_b += nb
            print(f"  [{split}] {clip}: {nf} frames, {nb} boxes", flush=True)
        stats[split] = (tot_f, tot_b)
    return stats


def _ard_video(vid: str) -> Path | None:
    """ARD-MAV's video directory is lowercase `videos`, which the rest of the repo already
    knows (`_ard_all` globs it). Spelling it `Videos` here cost a build: it silently
    matched nothing on Linux and produced a complete, valid, EMPTY dataset -- exit 0, a
    BUILD.json, and three line-aligned lists of zero pairs. Resolve it case-insensitively
    and fail loudly on nothing found."""
    for parent in (ARD_ROOT / "videos", ARD_ROOT / "Videos"):
        p = parent / f"{vid}.mp4"
        if p.exists():
            return p
    return None


def build_ardmav(root: Path, stride: int, dt: int = YOLOMG_MASK32_DT):
    all_ids = _ard_all()
    if not all_ids:
        raise RuntimeError(f"no ARD-MAV videos under {ARD_ROOT / 'videos'}")
    splits = {"train": [v for v in all_ids
                        if v not in ARD_TEST_IDS and v not in ARD_VAL_IDS],
              "val": ARD_VAL_IDS, "test": ARD_TEST_IDS}
    stats = {}
    for split, vids in splits.items():
        tot_f = tot_b = 0
        for vid in vids:
            vp = _ard_video(vid)
            if vp is None:
                print(f"  [{split}] {vid}: NO VIDEO under {ARD_ROOT}", flush=True)
                continue
            cap, n, _ = _video_reader(vp)
            try:
                nf, nb = _emit_video(root, split, vid, parse_ardmav(vid),
                                     cap, n, stride if split == "train" else stride * 3, dt)
            finally:
                cap.release()
            tot_f += nf
            tot_b += nb
            print(f"  [{split}] {vid}: {nf} frames, {nb} boxes", flush=True)
        stats[split] = (tot_f, tot_b)
    return stats



def build_local(root: Path, stride: int, dt: int = YOLOMG_MASK32_DT):
    """The project's own task for the competitor: train on 07_05, test on 10_06.

    Same videos, same annotations and the same time-ordered 85/15 val cut as our arm --
    `build_local_tiled` in make_dataset_external -- so the only thing that differs is each
    detector's own input representation. 548 annotated frames is a small training set for
    both arms equally.
    """
    boxes_train = parse_repo_gt(LOCAL_TRAIN_GT)
    pos = sorted(f for f, b in boxes_train.items() if b)
    if not pos:
        raise RuntimeError(f"no annotated frames in {LOCAL_TRAIN_GT}")
    cut = int(len(pos) * 0.85)
    plan = {"train": (LOCAL_TRAIN_VIDEO, boxes_train, set(pos[:cut][::stride])),
            # Every held-out frame, matching build_local_tiled exactly -- see the note there.
            "val": (LOCAL_TRAIN_VIDEO, boxes_train, set(pos[cut:])),
            "test": (LOCAL_TEST_VIDEO, parse_repo_gt(LOCAL_TEST_GT), None)}

    stats = {}
    for split, (video, boxes, keep) in plan.items():
        cap, n, _ = _video_reader(video)
        try:
            # A stride of 1 with an explicit frame set: the emitter selects on
            # `f % stride == 0`, so the set is applied by filtering the box map instead.
            sel = {f: b for f, b in boxes.items() if keep is None or f in keep}
            nf, nb = _emit_video(root, split, video.stem, sel, cap, n, 1, dt)
        finally:
            cap.release()
        stats[split] = (nf, nb)
        print(f"  [{split}] {video.stem}: {nf} frames, {nb} boxes", flush=True)
    return stats


def write_lists(root: Path) -> dict:
    """Write the paired path lists, then verify the pairing that upstream assumes.

    Upstream matches `train.txt` line N to `train2.txt` line N. Nothing checks it. A single
    missing mask would shift every subsequent pair by one and train the model on other
    frames' motion -- which does not crash, does not warn, and simply produces a weak
    baseline. So: derive both lists from one sorted list, then assert stem equality.
    """
    counts = {}
    empty = []
    for split in ("train", "val", "test"):
        imgs = sorted((root / "images" / split).glob("*.jpg"))
        if not imgs:
            empty.append(split)
        pairs = [(p, root / MASK_DIR / split / p.name) for p in imgs]
        missing = [p.name for p, m in pairs if not m.exists()]
        if missing:
            raise RuntimeError(f"{split}: {len(missing)} images have no mask, "
                               f"e.g. {missing[:3]} -- refusing to write a skewed pairing")
        (root / f"{split}.txt").write_text(
            "".join(f"{p.resolve()}\n" for p, _ in pairs), encoding="utf-8")
        (root / f"{split}2.txt").write_text(
            "".join(f"{m.resolve()}\n" for _, m in pairs), encoding="utf-8")
        a = (root / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        b = (root / f"{split}2.txt").read_text(encoding="utf-8").splitlines()
        assert len(a) == len(b), f"{split}: list lengths differ"
        assert all(Path(x).name == Path(y).name for x, y in zip(a, b)), \
            f"{split}: image/mask lists are not line-aligned"
        counts[split] = len(a)
    if empty:
        # An empty build is the most dangerous outcome this script has, because every
        # downstream check passes: exit 0, a BUILD.json, three lists that are trivially
        # line-aligned. It cost one build already (a capital V in a directory name) and
        # would have cost a night of GPU time had the training job not tripped over it.
        raise RuntimeError(
            f"no images were written for split(s) {empty} -- refusing to declare an empty "
            f"dataset built. Check that the source videos resolved: an empty build is "
            f"indistinguishable from a good one downstream.")
    return counts


def write_yaml(root: Path, name: str):
    (root / f"{name}.yaml").write_text(
        "\n".join([f"train: {(root / 'train.txt').resolve()}",
                   f"train2: {(root / 'train2.txt').resolve()}",
                   f"val: {(root / 'val.txt').resolve()}",
                   f"val2: {(root / 'val2.txt').resolve()}",
                   f"test: {(root / 'test.txt').resolve()}",
                   f"test2: {(root / 'test2.txt').resolve()}",
                   "", "nc: 1", "names: ['Drone']", ""]),
        encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=("nps", "ardmav", "local"))
    ap.add_argument("--stride", type=int, default=4,
                    help="train stride; val/test use 3x this, matching our own builders")
    ap.add_argument("--dt", type=int, default=YOLOMG_MASK32_DT,
                    help="mask tap spacing; 2 is upstream's mask32 and the default")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.out) if a.out else OUT_ROOT / f"yolomg_{a.dataset}"
    for split in ("train", "val", "test"):
        for sub in ("images", MASK_DIR, "labels"):
            (root / sub / split).mkdir(parents=True, exist_ok=True)

    builder = {"nps": build_nps, "ardmav": build_ardmav,
               "local": build_local}[a.dataset]
    stats = builder(root, a.stride, a.dt)
    counts = write_lists(root)
    write_yaml(root, a.dataset)

    (root / "BUILD.json").write_text(json.dumps({
        "built_utc": _utcnow(),
        "task": f"yolomg-{a.dataset}",
        "comparator": "YOLOMG (arXiv:2503.07115), mask32",
        "mask_dt": a.dt, "mask_span_frames": 2 * a.dt + 1, "causal": False,
        "stride_train": a.stride, "stride_eval": a.stride * 3,
        "full_frame": True, "tile": None, "min_side": 0.0,
        "splits": {k: list(v) for k, v in
                   ({"train": NPS_TRAIN, "val": NPS_VAL, "test": NPS_TEST}
                    if a.dataset == "nps" else
                    {"train": ["07_05"], "val": ["07_05"], "test": ["10_06"]}
                    if a.dataset == "local" else
                    {"train": [v for v in _ard_all() if v not in ARD_TEST_IDS
                               and v not in ARD_VAL_IDS],
                     "val": ARD_VAL_IDS, "test": ARD_TEST_IDS}).items()},
        "counts": counts,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nYOLOMG {a.dataset} -> {root}")
    for split, (nf, nb) in stats.items():
        print(f"  {split}: {nf} frames / {nb} boxes  (listed: {counts[split]})")


if __name__ == "__main__":
    main()
