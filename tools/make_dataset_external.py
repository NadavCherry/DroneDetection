#!/usr/bin/env python3
"""Convert external tiny-drone datasets (ARD-MAV, NPS-Drones) into

  (a) the repo's YOLO training layout  (images/ + labels/ + data.yaml), and
  (b) per-video dronedet GT-json files (dronedet/gt.py schema) for evaluation.

Everything downstream scores by CENTER DISTANCE (not IoU), so exact box size is
not critical; sub-`min_side` boxes are inflated to `min_side` (centered) purely
to keep YOLO's IoU-based label assignment stable on few-pixel targets -- the same
lesson baked into the repo's tiny specialist (see docs/reports/round1-pipeline.md).

Formats handled
---------------
ARD-MAV : ARD-MAV/videos/<id>.mp4  +  ARD-MAV/Annotations/<id>/<id>_XXXX.xml
          VOC XML, filename index XXXX is 1-based -> decoded frame XXXX-1.
          bndbox = (xmin,ymin,xmax,ymax) in 1920x1080.
NPS     : Videos/<Clip_N>.*  +  Video_Annotation/Clip_N_gt.txt
          lines "time_layer: F detections: (y1,x1,y2,x2), (..), ...".
          F is 1-based -> decoded frame F-1.  tuple is (top,left,bottom,right).

Read videos strictly sequentially (never seek) -- matches dronedet.video.frames
and sidesteps the MP4 edit-list pre-roll gotchas noted in CLAUDE.md.
"""
import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ARD_ROOT = REPO / "data/external/ard_mav/ARD-MAV"
NPS_VID = REPO / "data/external/nps/Videos_x/Videos"
NPS_ANN = REPO / "data/external/nps/ann-v1_x/Video_Annotation-v1"

#: Dogfight's RE-annotations, which are what the published NPS numbers actually use.
#: `git clone --depth 1 --filter=blob:none --sparse https://github.com/mwaseema/Drone-Detection`
#: then `git sparse-checkout set annotations`.
NPS_ANN_DOGFIGHT = REPO / "data/external/nps/dogfight/annotations/NPS-Drones-Dataset"

#: NPS publishes NO official split. This is the de-facto one every comparable paper uses
#: (Dogfight, TransVisDrone, YOLOMG): clips 1-36 train, 37-40 val, 41-50 test. Quote it as
#: "the Dogfight split" and never as "the official split", because there is not one.
NPS_TRAIN = tuple(f"Clip_{i:03d}" for i in range(1, 37))
NPS_VAL = tuple(f"Clip_{i:03d}" for i in range(37, 41))
NPS_TEST = tuple(f"Clip_{i:03d}" for i in range(41, 51))
OUT_ROOT = REPO / "work/ext_datasets"

# ARD-MAV official split (Guo et al.): 15 test videos, rest train/val.
ARD_TEST_IDS = [f"phantom{n:02d}" for n in
                (5, 8, 9, 10, 19, 30, 41, 43, 46, 47, 58, 63, 65, 70, 86)]
# 5 videos held out of the 45 train/val pool for YOLO val (whole-video holdout).
ARD_VAL_IDS = ["phantom06", "phantom23", "phantom45", "phantom61", "phantom79"]

# Channel spacing for the temporal stack, in frames. 6 is not a free parameter here:
# it is `DT` in make_datasets_v3.py, so the shipped weights, the ablation in
# work/ablation/REPORT.md and anything built by this file all describe the same
# 13-frame aperture (t-12, t-6, t). Change it only as a declared ablation -- MISSION
# item 7 asks for exactly that sweep -- and never silently, or two "temporal" numbers
# in the same table will mean different things.
TEMPORAL_DT = 6


# ----------------------------------------------------------------------------- parsing
def parse_ardmav(vid_id):
    """-> {frame0: [[x1,y1,x2,y2], ...]}  (frame0 is 0-based decoded index)."""
    ann_dir = ARD_ROOT / "Annotations" / vid_id
    out = {}
    for xf in sorted(ann_dir.glob(f"{vid_id}_*.xml")):
        k = int(xf.stem.split("_")[-1])          # 1-based label index
        frame0 = k - 1
        boxes = []
        for o in ET.parse(xf).getroot().findall("object"):
            b = o.find("bndbox")
            boxes.append([int(float(b.find(t).text))
                          for t in ("xmin", "ymin", "xmax", "ymax")])
        out[frame0] = boxes                      # may be [] if drone absent
    return out


_NPS_TUP = re.compile(r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)")


def parse_nps(clip_id):
    """clip_id like 'Clip_5' -> {frame0: [[x1,y1,x2,y2], ...]}."""
    txt = NPS_ANN / f"{clip_id}_gt.txt"
    out = {}
    for ln in open(txt, encoding="utf-8"):
        m = re.search(r"time_layer:\s*(\d+)", ln)
        if not m:
            continue
        frame0 = int(m.group(1)) - 1             # 1-based -> 0-based
        boxes = []
        for a, b, c, d in _NPS_TUP.findall(ln):
            y1, x1, y2, x2 = int(a), int(b), int(c), int(d)   # (top,left,bottom,right)
            boxes.append([x1, y1, x2, y2])
        out[frame0] = boxes
    return out


def parse_nps_dogfight(clip_id):
    """Dogfight's re-annotation of NPS: ``frame,id,x1,y1,x2,y2`` -> {frame0: [[x1,y1,x2,y2]]}.

    THIS is the annotation set the published NPS numbers are computed on -- TransVisDrone's
    0.95 and GLAD's 0.89 both use it, not Purdue's originals. Training or scoring against
    Purdue's would make our NPS number incomparable to every published one, which is the
    only reason to run NPS at all.

    Three annotation sets ship for this dataset and NO two share a coordinate convention:

        Purdue v1        ``time_layer: F detections: (y1,x1,y2,x2)``   corners, SWAPPED order
        Purdue v2        ``frame,id,x,y,w,h,conf,-1,-1,-1``            MOT style, sub-pixel
        Dogfight (here)  ``frame,id,x1,y1,x2,y2``                      corners, normal order

    Reading one with another's parser does not fail. It yields plausible boxes in the wrong
    places, and on a few-pixel target a transposed corner is a total miss reported as a low
    score. Hence a separate function per format rather than one that guesses.

    FRAME INDICES ARE 0-BASED IN THIS FILE, unlike Purdue's and unlike ARD-MAV's XML.
    They are used as-is.

    This was got wrong once, and the symptom was a plausible number rather than an error.
    The code subtracted 1, as it correctly does for the other two formats, which put every
    box on the previous frame and one box on frame -1. NPS is drone-to-drone footage: in
    Clip_041 the target moves ~8 px between consecutive frames against a 10 px box, so a
    one-frame shift drives IoU against the true position to roughly zero. Our NPS AP came
    out at 0.15 where the published bar is 0.89-0.95, and nothing failed.

    The evidence, so nobody has to re-derive it:

        Clip_041.txt   first line "0,1,680,650,690,660", min frame 0, max frame 1528
        the video      decodes to 1529 frames

    0..1528 inclusive is exactly 1529 frames, so the annotations are 0-based and complete.
    `test_nps_dogfight_frames_are_zero_based` pins this against the shipped file.
    """
    txt = NPS_ANN_DOGFIGHT / f"{clip_id}.txt"
    out: dict[int, list] = {}
    for ln in txt.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = ln.strip().split(",")
        if len(parts) < 6:
            continue
        try:
            f, _id, x1, y1, x2, y2 = (float(p) for p in parts[:6])
        except ValueError:
            continue
        if f < 0:
            raise ValueError(f"{txt.name}: negative frame index {f}")
        out.setdefault(int(f), []).append([x1, y1, x2, y2])
    return out


def _nps_video(clip_id):
    """Clip_007 -> the .mov on disk. Dogfight zero-pads to 3 digits, Purdue does not."""
    n = int(clip_id.split("_")[1])
    for cand in (f"Clip_{n}.mov", f"Clip_{n:03d}.mov", f"Clip_{n}.MOV"):
        p = NPS_VID / cand
        if p.exists():
            return p
    return None


def build_nps_tiled(stride_train, stride_val, min_side, tile=640, temporal=False,
                    dt=TEMPORAL_DT, chroma_444=False):
    """NPS-Drones on the Dogfight split, single-frame or temporal.

    Same builder for both arms, differing only in which extractor is called, so the two
    arms of the A/B cannot drift apart in tiling, stride or negatives.
    """
    name = "nps_yolo_temporal" if temporal else "nps_yolo_tiled"
    root = OUT_ROOT / name
    stats = {"train": [0, 0], "val": [0, 0]}
    extractor = extract_yolo_tiled_temporal if temporal else extract_yolo_tiled
    print(f"NPS {'temporal' if temporal else 'single-frame'} build -> {root}")
    for split, ids, stride in (("train", NPS_TRAIN, stride_train),
                               ("val", NPS_VAL, stride_val)):
        for clip in ids:
            video = _nps_video(clip)
            if video is None:
                print(f"  [{split}] {clip}: NO VIDEO, skipped")
                continue
            boxes = parse_nps_dogfight(clip)
            pos = [f for f, b in boxes.items() if b]
            chosen = set(pos[::stride])
            kw = dict(tile=tile)
            if temporal:
                kw.update(dt=dt, chroma_444=chroma_444)
            ni, nb = extractor(video, boxes, chosen,
                               root / "images" / split, root / "labels" / split,
                               clip, min_side, **kw)
            stats[split][0] += ni
            stats[split][1] += nb
            print(f"  [{split}] {clip}: {ni} tiles, {nb} boxes", flush=True)
    write_data_yaml(root, build={"task": name, "min_side": float(min_side),
                                 "tile": tile, "temporal": temporal,
                                 "dt": dt if temporal else None,
                                 "split": "dogfight-1-36/37-40/41-50",
                                 "annotations": "dogfight"})
    print(f"\nNPS {'TEMPORAL' if temporal else 'TILED'} ({tile}px) -> {root}")
    print(f"  train: {stats['train'][0]} tiles / {stats['train'][1]} boxes")
    print(f"  val:   {stats['val'][0]} tiles / {stats['val'][1]} boxes")
    return root


def build_nps_test_gt_dogfight():
    """Per-video GT jsons for the 10 Dogfight test clips, for tools/evaluate.py."""
    out = OUT_ROOT / "gt" / "nps"
    for clip in NPS_TEST:
        video = _nps_video(clip)
        if video is None:
            print(f"  nps-test {clip}: NO VIDEO, skipped")
            continue
        no, nb = write_gt_json(video, parse_nps_dogfight(clip), out / f"{clip}.json")
        print(f"  nps-test {clip}: {no} objs, {nb} boxes -> gt/nps/{clip}.json")


# ----------------------------------------------------------------------------- helpers
def _inflate(x1, y1, x2, y2, min_side, W, H):
    """Return center-format cx,cy,w,h with each side >= min_side, clipped to frame."""
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = max(x2 - x1, min_side), max(y2 - y1, min_side)
    cx = min(max(cx, w / 2), W - w / 2)
    cy = min(max(cy, h / 2), H - h / 2)
    return cx, cy, w, h


def extract_yolo(video_path, boxes_by_frame, frame_ids, img_dir, lbl_dir,
                 prefix, min_side, quality=92):
    """Decode `video_path` sequentially; for each frame in `frame_ids` (a set),
    write a full-res jpg + YOLO label. Returns (n_images, n_boxes)."""
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    want = set(frame_ids)
    cap = cv2.VideoCapture(str(video_path))
    idx, n_img, n_box = 0, 0, 0
    last = max(want) if want else -1
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            H, W = frame.shape[:2]
            boxes = boxes_by_frame.get(idx, [])
            lines = []
            for (x1, y1, x2, y2) in boxes:
                cx, cy, w, h = _inflate(x1, y1, x2, y2, min_side, W, H)
                lines.append(f"0 {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f}")
                n_box += 1
            stem = f"{prefix}_{idx:05d}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            n_img += 1
        idx += 1
    cap.release()
    return n_img, n_box


def _jpeg_params(quality, chroma_444):
    """JPEG flags. ``chroma_444`` disables the 2x2 chroma subsampling.

    Irrelevant for an ordinary BGR frame and *not* irrelevant for a temporal stack.
    OpenCV writes a 3-channel array as BGR: it converts to YCbCr and, by default,
    stores Cb/Cr at half resolution in each axis. In a temporal stack the three
    channels are three different instants, so the chroma planes carry the
    *inter-frame difference* -- which is the entire signal the stack exists to
    provide -- and subsampling halves its resolution before the model ever sees it.
    On an 11.8 px median target moving a few px per frame that is not a rounding
    error. Left at the shipped default so this builder reproduces the existing
    representation exactly; ``--chroma-444`` makes it a one-flag ablation.
    """
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if chroma_444:
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    return params


def _emit_tiles(image, boxes, img_dir, lbl_dir, stem_prefix, min_side, tile,
                rng, jitter, neg_per_frame, quality, chroma_444=False):
    """Cut and write every tile for ONE already-prepared image. Returns (n_img, n_box).

    Split out of `extract_yolo_tiled` so the single-frame and temporal builders share
    one copy of the window and label arithmetic. The jitter draw, the negative
    rejection test and the centre-in-tile rule are each easy to reimplement slightly
    differently, and two builders that tile *almost* the same way would make the
    single-frame vs temporal comparison measure the tiling as well as the input.
    """
    H, W = image.shape[:2]
    n_img = n_box = 0
    windows = []
    for (x1, y1, x2, y2) in boxes:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        jx = rng.uniform(-jitter, jitter) * tile
        jy = rng.uniform(-jitter, jitter) * tile
        x0 = int(min(max(cx + jx - tile / 2, 0), max(W - tile, 0)))
        y0 = int(min(max(cy + jy - tile / 2, 0), max(H - tile, 0)))
        windows.append(("pos", x0, y0))
    for _ in range(neg_per_frame if boxes else neg_per_frame + 1):
        x0 = rng.randint(0, max(W - tile, 0))
        y0 = rng.randint(0, max(H - tile, 0))
        # keep only if it contains no drone center
        if all(not (x0 <= (b[0]+b[2])/2 <= x0+tile and y0 <= (b[1]+b[3])/2 <= y0+tile)
               for b in boxes):
            windows.append(("neg", x0, y0))
    for k, (kind, x0, y0) in enumerate(windows):
        tw = min(tile, W); th = min(tile, H)
        crop = image[y0:y0+th, x0:x0+tw]
        if crop.shape[0] != tile or crop.shape[1] != tile:
            pad = np.zeros((tile, tile, 3), np.uint8)
            pad[:crop.shape[0], :crop.shape[1]] = crop
            crop = pad
        lines = []
        for (x1, y1, x2, y2) in boxes:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if not (x0 <= cx <= x0+tile and y0 <= cy <= y0+tile):
                continue
            lcx, lcy = cx - x0, cy - y0
            w, h = max(x2 - x1, min_side), max(y2 - y1, min_side)
            lines.append(f"0 {lcx/tile:.6f} {lcy/tile:.6f} {w/tile:.6f} {h/tile:.6f}")
            n_box += 1
        stem = f"{stem_prefix}_{k}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), crop, _jpeg_params(quality, chroma_444))
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_img += 1
    return n_img, n_box


def extract_yolo_tiled(video_path, boxes_by_frame, frame_ids, img_dir, lbl_dir,
                       prefix, min_side, tile=640, jitter=0.35, neg_per_frame=1,
                       quality=92):
    """Native-resolution tiles: for each selected frame, emit one `tile`x`tile`
    crop centered (with jitter) on each drone -- the drone keeps its TRUE pixel
    size (no downscale) -- plus `neg_per_frame` random drone-free tiles as hard
    negatives. Labels are all boxes falling inside the tile, in tile coords.

    SINGLE-FRAME appearance only. See `extract_yolo_tiled_temporal` for the
    representation this project's own ablation measures at AP 1.000 against this
    one's 0.110; the two are kept side by side so the comparison can be run.
    """
    import random
    rng = random.Random(1234)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    want = set(frame_ids)
    cap = cv2.VideoCapture(str(video_path))
    idx, n_img, n_box = 0, 0, 0
    last = max(want) if want else -1
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            ni, nb = _emit_tiles(frame, boxes_by_frame.get(idx, []), img_dir, lbl_dir,
                                 f"{prefix}_{idx:05d}", min_side, tile,
                                 rng, jitter, neg_per_frame, quality)
            n_img += ni
            n_box += nb
        idx += 1
    cap.release()
    return n_img, n_box


def _tap_indices(n_held: int, dt: int) -> list[int]:
    """Buffer positions of the three taps ``[t-2*dt, t-dt, t]``, clamped at the start.

    Mirrors `make_datasets_v3.clamp_chans`' ``grays[max(0, t - 2*DT)]`` exactly. Before
    2*dt frames have been seen the older taps repeat the first frame, so the model is
    trained on the same short-trail regime that inference produces at stream start --
    without it the first 12 frames of every sequence are a distribution the model has
    never seen, which is precisely when an interceptor most needs a detection.
    """
    return [max(0, n_held - 1 - 2 * dt), max(0, n_held - 1 - dt), n_held - 1]


def _stack_aligned_to_now(buf, dt: int):
    """The three taps, each warped into the CURRENT frame's coordinates.

    Alignment is to frame *t*, not to frame 0, and that is the whole point.

    `make_datasets_v3` registers every frame to the sequence's first, which is fine on one
    571-frame clip whose camera barely moves (its recorded shifts stay under ~2 px). On
    ARD-MAV -- 60 videos, ~1,800 frames each, a genuinely moving camera -- that shift
    accumulates without bound, and the canvas is the same size as the frame, so content
    walks off the edge. Measured on the first build: ground-truth boxes shifted outside
    the canvas were silently dropped, costing **78 % of the labels on phantom23** and 35 %
    overall (11,831 boxes against the single-frame build's 17,407). Worse than the loss,
    it broke the experiment: the two arms are supposed to differ only in channel content,
    and the temporal arm was training on a third fewer targets -- handicapping precisely
    the claim under test, in the direction that would have made it look false.

    Aligning to *now* bounds every warp by the camera's motion over `dt` frames instead of
    over the whole video, and leaves the current frame unwarped. So:

      * GT boxes at time t need NO shift and nothing falls off the canvas;
      * only the two older taps are cropped at the edges, by a bounded amount;
      * detections at inference are already in original-frame coordinates.

    Translation-only: content at p in frame j sits at p + d_j in reference coordinates, so
    into frame t's it is p + (d_j - d_t). The relative shift is the difference of the
    accumulated ones, and for j = t it is exactly zero.
    """
    idx = _tap_indices(len(buf), dt)
    _, dx_now, dy_now = buf[idx[-1]]
    taps = []
    for i in idx:
        gray, dx, dy = buf[i]
        rx, ry = dx - dx_now, dy - dy_now
        if rx == 0.0 and ry == 0.0:
            taps.append(gray)                       # the current frame: never resampled
        else:
            m = np.float32([[1.0, 0.0, rx], [0.0, 1.0, ry]])
            taps.append(cv2.warpAffine(gray, m, (gray.shape[1], gray.shape[0]),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    return taps


def extract_yolo_tiled_temporal(video_path, boxes_by_frame, frame_ids, img_dir, lbl_dir,
                                prefix, min_side, tile=640, jitter=0.35, neg_per_frame=1,
                                quality=92, dt=TEMPORAL_DT, chroma_444=False,
                                stab_mode="translation"):
    """`extract_yolo_tiled`, but each tile is a stabilised 3-moment temporal stack.

    This is the representation the project's own ablation measures at AP 1.000 against
    a single frame's 0.110 on the same video and target, and the one three independent
    groups have since reproduced. The ARD-MAV configs were training on single frames,
    so a number from them described a method this project does not claim.

    Same recipe as `make_datasets_v3.py` tap for tap -- three grayscale moments at
    ``np.dstack([t-2*dt, t-dt, t])``, oldest first (reversing the order silently costs
    the model its arrow of time), clamped warmup per `_tap_indices` -- with two
    deliberate differences, both forced by ARD-MAV's scale.

    1. Streams a `2*dt+1` ring buffer instead of holding every frame. `make_datasets_v3`
       can afford `load_stabilized()` on one 571-frame clip; 60 videos of 1920x1080 would
       ask for gigabytes each.
    2. **Taps are aligned to the current frame, not to frame 0.** See
       `_stack_aligned_to_now` for why: registering to frame 0 accumulates without bound
       on a moving camera, walks content off a canvas the size of the frame, and cost 35 %
       of the labels on the first build. Ground-truth boxes therefore need NO shift here
       and stay in original-frame coordinates -- which is also where the evaluator's GT
       lives, so nothing has to be undone downstream.
    """
    import random
    from collections import deque

    from dronedet.stabilize import Stabilizer

    rng = random.Random(1234)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    want = set(frame_ids)
    if not want:
        return 0, 0
    last = max(want)

    stab = Stabilizer(stab_mode)
    # (gray, dx, dy) -- RAW grayscale plus its accumulated shift. Storing the frames
    # unwarped is what makes alignment-to-now possible: the warp for each tap is decided
    # at emit time, once the current frame's shift is known.
    buf = deque(maxlen=2 * dt + 1)
    cap = cv2.VideoCapture(str(video_path))
    idx, n_img, n_box = 0, 0, 0
    try:
        while idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            # Every frame is registered, not only the selected ones: the accumulator is a
            # running chain, so skipping frames would break it and quietly misregister
            # everything after the first gap.
            m = stab.update(frame)
            buf.append((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                        float(m[0, 2]), float(m[1, 2])))
            if idx in want:
                ni, nb = _emit_tiles(np.dstack(_stack_aligned_to_now(buf, dt)),
                                     boxes_by_frame.get(idx, []),
                                     img_dir, lbl_dir, f"{prefix}_{idx:05d}",
                                     min_side, tile, rng, jitter, neg_per_frame,
                                     quality, chroma_444)
                n_img += ni
                n_box += nb
            idx += 1
    finally:
        cap.release()
    return n_img, n_box


USER_SPLIT_AT = 342   # 07_05: frames < this -> train, >= -> val (repo convention)


def parse_repo_gt(path, frange=None):
    """dronedet gt.json -> {frame0: [[x1,y1,x2,y2],...]} for NON-ignore objects only.
    Used for the user's 07_05 (far drone). frange=(lo,hi) restricts frames."""
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, o in g["objects"].items():
        if o.get("ignore"):
            continue
        for f, b in o["frames"].items():
            f = int(f)
            if frange and not (frange[0] <= f < frange[1]):
                continue
            cx, cy, w, h = b
            out.setdefault(f, []).append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out


def _index_split(ids):
    """Positional 70/20/10 whole-video split (test = idx%10==0, val = idx%10 in {1,2})."""
    tr, va, te = [], [], []
    for i, v in enumerate(sorted(ids)):
        (te if i % 10 == 0 else va if i % 10 in (1, 2) else tr).append(v)
    return {"train": tr, "val": va, "test": te}


def combined_splits(legacy: bool = False):
    """Whole-video splits per dataset, for the combined multi-dataset corpus.

    ARD-MAV uses **its published split** (``ARD_TEST_IDS``, 15 test videos, Guo et al.),
    which is the only way a number here can be compared with a published one.

    ``legacy=True`` restores the positional ``idx % 10`` split that rounds 5-7 actually
    used. That split ignored ``ARD_TEST_IDS`` -- defined 168 lines above it and never
    referenced -- and chose 6 test videos by position, so **most of the official test
    videos were in the training set** and the resulting ARD-MAV numbers cannot be placed
    beside MGMD's or GLAD's. Keep the flag only to regenerate those old artifacts; do not
    report a number produced with it.

    NPS-Drones publishes no official split, so it keeps the positional one either way --
    say so whenever an NPS number is quoted, rather than implying a standard split.
    """
    nps_ids = [Path(p).stem.replace("_gt", "") for p in NPS_ANN.glob("Clip_*_gt.txt")]
    if legacy:
        ard = _index_split(_ard_all())
    else:
        train_ids = [v for v in _ard_all()
                     if v not in ARD_TEST_IDS and v not in ARD_VAL_IDS]
        ard = {"train": sorted(train_ids), "val": sorted(ARD_VAL_IDS),
               "test": sorted(ARD_TEST_IDS)}
    return {"ardmav": ard, "nps": _index_split(nps_ids),
            "user": {"train": ["07_05"], "val": ["07_05"], "test": ["10_06"]}}


def _sources_for(dataset, vid):
    if dataset == "ardmav":
        return ARD_ROOT / "videos" / f"{vid}.mp4", parse_ardmav(vid)
    if dataset == "nps":
        return _find_nps_video(vid), parse_nps(vid)
    return None, None


def build_combined_tiled(stride_train, stride_val, min_side, tile=640):
    """One merged native-res tiled dataset from ARD-MAV + NPS + user 07_05, with
    per-dataset whole-video splits. Tiles are prefixed <dataset>__<vid> so their
    origin is recoverable."""
    root = OUT_ROOT / "combined_tiled"
    sp = combined_splits()
    stats = {"train": [0, 0], "val": [0, 0]}

    def do(dataset, vid, split, stride):
        vpath, boxes = _sources_for(dataset, vid)
        if vpath is None or not Path(vpath).exists():
            print(f"  !! missing {dataset}/{vid}"); return
        pos = [f for f, b in boxes.items() if b]
        chosen = set(pos[::stride])
        ni, nb = extract_yolo_tiled(vpath, boxes, chosen,
                                    root / "images" / split, root / "labels" / split,
                                    f"{dataset}__{vid}", min_side, tile=tile)
        stats[split][0] += ni; stats[split][1] += nb
        print(f"  [{split}] {dataset}/{vid}: {ni} tiles, {nb} boxes")

    for ds in ("ardmav", "nps"):
        for v in sp[ds]["train"]:
            do(ds, v, "train", stride_train)
        for v in sp[ds]["val"]:
            do(ds, v, "val", stride_val)
    # user 07_05: temporal split of the far (black) drone
    for split, fr, stride in (("train", (0, USER_SPLIT_AT), stride_train),
                              ("val", (USER_SPLIT_AT, 10 ** 9), stride_val)):
        boxes = parse_repo_gt("work/gt_user.json", frange=fr)
        pos = [f for f, b in boxes.items() if b]
        ni, nb = extract_yolo_tiled("data/videos/07_05.mp4", boxes, set(pos[::stride]),
                                    root / "images" / split, root / "labels" / split,
                                    "user__07_05", min_side, tile=tile)
        stats[split][0] += ni; stats[split][1] += nb
        print(f"  [{split}] user/07_05: {ni} tiles, {nb} boxes")
    write_data_yaml(root)
    print(f"\nCOMBINED TILED ({tile}px) -> {root}")
    print(f"  train: {stats['train'][0]} tiles / {stats['train'][1]} boxes")
    print(f"  val:   {stats['val'][0]} tiles / {stats['val'][1]} boxes")
    # record the split so eval knows the test videos
    (root / "splits.json").write_text(json.dumps(sp, indent=1), encoding="utf-8")
    return root


def build_combined_test_gt():
    """Write per-dataset test GT jsons for the combined split's test videos."""
    sp = combined_splits()
    base = OUT_ROOT / "gt_test"
    for v in sp["ardmav"]["test"]:
        write_gt_json(ARD_ROOT / "videos" / f"{v}.mp4", parse_ardmav(v), base / "ardmav" / f"{v}.json")
    for v in sp["nps"]["test"]:
        vid = _find_nps_video(v)
        if vid:
            write_gt_json(vid, parse_nps(v), base / "nps" / f"{v}.json")
    # user test = 10_06 (reuse hardened GT, corrected video path)
    g = json.loads(Path("realtime/work/gt_1006_v2.json").read_text(encoding="utf-8"))
    g["video"] = "data/videos/10_06.mp4"
    (base / "user").mkdir(parents=True, exist_ok=True)
    (base / "user" / "10_06.json").write_text(json.dumps(g), encoding="utf-8")
    print(f"test GT: ardmav={len(sp['ardmav']['test'])} nps={len(sp['nps']['test'])} user=1 -> {base}")


def build_ardmav_train_tiled(stride_train, stride_val, min_side, tile=640):
    root = OUT_ROOT / "ardmav_yolo_tiled"
    train_ids = [v for v in _ard_all() if v not in ARD_TEST_IDS and v not in ARD_VAL_IDS]
    stats = {"train": [0, 0], "val": [0, 0]}
    for split, ids, stride in (("train", train_ids, stride_train),
                               ("val", ARD_VAL_IDS, stride_val)):
        for vid in ids:
            boxes = parse_ardmav(vid)
            pos = [f for f, b in boxes.items() if b]
            chosen = set(pos[::stride])
            ni, nb = extract_yolo_tiled(ARD_ROOT / "videos" / f"{vid}.mp4", boxes, chosen,
                                        root / "images" / split, root / "labels" / split,
                                        vid, min_side, tile=tile)
            stats[split][0] += ni
            stats[split][1] += nb
            print(f"  [{split}] {vid}: {ni} tiles, {nb} boxes")
    write_data_yaml(root, build={"task": "ardmav-train-tiled",
                                 "min_side": float(min_side), "tile": tile,
                                 "temporal": False})
    print(f"\nARD-MAV TILED YOLO ({tile}px) -> {root}")
    print(f"  train: {stats['train'][0]} tiles / {stats['train'][1]} boxes")
    print(f"  val:   {stats['val'][0]} tiles / {stats['val'][1]} boxes")
    return root


def build_ardmav_temporal_tiled(stride_train, stride_val, min_side, tile=640,
                                dt=TEMPORAL_DT, chroma_444=False):
    """ARD-MAV as stabilised temporal stacks, on the OFFICIAL split.

    The sibling of `build_ardmav_train_tiled`, differing only in the input
    representation, so `temporal_ardmav` vs `baseline_ardmav` isolates exactly one
    variable: whether the model sees motion. Same videos, same split, same tiles, same
    stride, same labels.

    The 15 official test videos are never opened here -- they are scored, not trained
    on -- and the 5 val videos are a whole-video holdout, so no frame of a val sequence
    can reach train through a neighbouring tile.
    """
    root = OUT_ROOT / "ardmav_yolo_temporal"
    train_ids = [v for v in _ard_all() if v not in ARD_TEST_IDS and v not in ARD_VAL_IDS]
    stats = {"train": [0, 0], "val": [0, 0]}
    print(f"temporal stacks: dt={dt} (taps t-{2*dt}, t-{dt}, t), "
          f"chroma {'4:4:4' if chroma_444 else '4:2:0 (shipped default)'}, "
          f"min_side={min_side}")
    for split, ids, stride in (("train", train_ids, stride_train),
                               ("val", ARD_VAL_IDS, stride_val)):
        for vid in ids:
            boxes = parse_ardmav(vid)
            pos = [f for f, b in boxes.items() if b]
            chosen = set(pos[::stride])
            ni, nb = extract_yolo_tiled_temporal(
                ARD_ROOT / "videos" / f"{vid}.mp4", boxes, chosen,
                root / "images" / split, root / "labels" / split,
                vid, min_side, tile=tile, dt=dt, chroma_444=chroma_444)
            stats[split][0] += ni
            stats[split][1] += nb
            print(f"  [{split}] {vid}: {ni} tiles, {nb} boxes")
    write_data_yaml(root, build={"task": "ardmav-temporal-tiled",
                                 "min_side": float(min_side), "tile": tile,
                                 "temporal": True, "dt": dt,
                                 "chroma_444": chroma_444})
    print(f"\nARD-MAV TEMPORAL YOLO ({tile}px, dt={dt}) -> {root}")
    print(f"  train: {stats['train'][0]} tiles / {stats['train'][1]} boxes")
    print(f"  val:   {stats['val'][0]} tiles / {stats['val'][1]} boxes")
    return root


def _feather_paste(dst, patch, cx, cy, haze=0.0):
    """Radial-feathered paste (from the ft5 recipe); haze blends toward local bg."""
    ph, pw = patch.shape[:2]
    x0, y0 = cx - pw // 2, cy - ph // 2
    roi = dst[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
    p = patch.astype(np.float32)
    if haze > 0.01:
        p = (1 - haze) * p + haze * roi.reshape(-1, 3).mean(0)
    yy, xx = np.mgrid[0:ph, 0:pw]
    r = np.hypot((xx - pw / 2) / (pw / 2 + 1e-6), (yy - ph / 2) / (ph / 2 + 1e-6))
    a = np.clip(1.6 - 1.6 * r, 0, 1)[..., None]
    dst[y0:y0 + ph, x0:x0 + pw] = (a * p + (1 - a) * roi).astype(np.uint8)


def build_black_paste(n_tiles=5000, tile=640, min_side=12):
    """Add black-drone paste tiles to the combined dataset. Harvests the user's
    black drone (tiny 'far' + large 'near' from 07_05) and pastes multi-scale onto
    DIVERSE ARD-MAV/NPS drone-free backgrounds -> teaches 'black drone on any
    background, any size' rather than the single 07_05 scene."""
    import random
    rng = random.Random(20260705)
    root = OUT_ROOT / "combined_tiled"
    img_dir, lbl_dir = root / "images/train", root / "labels/train"

    # 1. harvest black-drone crops from 07_05 (far = tiny flying, near = large)
    g = json.loads(Path("work/gt_user.json").read_text(encoding="utf-8"))
    far, near = g["objects"]["far"]["frames"], g["objects"]["near"]["frames"]
    cap = cv2.VideoCapture("data/videos/07_05.mp4")
    cache, idx = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        cache[idx] = fr
        idx += 1
    cap.release()

    def crop(frames_d, keys, pad):
        bank = []
        for f in keys:
            b = frames_d.get(str(f))
            fr = cache.get(f)
            if b is None or fr is None:
                continue
            cx, cy, w, h = b
            r = int(max(w, h) / 2 + pad)
            x0, y0 = int(cx - r), int(cy - r)
            if 0 <= x0 and 0 <= y0 and x0 + 2 * r <= fr.shape[1] and y0 + 2 * r <= fr.shape[0]:
                bank.append(fr[y0:y0 + 2 * r, x0:x0 + 2 * r].copy())
        return bank
    far_bank = crop(far, [int(k) for k in far][::2], 3)                 # tiny black drone
    near_bank = crop(near, [int(k) for k in near][::40], 6)             # large black drone
    print(f"black patch bank: far={len(far_bank)} near={len(near_bank)}")
    if not far_bank:
        print("!! no black-drone crops harvested"); return

    # 2. background pool = drone-free ARD-MAV/NPS tiles already in the dataset
    bgs = []
    for lp in list(lbl_dir.glob("ardmav__*.txt")) + list(lbl_dir.glob("nps__*.txt")):
        if lp.read_text(encoding="utf-8").strip() == "":                                # empty label = negative
            ip = img_dir / (lp.stem + ".jpg")
            if ip.exists():
                bgs.append(ip)
    print(f"background pool (drone-free ardmav/nps tiles): {len(bgs)}")
    if not bgs:
        print("!! no background tiles"); return

    made = 0
    for i in range(n_tiles):
        bg = cv2.imread(str(rng.choice(bgs)))
        if bg is None:
            continue
        if bg.shape[:2] != (tile, tile):
            bg = cv2.resize(bg, (tile, tile))
        lines, taken = [], []
        for _ in range(rng.randint(1, 3)):                              # 1-3 drones/tile
            use_near = near_bank and rng.random() < 0.25
            p = rng.choice(near_bank if use_near else far_bank).copy()
            if rng.random() < 0.5:
                p = p[:, ::-1]
            s = rng.uniform(0.12, 0.55) if use_near else rng.uniform(0.5, 2.2)
            p = cv2.resize(p, None, fx=s, fy=s,
                           interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            if min(p.shape[:2]) < 5 or max(p.shape[:2]) > 130:
                continue
            p = np.clip(p.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
            if max(p.shape[:2]) < 10:
                p = cv2.GaussianBlur(p, (3, 3), 0)                      # tiny targets aren't crisp
            ph, pw = p.shape[:2]
            for _t in range(20):
                cx = rng.randint(pw // 2 + 4, tile - pw // 2 - 4)
                cy = rng.randint(ph // 2 + 4, tile - ph // 2 - 4)
                if all((cx - tx) ** 2 + (cy - ty) ** 2 > 40 ** 2 for tx, ty in taken):
                    break
            else:
                continue
            _feather_paste(bg, p, cx, cy, haze=rng.uniform(0.0, 0.5))
            taken.append((cx, cy))
            bw, bh = max(pw, min_side), max(ph, min_side)
            lines.append(f"0 {cx/tile:.6f} {cy/tile:.6f} {bw/tile:.6f} {bh/tile:.6f}")
        if not lines:
            continue
        stem = f"blackpaste__{i:05d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        made += 1
    print(f"added {made} black-drone paste tiles -> {img_dir}")


def write_gt_json(video_path, boxes_by_frame, out_path):
    """Write a dronedet/gt.py-schema GT json for one test video.
    Each simultaneous box becomes object 'drone_<i>' (identity is irrelevant to
    center-distance detection eval; it just needs every GT box present per frame)."""
    objects = {}
    for f in sorted(boxes_by_frame):
        for i, (x1, y1, x2, y2) in enumerate(boxes_by_frame[f]):
            obj = objects.setdefault(f"drone_{i}", {"ignore": False, "frames": {}})
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            obj["frames"][str(f)] = [cx, cy, float(x2 - x1), float(y2 - y1)]
    gt = {"video": str(video_path),
          "meta": {"shifts": {}, "exclude_frames": []},
          "objects": objects}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(gt), encoding="utf-8")
    return len(objects), sum(len(o["frames"]) for o in objects.values())


def write_data_yaml(root, names=("drone",), build: dict | None = None):
    lines = [f"path: {root.resolve()}", "train: images/train", "val: images/val",
             "names:"]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # BUILD.json: what this build actually DID, recorded by the code that did it.
    #
    # `tools/train.py` has to know whether labels on disk are true extents or inflated,
    # because both ARD-MAV builds write to the same directory and running the headline
    # experiment against the control's labels produces a plausible curve. It used to infer
    # that from box statistics, and inference is genuinely ambiguous here: `--label-px 24`
    # writes every box at exactly 24 (a fixed size), while `--min-side 12` is a FLOOR that
    # only lifts boxes already below it -- and NPS piles up at 10 px naturally because
    # Dogfight's corners are integers. No threshold on the size distribution separates
    # "floored at 10" from "genuinely smallest at 10".
    #
    # The builder does not have to guess: it knows. Recording it turns a heuristic into a
    # fact, and the heuristic stays only as a fallback for datasets built before this.
    if build is not None:
        (root / "BUILD.json").write_text(
            json.dumps({"built_utc": _utcnow(), **build}, indent=2) + "\n",
            encoding="utf-8")


def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- builders
def build_ardmav_train(stride_train, stride_val, min_side):
    root = OUT_ROOT / "ardmav_yolo"
    train_ids = [v for v in _ard_all() if v not in ARD_TEST_IDS and v not in ARD_VAL_IDS]
    stats = {"train": [0, 0], "val": [0, 0]}
    for split, ids, stride in (("train", train_ids, stride_train),
                               ("val", ARD_VAL_IDS, stride_val)):
        for vid in ids:
            boxes = parse_ardmav(vid)
            # subsample annotated frames that actually contain a drone, plus a
            # fraction of empty frames as hard negatives.
            pos = [f for f, b in boxes.items() if b]
            neg = [f for f, b in boxes.items() if not b]
            chosen = set(pos[::stride]) | set(neg[::max(stride * 4, 1)])
            ni, nb = extract_yolo(ARD_ROOT / "videos" / f"{vid}.mp4", boxes, chosen,
                                  root / "images" / split, root / "labels" / split,
                                  vid, min_side)
            stats[split][0] += ni
            stats[split][1] += nb
            print(f"  [{split}] {vid}: {ni} imgs, {nb} boxes")
    write_data_yaml(root)
    print(f"\nARD-MAV YOLO -> {root}")
    print(f"  train: {stats['train'][0]} imgs / {stats['train'][1]} boxes")
    print(f"  val:   {stats['val'][0]} imgs / {stats['val'][1]} boxes")
    return root


def build_ardmav_test_gt():
    out = OUT_ROOT / "gt" / "ardmav"
    for vid in ARD_TEST_IDS:
        boxes = parse_ardmav(vid)
        no, nb = write_gt_json(ARD_ROOT / "videos" / f"{vid}.mp4", boxes,
                               out / f"{vid}.json")
        print(f"  ardmav-test {vid}: {no} objs, {nb} boxes -> gt/ardmav/{vid}.json")


def build_nps_test_gt():
    """Purdue's ORIGINAL v1 annotations, for all 50 clips. NOT the benchmark ground truth.

    This writes to `gt/nps_purdue_v1`, deliberately not to `gt/nps`, and the separation is
    load-bearing rather than tidy-minded. `tools/evaluate.py` defines the evaluated set as
    `gt_dir.glob("*.json")` -- whatever files are in the directory ARE the test set. Both
    this function and `build_nps_test_gt_dogfight` used to write to `gt/nps`, so:

      * running this one filled the "test set" with all 50 clips, 36 of which are TRAINING
        clips, and the reported AP would have covered data the models were fitted on;
      * whichever ran last silently decided which annotation convention the 10 real test
        clips were scored under, and the two do not agree -- Purdue v1 is
        `(y1,x1,y2,x2)` while Dogfight's is `(x1,y1,x2,y2)` (see `parse_nps_dogfight`).
        A transposed corner on a few-pixel target is a total miss reported as a low score.

    Every published NPS number this project compares against -- TransVisDrone 0.95,
    GLAD 0.89 -- is computed on Dogfight's re-annotations, so `gt/nps` is Dogfight's and
    this one lives elsewhere.
    """
    out = OUT_ROOT / "gt" / "nps_purdue_v1"
    for txt in sorted(NPS_ANN.glob("Clip_*_gt.txt")):
        clip = txt.stem.replace("_gt", "")
        vid = _find_nps_video(clip)
        if vid is None:
            print(f"  !! no video for {clip}")
            continue
        boxes = parse_nps(clip)
        no, nb = write_gt_json(vid, boxes, out / f"{clip}.json")
        print(f"  nps-purdue-v1 {clip} ({vid.name}): {no} objs, {nb} boxes")
    print(f"\nwrote Purdue v1 GT for ALL 50 clips -> {out}\n"
          f"  This is NOT the benchmark test set. Scoring against it would include 36\n"
          f"  training clips and use a different corner convention. The benchmark GT is\n"
          f"  gt/nps, built by --task nps-gt-dogfight.")


def _ard_all():
    return sorted(p.stem for p in (ARD_ROOT / "videos").glob("*.mp4"))


def _find_nps_video(clip):
    if not NPS_VID.exists():
        return None
    for ext in (".mp4", ".mov", ".avi", ".MOV", ".MP4", ".m4v"):
        p = NPS_VID / f"{clip}{ext}"
        if p.exists():
            return p
    # some releases name them differently, e.g. Clip_5.mov vs clip_5
    cand = list(NPS_VID.glob(f"{clip}.*")) + list(NPS_VID.glob(f"{clip.lower()}.*"))
    return cand[0] if cand else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["ardmav-train", "ardmav-train-tiled",
                             "ardmav-temporal-tiled", "ardmav-gt", "nps-gt",
                             "combined-tiled", "combined-gt", "black-paste",
                             "nps-train-tiled", "nps-temporal-tiled", "nps-gt-dogfight",
                             "all"])
    ap.add_argument("--stride-train", type=int, default=4)
    ap.add_argument("--stride-val", type=int, default=10)
    ap.add_argument("--min-side", type=int, default=12)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--n-tiles", type=int, default=5000)
    ap.add_argument("--dt", type=int, default=TEMPORAL_DT,
                    help=f"temporal channel spacing in frames (default {TEMPORAL_DT}, "
                         "matching make_datasets_v3.DT; taps are t-2*dt, t-dt, t)")
    ap.add_argument("--chroma-444", action="store_true",
                    help="write JPEG without chroma subsampling. For a temporal stack "
                         "the chroma planes carry the inter-frame difference, which "
                         "4:2:0 stores at half resolution; off by default so the "
                         "shipped representation is reproduced exactly")
    a = ap.parse_args()
    if a.task in ("ardmav-train", "all"):
        build_ardmav_train(a.stride_train, a.stride_val, a.min_side)
    if a.task == "ardmav-train-tiled":
        build_ardmav_train_tiled(a.stride_train, a.stride_val, a.min_side, tile=a.tile)
    if a.task == "nps-train-tiled":
        build_nps_tiled(a.stride_train, a.stride_val, a.min_side, tile=a.tile,
                        temporal=False)
    if a.task == "nps-temporal-tiled":
        build_nps_tiled(a.stride_train, a.stride_val, a.min_side, tile=a.tile,
                        temporal=True, dt=a.dt, chroma_444=a.chroma_444)
    if a.task == "nps-gt-dogfight":
        build_nps_test_gt_dogfight()
    if a.task == "ardmav-temporal-tiled":
        build_ardmav_temporal_tiled(a.stride_train, a.stride_val, a.min_side,
                                    tile=a.tile, dt=a.dt, chroma_444=a.chroma_444)
    if a.task == "combined-tiled":
        build_combined_tiled(a.stride_train, a.stride_val, a.min_side, tile=a.tile)
    if a.task == "combined-gt":
        build_combined_test_gt()
    if a.task == "black-paste":
        build_black_paste(n_tiles=a.n_tiles, tile=a.tile, min_side=a.min_side)
    if a.task in ("ardmav-gt", "all"):
        build_ardmav_test_gt()
    if a.task in ("nps-gt", "all"):
        build_nps_test_gt()
