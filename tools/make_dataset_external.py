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
NPS_VID = REPO / "data/external/nps/videos"          # populated after Videos.zip extract
NPS_ANN = REPO / "data/external/nps/annots/Video_Annotation"
OUT_ROOT = REPO / "work/ext_datasets"

# ARD-MAV official split (Guo et al.): 15 test videos, rest train/val.
ARD_TEST_IDS = [f"phantom{n:02d}" for n in
                (5, 8, 9, 10, 19, 30, 41, 43, 46, 47, 58, 63, 65, 70, 86)]
# 5 videos held out of the 45 train/val pool for YOLO val (whole-video holdout).
ARD_VAL_IDS = ["phantom06", "phantom23", "phantom45", "phantom61", "phantom79"]


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
    for ln in open(txt):
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
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
            n_img += 1
        idx += 1
    cap.release()
    return n_img, n_box


def extract_yolo_tiled(video_path, boxes_by_frame, frame_ids, img_dir, lbl_dir,
                       prefix, min_side, tile=640, jitter=0.35, neg_per_frame=1,
                       quality=92):
    """Native-resolution tiles: for each selected frame, emit one `tile`x`tile`
    crop centered (with jitter) on each drone -- the drone keeps its TRUE pixel
    size (no downscale) -- plus `neg_per_frame` random drone-free tiles as hard
    negatives. Labels are all boxes falling inside the tile, in tile coords."""
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
            H, W = frame.shape[:2]
            boxes = boxes_by_frame.get(idx, [])
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
                crop = frame[y0:y0+th, x0:x0+tw]
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
                stem = f"{prefix}_{idx:05d}_{k}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
                n_img += 1
        idx += 1
    cap.release()
    return n_img, n_box


USER_SPLIT_AT = 342   # 07_05: frames < this -> train, >= -> val (repo convention)


def parse_repo_gt(path, frange=None):
    """dronedet gt.json -> {frame0: [[x1,y1,x2,y2],...]} for NON-ignore objects only.
    Used for the user's 07_05 (far drone). frange=(lo,hi) restricts frames."""
    g = json.loads(Path(path).read_text())
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


def combined_splits():
    """Whole-video 70/20/10 split per dataset (test=idx%10==0, val=idx%10 in {1,2}).
    Every dataset lands in train, val AND test."""
    def split(ids):
        tr, va, te = [], [], []
        for i, v in enumerate(sorted(ids)):
            (te if i % 10 == 0 else va if i % 10 in (1, 2) else tr).append(v)
        return {"train": tr, "val": va, "test": te}
    ard = split(_ard_all())
    nps_ids = [Path(p).stem.replace("_gt", "") for p in NPS_ANN.glob("Clip_*_gt.txt")]
    nps = split(nps_ids)
    return {"ardmav": ard, "nps": nps,
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
    (root / "splits.json").write_text(json.dumps(sp, indent=1))
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
    g = json.loads(Path("realtime/work/gt_1006_v2.json").read_text())
    g["video"] = "data/videos/10_06.mp4"
    (base / "user").mkdir(parents=True, exist_ok=True)
    (base / "user" / "10_06.json").write_text(json.dumps(g))
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
    write_data_yaml(root)
    print(f"\nARD-MAV TILED YOLO ({tile}px) -> {root}")
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
    g = json.loads(Path("work/gt_user.json").read_text())
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
        if lp.read_text().strip() == "":                                # empty label = negative
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
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
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
    out_path.write_text(json.dumps(gt))
    return len(objects), sum(len(o["frames"]) for o in objects.values())


def write_data_yaml(root, names=("drone",)):
    lines = [f"path: {root.resolve()}", "train: images/train", "val: images/val",
             "names:"]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n")


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
    out = OUT_ROOT / "gt" / "nps"
    for txt in sorted(NPS_ANN.glob("Clip_*_gt.txt")):
        clip = txt.stem.replace("_gt", "")
        vid = _find_nps_video(clip)
        if vid is None:
            print(f"  !! no video for {clip}")
            continue
        boxes = parse_nps(clip)
        no, nb = write_gt_json(vid, boxes, out / f"{clip}.json")
        print(f"  nps-test {clip} ({vid.name}): {no} objs, {nb} boxes")


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
                    choices=["ardmav-train", "ardmav-train-tiled", "ardmav-gt", "nps-gt",
                             "combined-tiled", "combined-gt", "black-paste", "all"])
    ap.add_argument("--stride-train", type=int, default=4)
    ap.add_argument("--stride-val", type=int, default=10)
    ap.add_argument("--min-side", type=int, default=12)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--n-tiles", type=int, default=5000)
    a = ap.parse_args()
    if a.task in ("ardmav-train", "all"):
        build_ardmav_train(a.stride_train, a.stride_val, a.min_side)
    if a.task == "ardmav-train-tiled":
        build_ardmav_train_tiled(a.stride_train, a.stride_val, a.min_side, tile=a.tile)
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
