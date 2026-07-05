#!/usr/bin/env python3
"""Render GT-vs-detection comparison panels for the external-dataset report.

Tiny drones (3-30 px) are invisible in a full-frame thumbnail, so each panel is a
zoomed crop centered on the GT drone, with GT (green) and our detection (red)
drawn. A context row shows the whole frame with a marker + inset so the reader
sees just how small the target is. Output PNGs land in docs/media/external/.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GREEN = (80, 220, 80)
RED = (60, 60, 235)
OUT = Path("docs/media/external")


def load_json(p):
    return json.loads(Path(p).read_text())


def grab_frames(video, idxs):
    want = set(idxs)
    got = {}
    cap = cv2.VideoCapture(video)
    i = 0
    last = max(want)
    while i <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = fr
        i += 1
    cap.release()
    return got


def gt_boxes(gt, f):
    out = []
    for name, o in gt["objects"].items():
        if o.get("ignore"):
            continue
        b = o["frames"].get(str(f))
        if b:
            cx, cy, w, h = b
            out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def det_boxes(det, f, conf):
    return [(d[0], d[1], d[2], d[3]) for d in det["frames"].get(str(f), []) if d[4] >= conf]


def crop_panel(frame, cx, cy, half, zoom, gts, dets, W, H):
    x0 = int(np.clip(cx - half, 0, max(W - 2 * half, 0)))
    y0 = int(np.clip(cy - half, 0, max(H - 2 * half, 0)))
    crop = frame[y0:y0 + 2 * half, x0:x0 + 2 * half].copy()
    crop = cv2.resize(crop, (2 * half * zoom, 2 * half * zoom), interpolation=cv2.INTER_NEAREST)
    for (bx1, by1, bx2, by2) in gts:
        cv2.rectangle(crop, (int((bx1 - x0) * zoom), int((by1 - y0) * zoom)),
                      (int((bx2 - x0) * zoom), int((by2 - y0) * zoom)), GREEN, 2)
    for (bx1, by1, bx2, by2) in dets:
        cv2.rectangle(crop, (int((bx1 - x0) * zoom), int((by1 - y0) * zoom)),
                      (int((bx2 - x0) * zoom), int((by2 - y0) * zoom)), RED, 1)
    return crop


def make_strip(video, detp, gtp, frames, conf, title, half=45, zoom=4):
    det, gt = load_json(detp), load_json(gtp)
    grabbed = grab_frames(video, frames)
    panels = []
    for f in frames:
        if f not in grabbed:
            continue
        fr = grabbed[f]
        H, W = fr.shape[:2]
        gts = gt_boxes(gt, f)
        dets = det_boxes(det, f, conf)
        if gts:
            cx = (gts[0][0] + gts[0][2]) / 2
            cy = (gts[0][1] + gts[0][3]) / 2
        elif dets:
            cx = (dets[0][0] + dets[0][2]) / 2
            cy = (dets[0][1] + dets[0][3]) / 2
        else:
            cx, cy = W / 2, H / 2
        p = crop_panel(fr, cx, cy, half, zoom, gts, dets, W, H)
        cv2.putText(p, f"f{f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        panels.append(p)
    if not panels:
        return None
    strip = np.hstack(panels)
    bar = np.zeros((34, strip.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, "GT", (strip.shape[1] - 150, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)
    cv2.putText(bar, "ours", (strip.shape[1] - 90, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2)
    return np.vstack([bar, strip])


def context_figure(video, gtp, frame, half=55, out="context.png"):
    """Whole frame (downscaled) + zoom inset, to show how tiny the target is."""
    gt = load_json(gtp)
    fr = grab_frames(video, [frame])[frame]
    H, W = fr.shape[:2]
    gts = gt_boxes(gt, frame)
    view = fr.copy()
    if gts:
        cx = int((gts[0][0] + gts[0][2]) / 2)
        cy = int((gts[0][1] + gts[0][3]) / 2)
        cv2.rectangle(view, (cx - half, cy - half), (cx + half, cy + half), GREEN, 3)
        inset = fr[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
        inset = cv2.resize(inset, (220, 220), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(inset, (0, 0), (219, 219), GREEN, 3)
        view[0:220, W - 220:W] = inset
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / out), view)
    return OUT / out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON list of panel specs")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in json.loads(Path(a.spec).read_text()):
        if s.get("context"):
            p = context_figure(s["video"], s["gt"], s["frame"], out=s["out"])
            print("wrote", p)
            continue
        strip = make_strip(s["video"], s["det"], s["gt"], s["frames"],
                           s.get("conf", 0.25), s["title"],
                           half=s.get("half", 45), zoom=s.get("zoom", 4))
        if strip is not None:
            cv2.imwrite(str(OUT / s["out"]), strip)
            print("wrote", OUT / s["out"])
