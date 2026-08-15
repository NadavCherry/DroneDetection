"""Run a trained YOLOMG over test videos and write OUR detection JSONs.

This is the join that makes the comparison a comparison. `tools/evaluate.py` deliberately
consumes detection JSONs rather than weights, precisely so a rival's model can be scored
under our protocol without its training stack touching our code -- see that file's header.
This script is the other half of that contract for YOLOMG.

RUN IT IN THE COMPETITOR'S ENVIRONMENT, NOT OURS
------------------------------------------------
It imports from `third_party/YOLOMG` (a YOLOv5 v6-era tree, numpy < 2) and writes plain
JSON. It must NOT import `dronedet`, because the two environments do not coexist: the
detection schema is six numbers and a label, so it is reproduced here as a literal rather
than imported. `dronedet/tests/test_sota_infer_yolomg.py` pins the two against each other
so this copy cannot drift from `Detection.as_list`.

WHAT IS SCORED, AND WHY EVERY FRAME
-----------------------------------
The masks are rebuilt on the fly at the same dt the model trained on, and EVERY frame of
each test video is scored -- not the strided subset used for training. Scoring the strided
subset would quietly change the denominator between the two arms and make the AP
incomparable to both our own number and the published one.

Frames within `dt` of either end cannot have a mask; they are emitted as empty detections
rather than skipped, because `tools/evaluate.py` scores a missing sequence as a total miss
and a skipped frame would otherwise remove its ground truth from the denominator. Losing
two frames at each end is a real, tiny cost to the competitor, and it is theirs by
construction: a two-sided mask cannot exist at a video boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
MG = REPO / "third_party" / "YOLOMG"
sys.path.insert(0, str(MG))
sys.path.insert(0, str(REPO))

from tools.sota.motion_mask import YOLOMG_MASK32_DT, fd5_mask  # noqa: E402
from tools.video_paths import resolve_all  # noqa: E402


def _letterbox(im, new_shape=1280, stride=32):
    """YOLOv5's letterbox, reduced to the inference path (no scaleup beyond 1.0 changes,
    no rect padding surprises). Returns the padded image and the (ratio, dw, dh) needed to
    map boxes back to original pixels."""
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, left, top


def _to_tensor(im, device):
    x = im.transpose((2, 0, 1))[::-1]                   # HWC BGR -> CHW RGB
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(0).to(device)


def run_video(model, video: Path, out_json: Path, imgsz: int, conf: float,
              iou: float, dt: int, device, method: str):
    from utils.general import non_max_suppression, scale_coords

    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    span = 2 * dt + 1
    buf: list = []
    frames: dict[str, list] = {}
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
        mid = i - dt
        rgb, mask_u8 = buf[dt], np.clip(fd5_mask(buf[0], buf[dt], buf[2 * dt]),
                                        0, 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

        lb_rgb, r, padx, pady = _letterbox(rgb, imgsz)
        lb_msk, *_ = _letterbox(mask_bgr, imgsz)
        with torch.no_grad():
            pred = model(_to_tensor(lb_rgb, device), _to_tensor(lb_msk, device))[0]
            pred = non_max_suppression(pred, conf, iou, max_det=300)[0]

        dets = []
        if pred is not None and len(pred):
            p = pred.clone()
            p[:, :4] = scale_coords(lb_rgb.shape[:2], p[:, :4], rgb.shape[:2]).round()
            for *xyxy, sc, _cls in p.cpu().numpy():
                dets.append([round(float(xyxy[0]), 2), round(float(xyxy[1]), 2),
                             round(float(xyxy[2]), 2), round(float(xyxy[3]), 2),
                             round(float(sc), 4), "drone"])
        frames[str(mid)] = dets
    cap.release()

    # Boundary frames: emitted empty, never omitted. See the module docstring.
    for f in list(range(0, min(dt, n_frames))) + \
             list(range(max(0, n_frames - dt), n_frames)):
        frames.setdefault(str(f), [])

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "video": video.name, "method": method,
        "meta": {"n_frames": n_frames, "imgsz": imgsz, "conf_floor": conf,
                 "nms_iou": iou, "mask_dt": dt, "comparator": "YOLOMG",
                 "scored_every_frame": True},
        "frames": frames,
    }), encoding="utf-8")
    return len(frames), sum(len(v) for v in frames.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gt-dir", required=True,
                    help="directory of per-sequence GT JSONs; NAMES the sequences. Videos "
                         "are resolved from --video-root by the same code our own arm "
                         "uses, and the output JSON is named after the GT stem -- NPS "
                         "ships Clip_41.mov against a GT called Clip_041, and naming the "
                         "output after the video would score every sequence as a miss.")
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True, help="output directory for detection JSONs")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--dt", type=int, default=YOLOMG_MASK32_DT)
    ap.add_argument("--method", default="yolomg")
    # 0.001 matches YOLOMG's own val.py --conf-thres and our evaluation floor. A higher
    # floor truncates the precision-recall curve and silently lowers the competitor's AP.
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    a = ap.parse_args()

    from models.experimental import attempt_load

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = attempt_load(a.weights, map_location=device).eval()

    out = Path(a.out)
    pairs, missing = resolve_all(Path(a.video_root), Path(a.gt_dir))
    if missing:
        # Not a warning. evaluate.py scores an unmatched sequence as a TOTAL MISS with its
        # full ground truth still charged, so a filename problem here would deflate the
        # COMPETITOR's AP and publish a wrong number in the direction that costs a
        # retraction. Our arm aborts on this; so does theirs.
        print(f"ABORT: {len(missing)} of {len(pairs) + len(missing)} GT sequences have no "
              f"video under {a.video_root}: {missing[:5]}", file=sys.stderr)
        return 2
    print(f"{len(pairs)} sequences to score", flush=True)
    for stem, vp in pairs:
        nf, nd = run_video(model, vp, out / f"{stem}.json", a.imgsz, a.conf,
                           a.iou, a.dt, device, a.method)
        print(f"  {stem} ({vp.name}): {nf} frames, {nd} detections", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
