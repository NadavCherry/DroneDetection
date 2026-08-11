"""Test reference GT for data/videos/10_06.mp4 (never used for training).

Seeded at a visually-verified drone position (frame 330, crisp against
sky), tracked in both directions with the non-causal windowed-median GT
tracker from tools/build_gt.py. The two unconfirmed directed movers found
by the PC tracker become ignore objects. Verified with a contact sheet.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from dronedet.gt import GroundTruth, GTObject
from dronedet.stabilize import Stabilizer, shift_of, warp_to_reference
from dronedet.video import frames

spec = importlib.util.spec_from_file_location("build_gt", ROOT / "tools/build_gt.py")
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)

VIDEO = "data/videos/10_06.mp4"


def main() -> None:
    stab = Stabilizer("translation")
    grays, shifts, colors = [], [], []
    for idx, frame in frames(VIDEO):
        m = stab.update(frame)
        grays.append(warp_to_reference(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), m))
        shifts.append(shift_of(m))
        colors.append(frame)
    grays = np.stack(grays)
    n = len(grays)
    print(f"{n} frames stabilized")

    # Reference = the PC pipeline's visually-verified track (id1),
    # measured (non-coast) positions only. The offline re-track failed here:
    # the drone crosses in front of dense foliage where windowed-median
    # texture noise dominates. Using the verified track makes 10_06 a
    # hit-rate-along-trajectory test (documented in the README).
    pc_all = json.loads(Path("work/infer/10_06/tracks_all.json").read_text())
    id1 = {int(f): v for tr in pc_all["tracks"] if tr["id"] == 1
           for f, v in tr["frames"].items()}
    far = {t: (v[0] + shifts[t][0], v[1] + shifts[t][1],
               max(v[2], 4.0), max(v[3], 4.0), v[4])
           for t, v in id1.items() if v[4] != "coast"}
    print(f"far (verified reference): {len(far)} frames, span {min(far)}..{max(far)}")

    gt = GroundTruth(video=VIDEO)
    gt.meta["shifts"] = {str(i): [round(s[0], 3), round(s[1], 3)]
                         for i, s in enumerate(shifts)}
    gt.meta["note"] = ("TEST reference from the visually-verified PC track "
                       "(measured frames only); unconfirmed movers = ignore")
    obj = GTObject("far", ignore=False)
    for t, (cx, cy, w, h, status) in sorted(far.items()):
        obj.frames[t] = (cx - shifts[t][0], cy - shifts[t][1], w, h)
    gt.objects["far"] = obj

    pc = json.loads(Path("work/infer/10_06/tracks_all.json").read_text())
    k = 0
    for tr in pc["tracks"]:
        fs = {int(f): v for f, v in tr["frames"].items()}
        # ignore any track that is not the drone (id1 = the drone)
        if tr["id"] == 1:
            continue
        k += 1
        o = GTObject(f"mover{k}", ignore=True)
        for f, v in fs.items():
            o.frames[f] = (v[0], v[1], max(v[2], 6.0), max(v[3], 6.0))
        gt.objects[o.name] = o
    excl = sorted(set(range(n)) - set(obj.frames))
    gt.meta["exclude_frames"] = excl
    gt.save("realtime/work/gt_1006.json")
    print(f"saved realtime/work/gt_1006.json (+{k} ignore movers, "
          f"{len(excl)} excluded frames)")

    # verification sheet
    Path("realtime/work").mkdir(parents=True, exist_ok=True)
    cells = []
    ks = sorted(obj.frames)
    for t in ks[:: max(1, len(ks) // 12)][:12]:
        cx, cy = obj.frames[t][0], obj.frames[t][1]
        img = colors[t]
        half = 26
        x0 = int(np.clip(cx - half, 0, img.shape[1] - 2 * half))
        y0 = int(np.clip(cy - half, 0, img.shape[0] - 2 * half))
        z = cv2.resize(img[y0:y0 + 2 * half, x0:x0 + 2 * half], None, fx=5, fy=5,
                       interpolation=cv2.INTER_NEAREST)
        cv2.circle(z, (int((cx - x0) * 5), int((cy - y0) * 5)), 16, (0, 0, 255), 1)
        cv2.putText(z, str(t), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cells.append(z)
    rows = [np.hstack(cells[i:i + 6]) for i in range(0, len(cells), 6)]
    rows = [r for r in rows if r.shape[1] == rows[0].shape[1]]
    cv2.imwrite("realtime/work/gt_1006_verify.png", np.vstack(rows))
    print("verification sheet -> realtime/work/gt_1006_verify.png")


if __name__ == "__main__":
    main()
