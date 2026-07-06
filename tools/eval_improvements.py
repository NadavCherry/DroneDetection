#!/usr/bin/env python3
"""Head-to-head: baseline appearance vs NWD-retrained appearance vs RGB+motion
fusion, on the held-out test clips of each dataset. Reports BOTH metrics the user
cares about, kept separate:
  * identification  -> per-frame center-distance AP / recall (tau=12)
  * tracking        -> full-pipeline coverage / false tracks

Run after the NWD and fusion models finish training. Detectors that need the
4-channel fusion input are stateful (motion buffer) and run frame-sequentially.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.methods.base import run_method
from dronedet.methods.yolo import YoloSahi
from dronedet.methods.fusion import FusionDetector
from dronedet.track import run_tracker_file
import tools.eval_external as EX
import tools.eval_tracks as ET

BASE_W = "work/runs/combined-m-p2-640/weights/best.pt"
NWD_W = "work/runs/combined-m-p2-640-nwd/weights/best.pt"
FUS_W = "work/runs/combined-fusion-s-p2-2/weights/best.pt"
FUS_M_W = "work/runs/combined-fusion-m-p2-2/weights/best.pt"

CLIPS = [
    ("phantom16", "ARD-MAV (moving)", "work/ext_datasets/gt_test/ardmav/phantom16.json"),
    ("Clip_19", "NPS (moving)", "work/ext_datasets/gt_test/nps/Clip_19.json"),
    ("10_06", "user black (near-static)", "work/ext_datasets/gt_test/user/10_06.json"),
]

DETECTORS = {
    "baseline": lambda: YoloSahi("app", tile=640, weights=BASE_W, drone_classes=None, conf=0.02),
    "nwd": lambda: YoloSahi("app", tile=640, weights=NWD_W, drone_classes=None, conf=0.02),
    "fusion": lambda: FusionDetector("fusion", weights=FUS_W, tile=640, conf=0.02),
    "fusion_m": lambda: FusionDetector("fusion_m", weights=FUS_M_W, tile=640, conf=0.02),
}


def score_det(gt_path, det_path, tau=12.0):
    with tempfile.TemporaryDirectory() as gd, tempfile.TemporaryDirectory() as dd:
        shutil.copy(gt_path, Path(gd) / "c.json")
        shutil.copy(det_path, Path(dd) / "c.json")
        agg, _ = EX.score_dir(gd, dd, tau, min_score=None)
    return agg


def score_track(gt_path, tracks_path):
    r = ET.score(gt_path, tracks_path)
    obj = next((v for v in r.values() if isinstance(v, dict) and "coverage" in v), {})
    return obj.get("coverage"), r.get("false_tracks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    ap.add_argument("--clips", nargs="+", default=[c[0] for c in CLIPS])
    ap.add_argument("--out", default="work/improve")
    ap.add_argument("--stop", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "dets").mkdir(parents=True, exist_ok=True)
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    clips = [c for c in CLIPS if c[0] in a.clips]

    rpath = out / "RESULTS.json"
    table = json.loads(rpath.read_text()) if rpath.exists() else {}   # merge with prior runs
    for det_name in a.detectors:
        table[det_name] = {}
        for clip, regime, gt_path in clips:
            video = json.loads(Path(gt_path).read_text())["video"]
            if not Path(video).exists():
                print(f"!! {clip}: missing video {video}"); continue
            method = DETECTORS[det_name]()
            print(f"\n=== [{det_name}] {clip} ({regime}) over {Path(video).name} ===", flush=True)
            ds = run_method(video, method, stop=a.stop, stab_mode="affine")
            dpath = out / "dets" / f"{det_name}__{clip}.json"
            ds.save(str(dpath))
            method.close() if hasattr(method, "close") else None

            det = score_det(gt_path, str(dpath))
            tpath = out / "tracks" / f"{det_name}__{clip}.json"
            run_tracker_file(video, str(dpath), str(tpath), min_score=0.2)
            cov, false = score_track(gt_path, str(tpath))
            table[det_name][clip] = {
                "regime": regime, "det_AP": round(det["ap"], 3), "det_R": round(det["R"], 3),
                "det_medErr": round(det["med_err"], 2), "fps": ds.meta.get("fps_end_to_end"),
                "track_cov": cov, "false_tracks": false,
            }
            print(f"  det AP={det['ap']:.3f} R={det['R']:.3f} | track cov={cov} false={false}", flush=True)

    rpath.write_text(json.dumps(table, indent=1))
    # print comparison tables over ALL detectors present (merged)
    dets = [d for d in ("baseline", "nwd", "fusion", "fusion_m") if d in table] or list(table)
    print("\n\n#### IDENTIFICATION (per-frame center-distance AP / recall)")
    print("| clip | " + " | ".join(dets) + " |")
    print("|---|" + "---|" * len(dets))
    for clip, regime, _ in clips:
        row = [f"{table[d].get(clip,{}).get('det_AP','-')}/{table[d].get(clip,{}).get('det_R','-')}"
               for d in dets]
        print(f"| {clip} ({regime}) | " + " | ".join(row) + " |")
    print("\n#### TRACKING (coverage / false tracks)")
    print("| clip | " + " | ".join(dets) + " |")
    print("|---|" + "---|" * len(dets))
    for clip, regime, _ in clips:
        row = [f"{table[d].get(clip,{}).get('track_cov','-')}/{table[d].get(clip,{}).get('false_tracks','-')}"
               for d in dets]
        print(f"| {clip} ({regime}) | " + " | ".join(row) + " |")
    print(f"\nsaved -> {rpath}")


if __name__ == "__main__":
    main()
