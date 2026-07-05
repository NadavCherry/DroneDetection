#!/usr/bin/env python3
"""Unified MAX pipeline (PC-MAX, accuracy-first).

  fused detection  ->  affine-aware Kalman tracker (+ re-acquisition)  ->
  track-level drone/clutter classification  ->  track-integrated detections

Profiles:
  v1  = mc-hybrid: combined-m-p2 appearance (SAHI multi-scale) + ego-motion-compensated
        motion, adaptively fused.  (uses only pieces built so far)
  v2  = ensemble: v1 + a temporal-stack expert, fused.  (adds motion-in-input)

Optionally scores tracked coverage against a GT json (--gt). Detection runs with
--stab affine so the tracker gets the full per-frame transform.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronedet.methods.base import run_method
from dronedet.track import run_tracker_file
from dronedet.trackclass import classify_tracks, mean_ego_motion


def prescan_ego(video, sample=200):
    """Median frame-to-frame camera translation over the first `sample` frames."""
    import numpy as np
    from dronedet.stabilize import Stabilizer
    from dronedet.video import frames
    stab = Stabilizer("affine")
    prev, shifts = None, []
    for _idx, fr in frames(video, stop=sample):
        m = stab.update(fr)
        if prev is not None:
            A = np.linalg.inv(np.vstack([m, [0, 0, 1]])) @ np.vstack([prev, [0, 0, 1]])
            shifts.append(float(np.hypot(A[0, 2], A[1, 2])))
        prev = m
    return float(np.median(shifts)) if shifts else 0.0


def build_detector(profile, weights, near_static, temporal_weights=None, tile=640):
    """Regime-adaptive: near-static -> motion+appearance (mc-hybrid), the black
    drone needs colour-blind motion; moving camera -> appearance-only SAHI, which
    is cleaner and faster (motion is parallax clutter there)."""
    from dronedet.methods.mc_hybrid import MCHybrid
    from dronedet.methods.yolo import YoloSahi
    if profile == "v2":
        from dronedet.methods.max_ensemble import MaxEnsemble
        return MaxEnsemble("max-v2", weights=weights, temporal_weights=temporal_weights,
                           tile=tile, near_static=near_static)
    if near_static:
        return MCHybrid("max-v1-mc", weights=weights, tile=tile, drone_classes=None)
    return YoloSahi("max-v1-app", tile=tile, weights=weights, drone_classes=None, conf=0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--profile", choices=["v1", "v2"], default="v1")
    ap.add_argument("--weights", required=True, help="combined appearance model")
    ap.add_argument("--temporal-weights", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--min-score", type=float, default=0.2)
    ap.add_argument("--gt", default=None, help="score tracked coverage against this GT json")
    ap.add_argument("--gt-lt", type=int, default=None, help="restrict GT eval to frames < this")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 0. regime pre-scan (cheap: stabilizer only, no detector)
    ego = prescan_ego(a.video)
    near_static = ego < 1.5
    print(f"[max-{a.profile}] ego-motion {ego:.2f} px/frame -> "
          f"{'NEAR-STATIC (motion+appearance)' if near_static else 'MOVING (appearance-only)'}", flush=True)

    # 1. fused detection (affine transforms stored for the tracker)
    method = build_detector(a.profile, a.weights, near_static, a.temporal_weights)
    ds = run_method(a.video, method, stop=a.stop, stab_mode="affine")
    ds.save(str(out / "dets.json"))
    print(f"[max-{a.profile}] {ds.meta.get('fps_end_to_end')} fps, "
          f"{sum(len(v) for v in ds.frames.values())} dets", flush=True)

    # 2. affine-aware tracker + re-acquisition
    run_tracker_file(a.video, str(out / "dets.json"), str(out / "tracks.json"),
                     min_score=a.min_score)

    # 3. track-level drone/clutter classification -> keep drone/near tracks.
    #    Regime-adaptive: trust colour-blind motion evidence only on a near-static
    #    camera; on a moving camera require appearance confirmation (motion is
    #    parallax clutter there, appearance is strong & clean).
    dets_payload = json.loads((out / "dets.json").read_text())
    raw = json.loads((out / "tracks.json").read_text())
    cls = classify_tracks(raw, dets_payload, allow_motion=near_static)
    keep = [tr for tr in raw["tracks"] if cls[tr["id"]]["cls"] != "other"]
    (out / "tracks_drone.json").write_text(json.dumps(
        {**raw, "tracks": keep, "classification": {str(k): v for k, v in cls.items()}}))
    print(f"tracks: {len(raw['tracks'])} confirmed -> {len(keep)} drone/near after classification")

    # 4. optional tracked-coverage eval
    if a.gt:
        gt = json.loads(Path(a.gt).read_text())
        if a.gt_lt is not None:
            for o in gt["objects"].values():
                o["frames"] = {f: b for f, b in o["frames"].items() if int(f) < a.gt_lt}
        tmp_gt = out / "_gt.json"
        tmp_gt.write_text(json.dumps(gt))
        import tools.eval_tracks as ET
        for label, tks in (("all confirmed", out / "tracks.json"),
                           ("drone-classified", out / "tracks_drone.json")):
            r = ET.score(str(tmp_gt), str(tks))
            obj = next((v for v in r.values() if isinstance(v, dict) and "coverage" in v), {})
            print(f"  [{label}] coverage={obj.get('coverage')} id_sw={obj.get('id_switches')} "
                  f"med_err={obj.get('med_err_px')} false_tracks={r['false_tracks']}")


if __name__ == "__main__":
    main()
