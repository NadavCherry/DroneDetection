#!/usr/bin/env python3
"""Score a model on a benchmark and write a Scorecard.

Deliberately takes **detection JSONs**, not weights. Producing detections is a separate,
GPU-bound step (`python -m dronedet detect`, `tools/run_max.py`, or a competitor's own
inference script); scoring them is cheap, deterministic and needs nothing but numpy. That
split is what lets a *rival's* released weights be scored under our protocol without
their training stack ever touching this code — which is the answer to "you have never run
the baseline".

    # 1. produce detections however you like, one JSON per sequence
    python -m dronedet detect --video data/external/ard_mav/ARD-MAV/videos/phantom05.mp4 \
        --method moe3-stacked --out work/det/ardmav/phantom05.json

    # 2. score them into a scorecard
    python tools/evaluate.py --dataset ardmav --model trueextent-nwd \
        --gt work/prepared/ardmav/gt --dets work/det/ardmav --out work/scorecards/ours.json

    # 3. compare (tools/compare.py)

Sequences are paired by **filename stem**, so `gt/phantom05.json` scores against
`dets/phantom05.json`. A GT file with no matching detection file is scored as a total
miss rather than skipped -- skipping it would silently improve the number, which is the
kind of quiet favour that makes a benchmark result untrustworthy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.catalog import DATASETS  # noqa: E402
from benchmarks.protocol import BY_KEY as PROTOCOLS  # noqa: E402
from benchmarks.scorecard import Scorecard, SequenceResult, pooled_ap  # noqa: E402
from dronedet import metrics as M  # noqa: E402
from dronedet.detections import DetectionSet  # noqa: E402
from dronedet.gt import GroundTruth  # noqa: E402


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _sha256(path: Path) -> str:
    if not path or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _outcome_name(record: M.Record) -> str:
    """Map a metrics Record onto the scorecard's compact outcome string."""
    if record.outcome == "distractor":
        return f"distractor:{record.obj}"
    return record.outcome


def score_sequence(gt_path: Path, det_path: Path | None, *, protocol, tau: float,
                   targets: set[str] | None, conditions: tuple[str, ...]) -> SequenceResult:
    """One sequence -> one SequenceResult, losslessly enough to re-argue later."""
    gt = GroundTruth.load(gt_path)

    if det_path is None or not det_path.exists():
        # Total miss, recorded explicitly. n_gt still counts, so recall is charged.
        n_gt = sum(1 for o in gt.objects.values() if not o.ignore for _ in o.frames)
        return SequenceResult(sequence=gt_path.stem, n_gt=n_gt, n_frames=0,
                              conditions=list(conditions), detections=[],
                              distractor_instances={})

    ds = DetectionSet.load(det_path)
    rule = "iou" if protocol.matcher == "iou" else "centre"
    ev = M.evaluate(gt, ds, rule=rule, tau=tau,
                    iou_thr=protocol.iou_threshold or 0.5, targets=targets)

    return SequenceResult(
        sequence=gt_path.stem,
        n_gt=ev.n_gt,
        n_frames=ev.n_frames,
        conditions=list(conditions),
        detections=[(float(r.score), _outcome_name(r)) for r in ev.records],
        distractor_instances=dict(ev.distractor_instances_by_object),
        target_px_median=_median_target_px(gt, targets),
    )


def _median_target_px(gt: GroundTruth, targets: set[str] | None) -> float | None:
    import math
    sides = []
    for name, o in gt.objects.items():
        is_target = (name in targets) if targets is not None else (not o.ignore)
        if not is_target:
            continue
        sides.extend(math.sqrt(max(b[2] * b[3], 0.0)) for b in o.frames.values())
    if not sides:
        return None
    sides.sort()
    return float(sides[len(sides) // 2])


def build_scorecard(dataset_key: str, model: str, gt_dir: Path, det_dir: Path, *,
                    protocol_key: str | None = None, split: str = "",
                    tau: float = 12.0, targets: set[str] | None = None,
                    conditions_map: dict[str, tuple[str, ...]] | None = None,
                    only: set[str] | None = None,
                    weights: Path | None = None, seed: int | None = None) -> Scorecard:
    ds_entry = DATASETS.get(dataset_key)
    if protocol_key is None:
        if ds_entry and ds_entry.official_protocol is not None:
            protocol_key = next((k for k, v in PROTOCOLS.items()
                                 if v is ds_entry.official_protocol), "specklock-centre")
        else:
            protocol_key = "specklock-centre"
    protocol = PROTOCOLS[protocol_key]

    if not split:
        split = protocol.split or "unspecified"

    gt_files = sorted(p for p in gt_dir.glob("*.json"))
    if only is not None:
        gt_files = [p for p in gt_files if p.stem in only]
    if not gt_files:
        raise SystemExit(f"no ground-truth JSONs under {gt_dir}"
                         + (f" matching {sorted(only)}" if only else ""))

    conditions_map = conditions_map or {}
    seqs = []
    missing = []
    for gp in gt_files:
        dp = det_dir / gp.name
        if not dp.exists():
            missing.append(gp.stem)
        seqs.append(score_sequence(gp, dp if dp.exists() else None, protocol=protocol,
                                   tau=tau, targets=targets,
                                   conditions=conditions_map.get(gp.stem, ())))
    if missing:
        print(f"warning: {len(missing)} sequence(s) had no detections and are scored as "
              f"total misses (not skipped): {', '.join(missing[:8])}"
              + (" ..." if len(missing) > 8 else ""), file=sys.stderr)

    return Scorecard(
        model=model, dataset_key=dataset_key, protocol_key=protocol_key, split=split,
        sequences=seqs,
        git_sha=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain")),
        weights_sha256=_sha256(weights) if weights else "",
        command=" ".join(sys.argv),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seed=seed,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help=f"one of {sorted(DATASETS)}")
    ap.add_argument("--model", required=True, help="name for this model in the tables")
    ap.add_argument("--gt", required=True, type=Path, help="directory of per-sequence GT JSONs")
    ap.add_argument("--dets", required=True, type=Path, help="directory of per-sequence detection JSONs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS),
                    help="default: the dataset's official protocol")
    ap.add_argument("--split", default="", help="name of the split actually evaluated — "
                                                "say 'single-clip-x' if that is what it is")
    ap.add_argument("--tau", type=float, default=12.0)
    ap.add_argument("--targets", nargs="*", help="GT object names counting as positives; "
                                                 "everything else becomes a distractor")
    ap.add_argument("--official-split", action="store_true",
                    help="restrict to the dataset's published test sequences")
    ap.add_argument("--conditions", type=Path,
                    help="JSON mapping sequence -> [condition, ...]; without it no "
                         "night/rain/fog table can be produced")
    ap.add_argument("--weights", type=Path, help="hashed into the scorecard for provenance")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)

    if a.dataset not in DATASETS:
        raise SystemExit(f"unknown dataset '{a.dataset}'; known: {sorted(DATASETS)}")
    entry = DATASETS[a.dataset]

    only = None
    split = a.split
    if a.official_split:
        if not entry.official_test:
            raise SystemExit(f"{a.dataset} publishes no official test split — name the split "
                             "you are using with --split instead of implying a standard one")
        only = set(entry.official_test)
        split = split or "official-test"

    conditions_map = {}
    if a.conditions:
        conditions_map = {k: tuple(v) for k, v in json.loads(a.conditions.read_text()).items()}

    card = build_scorecard(
        a.dataset, a.model, a.gt, a.dets, protocol_key=a.protocol, split=split,
        tau=a.tau, targets=set(a.targets) if a.targets else None,
        conditions_map=conditions_map, only=only, weights=a.weights, seed=a.seed)
    card.save(a.out)

    ap_value = pooled_ap(card.sequences)
    print(f"{card.model} on {card.dataset_key} [{card.split}] under {card.protocol_key}")
    print(f"  {card.n_sequences} sequences, {card.n_gt:,} instances, {card.n_frames:,} frames")
    print(f"  pooled AP = {ap_value:.3f}")
    hits, total = card.distractor_hits(0.5, ("bird", "plane", "airplane", "helicopter"))
    if total:
        print(f"  confuser hits at 0.5: {hits} / {total:,} instances")
    if not conditions_map:
        print("  no condition labels supplied — the by-condition table will be empty",
              file=sys.stderr)
    if card.git_dirty:
        print("  warning: working tree is dirty; this scorecard is not reproducible from a "
              "commit", file=sys.stderr)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
