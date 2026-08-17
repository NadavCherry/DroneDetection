#!/usr/bin/env python3
"""Compare methods on ONE test video, with a within-sequence interval.

`tools/compare.py` resamples SEQUENCES, which is the right unit and needs several of them.
The project's own task has one test flight (10_06), so that tool cannot produce an interval
here at all. This one resamples contiguous 30-frame BLOCKS of the single sequence instead.

Read `benchmarks/block_bootstrap` before quoting anything this prints. The short version:
it answers *is this difference stable across the segments of this flight*, and it does NOT
answer *does this difference generalise to another flight*. Two videos cannot answer the
second question and no resampling scheme invents the missing evidence.

    python tools/compare_single_video.py \
        --gt work/ext_datasets/gt/local/10_06.json \
        --dets temporal=work/det/local/temporal_local_ab-s0 \
               singleframe=work/det/local/singleframe_local_ab-s0 \
               yolomg=work/det/local/yolomg_local_seed0 \
        --baseline singleframe --out work/reports/local_10_06.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.block_bootstrap import (blocks_from_detections,  # noqa: E402
                                        paired_block_bootstrap)
from benchmarks.protocol import BY_KEY as PROTOCOLS  # noqa: E402
from dronedet import metrics as M  # noqa: E402
from dronedet.console import use_utf8_stdio  # noqa: E402
from dronedet.detections import DetectionSet  # noqa: E402
from dronedet.gt import GroundTruth  # noqa: E402


def evaluate_one(gt_path: Path, det_path: Path, *, protocol, tau: float,
                 targets: set[str] | None):
    """-> (gt_per_frame, dets_per_frame, distractor_hits, n_gt, n_frames)."""
    gt = GroundTruth.load(gt_path)
    ds = DetectionSet.load(det_path)
    rule = "iou" if protocol.matcher == "iou" else "centre"
    ev = M.evaluate(gt, ds, rule=rule, tau=tau,
                    iou_thr=protocol.iou_threshold or 0.5, targets=targets)

    gt_per_frame: dict[int, int] = {}
    for name, o in gt.objects.items():
        if o.ignore:
            continue
        if targets is not None and name not in targets:
            continue
        for f in o.frames:
            gt_per_frame[int(f)] = gt_per_frame.get(int(f), 0) + 1

    det_per_frame: dict[int, list] = {}
    distractors: dict[str, int] = {}
    for r in ev.records:
        if r.outcome in ("tp", "fp"):
            det_per_frame.setdefault(r.frame, []).append((float(r.score),
                                                          r.outcome == "tp"))
        elif r.outcome == "distractor":
            # A detection that landed on a labelled non-target -- here, a bird. This is
            # the only corpus in the project where that number exists, because it is the
            # only one whose annotations mark the distractors.
            distractors[r.obj or "?"] = distractors.get(r.obj or "?", 0) + 1
    return gt_per_frame, det_per_frame, distractors, ev.n_gt, ev.n_frames


def main() -> int:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--dets", required=True, nargs="+",
                    help="name=dir pairs; the dir holds <gt_stem>.json")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--protocol", default="specklock-centre", choices=sorted(PROTOCOLS))
    ap.add_argument("--tau", type=float, default=12.0)
    ap.add_argument("--block-frames", type=int, default=30,
                    help="1 s at 30 fps. Long enough to contain the correlation between "
                         "neighbouring frames, short enough to leave blocks to resample.")
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--converged-floor", type=float, default=0.05,
                    help="An arm scoring below this is treated as NOT CONVERGED rather "
                         "than as a fair loss. Beating a model that failed to train is "
                         "not a result, and publishing it as one is a retraction.")
    a = ap.parse_args()

    protocol = PROTOCOLS[a.protocol]
    targets = set(a.targets) if a.targets else None
    stem = a.gt.stem

    arms = {}
    for spec in a.dets:
        if "=" not in spec:
            raise SystemExit(f"--dets wants name=dir, got {spec!r}")
        name, d = spec.split("=", 1)
        p = Path(d) / f"{stem}.json"
        if not p.exists():
            raise SystemExit(f"{name}: no detections at {p}")
        arms[name] = evaluate_one(a.gt, p, protocol=protocol, tau=a.tau, targets=targets)

    if a.baseline not in arms:
        raise SystemExit(f"baseline {a.baseline!r} not among {sorted(arms)}")

    gt_frames, _, _, n_gt, n_frames = arms[a.baseline]
    blocks = {n: blocks_from_detections(gt_frames, d, block_frames=a.block_frames,
                                        n_frames=n_frames)
              for n, (_, d, _, _, _) in arms.items()}
    n_blocks = len(blocks[a.baseline])

    L = [f"# {stem} — one held-out flight",
         "",
         f"Protocol `{a.protocol}` ({'centre distance' if protocol.matcher != 'iou' else 'IoU'}"
         f"{f', tau={a.tau:g} px' if protocol.matcher != 'iou' else ''}), "
         f"{n_gt:,} labelled instances over {n_frames:,} frames, "
         f"resampled as **{n_blocks} blocks of {a.block_frames} frames**.",
         "",
         "> **This interval is WITHIN ONE SEQUENCE.** It measures whether a difference "
         "holds across the segments of this single flight. It is *not* evidence that the "
         "difference generalises to another flight — two videos cannot support that claim, "
         "and resampling one of them harder does not change what was measured.",
         "",
         "| method | AP | Δ vs baseline | 95% CI on Δ | p | verdict |",
         "|---|---|---|---|---|---|"]

    from benchmarks.block_bootstrap import _ap
    not_converged: list[str] = []
    base_ap = _ap(range(n_blocks), blocks[a.baseline])
    L.append(f"| **{a.baseline}** (baseline) | {base_ap:.3f} | — | — | — | — |")

    for name in sorted(k for k in arms if k != a.baseline):
        r = paired_block_bootstrap(blocks[name], blocks[a.baseline],
                                   block_frames=a.block_frames,
                                   n_resamples=a.n_resamples, seed=a.seed)
        if r.statistic_a < a.converged_floor:
            # Near-zero AP is what a model that never trained looks like, not what a
            # method that lost looks like. Drawing that distinction here rather than
            # leaving it to the reader is the difference between a result and a
            # retraction: YOLOMG scored 0.005-0.010 on this corpus while its OWN
            # validation mAP50 never exceeded 0.0025 over 100 epochs and three seeds.
            verdict = "**DID NOT CONVERGE — not a fair comparison**"
            not_converged.append(name)
        elif r.significant:
            verdict = "**better**" if r.observed > 0 else "**worse**"
        else:
            verdict = "no difference"
        L.append(f"| {name} | {r.statistic_a:.3f} | {r.observed:+.3f} | "
                 f"[{r.lo:+.3f}, {r.hi:+.3f}] | {r.p_value:.4f} | {verdict} |")

    if not_converged:
        L += ["",
              f"> ⚠ **{', '.join(not_converged)} scored below {a.converged_floor:g} AP and is "
              f"treated as NOT CONVERGED.** A near-zero AP is what a model that never "
              f"trained looks like, not what a method that lost looks like. Do not quote a "
              f"margin over it as a win: check its own training curve first, and if it also "
              f"failed on its own validation set then this corpus says something about "
              f"trainability, not about the method.",
              ""]

    # False alarms on labelled distractors -- the question neither public benchmark can
    # answer, because neither labels its birds.
    any_distractor = any(d for _, _, d, _, _ in arms.values())
    L += ["", "### False alarms on labelled distractors", ""]
    if not any_distractor:
        L.append("_No detection landed on a labelled distractor in any arm. In this GT the "
                 "birds are `ignore=True`, so a hit on one is recorded here rather than "
                 "silently counted as a false positive against the background._")
    else:
        L += ["| method | " + " | ".join(sorted({k for _, _, d, _, _ in arms.values()
                                                 for k in d})) + " | total |",
              "|---" * (2 + len({k for _, _, d, _, _ in arms.values() for k in d})) + "|"]
        keys = sorted({k for _, _, d, _, _ in arms.values() for k in d})
        for name in sorted(arms):
            d = arms[name][2]
            L.append(f"| {name} | " + " | ".join(str(d.get(k, 0)) for k in keys)
                     + f" | {sum(d.values())} |")

    out = "\n".join(L) + "\n"
    print(out)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
