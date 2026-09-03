#!/usr/bin/env python3
"""Accuracy as a function of target size, for every arm, on one evaluator.

THE QUESTION
------------
Our method loses to YOLOMG on NPS overall (0.487 vs 0.527) and on ARD-MAV overall (0.809
vs 0.834), and beats it heavily on an 8 px local task (0.840 vs 0.604). Those three facts
are only contradictory if "accuracy" is one number. They are consistent -- and far more
useful -- if accuracy depends on target size and the two methods have different curves
that cross somewhere.

This measures that curve instead of asserting it. If a crossover exists, it is a specific,
falsifiable, operationally meaningful claim: below N pixels use this method, above N do
not bother. If it does not exist, the honest conclusion is that we simply lose, and the
right thing to do is say so.

WHAT IS HELD FIXED
------------------
Everything except target size. Same evaluator, same matching rule, same IoU threshold,
same confidence floor, same sequences, same pooling. The arms differ only in which
detections file they read. A curve produced any other way measures the protocol, not the
size dependence -- which is the mistake `tools/protocol_sweep.py` exists to catch.

READING THE OUTPUT HONESTLY
---------------------------
Three things are reported for every cell and none of them are optional:

  n      GT instances in that bin. An AP over 12 boxes is noise. Bins under --min-gt are
         printed but flagged, never silently dropped and never quietly merged.
  seeds  mean +- sample standard deviation over training seeds. One seed is an anecdote;
         the spread between seeds on these datasets is routinely larger than the gap
         between methods, which is exactly the fact a single-seed table conceals.
  CI     95 % bootstrap interval resampling SEQUENCES, not frames. Frames within a
         sequence are massively correlated -- consecutive frames of one flight are nearly
         the same picture -- so a frame-level bootstrap would report an interval several
         times too narrow and manufacture significance that is not there.

    PYTHONPATH=. python tools/size_curve.py \
        --arm ours work/ext_datasets/gt/nps work/det/nps/temporal_nps-s0 \
        --arm yolomg work/ext_datasets/gt/nps work/det/nps/yolomg_nps_seed0 \
        --dataset nps --out work/reports/size_curve
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.fast_bootstrap import (paired_bootstrap_pooled_ap,  # noqa: E402
                                       paired_permutation_pooled_ap)
from dronedet import metrics as M  # noqa: E402
from dronedet.console import use_utf8_stdio  # noqa: E402
from dronedet.detections import DetectionSet  # noqa: E402
from dronedet.gt import GroundTruth  # noqa: E402

BIN_SETS = {"mission": M.MISSION_BINS, "aitod": M.SIZE_BINS}


class _Unit:
    """One sequence's results inside ONE size bin, shaped for the paired tests.

    `benchmarks.fast_bootstrap` wants `.sequence`, `.n_gt` and `.detections` as
    `(score, outcome)`. Reusing that module rather than writing a second bootstrap is
    deliberate: this project already refuses to compare APs computed two different ways,
    and quietly running a *differently implemented* significance test beside the one in
    `tools/make_summary.py` would be the same mistake one level up.
    """

    __slots__ = ("sequence", "n_gt", "detections")

    def __init__(self, sequence, n_gt, detections):
        self.sequence, self.n_gt, self.detections = sequence, n_gt, detections


def _units(per_seq, seqs, bname):
    """Per-sequence units for one bin: in-bin TPs plus every FP, as `ap_by_bins` scores it."""
    out = []
    for s in seqs:
        d = per_seq[s]
        det = [(float(x), "tp") for x in d["tp"][bname]]
        det += [(float(x), "fp") for x in d["fp"]]
        out.append(_Unit(s, d["n_gt"][bname], det))
    return out


def paired_bin_test(per_seq_a, per_seq_b, seqs, bname, n_resamples: int, seed: int = 0):
    """Seed-matched paired bootstrap AND permutation for one size bin.

    Both must agree before anything is called significant, matching
    `tools/make_summary.paired`. The bootstrap alone will call a small-N difference
    significant; requiring the permutation test too makes a thin comparison
    "inconclusive" rather than manufacturing confidence it has not earned.
    """
    ua, ub = _units(per_seq_a, seqs, bname), _units(per_seq_b, seqs, bname)
    if len(seqs) < 2 or not any(u.n_gt for u in ua):
        return None
    r = paired_bootstrap_pooled_ap(ua, ub, n_resamples=n_resamples, seed=seed)
    r["p_perm"] = paired_permutation_pooled_ap(ua, ub, n_resamples=n_resamples, seed=seed)
    r["n_seq"] = len(seqs)
    r["significant"] = (r["lo"] > 0 or r["hi"] < 0) and r["p_perm"] < 0.05
    return r


def _ap(tp_scores: np.ndarray, fp_scores: np.ndarray, n_gt: int) -> float:
    """All-point interpolated AP from score arrays. Identical to
    `dronedet.metrics.average_precision`, but vectorised.

    The readable version -- rebuilding a list of Record objects per bootstrap resample
    and calling the reference implementation -- is correct and unusably slow here: a
    1,000-resample interval over 5 bins is 5,000 sorts of ~200k Python objects, which ran
    for 16 minutes on ONE of nine arms before being cancelled. Same arithmetic, numpy
    arrays instead of objects. `--self-check` asserts the two agree.
    """
    if n_gt == 0:
        return 0.0
    n_tp, n_fp = tp_scores.size, fp_scores.size
    if n_tp + n_fp == 0:
        return 0.0
    scores = np.concatenate((tp_scores, fp_scores))
    is_tp = np.concatenate((np.ones(n_tp), np.zeros(n_fp)))
    # Stable sort so ties resolve in the same order the reference implementation sees
    # (it sorts a list of records by -score, and Python's sort is stable).
    order = np.argsort(-scores, kind="stable")
    is_tp = is_tp[order]
    tp = np.cumsum(is_tp)
    fp = np.cumsum(1.0 - is_tp)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - prev) * precision))


def eval_arm(gt_dir: Path, det_dir: Path, *, rule: str, tau: float, iou: float,
             conf: float, bins, check: bool = True):
    """Evaluate once, then reduce each sequence to the arrays a bootstrap needs.

    -> {sequence: {"tp": {bin: scores}, "fp": scores, "n_gt": {bin: int}}}

    Collapsing to arrays HERE is what makes the resampling cheap: a resample is then a
    concatenate and a sort of floats, with no per-detection Python object touched again.
    """
    per_seq: dict[str, dict] = {}
    for gp in sorted(gt_dir.glob("*.json")):
        dp = det_dir / gp.name
        if not dp.exists():
            continue
        gt = GroundTruth.load(gp)
        ds = DetectionSet.load(dp)
        if conf > 0:
            f = DetectionSet(video=ds.video, method=ds.method)
            for fr, d in ds.frames.items():
                f.frames[fr] = [x for x in d if x.score >= conf]
            ds = f
        ev = M.evaluate(gt, ds, rule=rule, tau=tau, iou_thr=iou)

        fp = np.array([r.score for r in ev.records if r.outcome == "fp"], dtype=float)
        tp_by_bin, n_by_bin = {}, {}
        for bname, lo, hi in bins:
            tp_by_bin[bname] = np.array(
                [r.score for r in ev.records
                 if r.outcome == "tp" and lo <= r.gt_size < hi], dtype=float)
            n_by_bin[bname] = sum(1 for s in ev.gt_sizes if lo <= s < hi)

        ### Prove the fast path against the reference implementation on real data, once
        ### per sequence. `_ap` re-derives arithmetic that dronedet.metrics already owns,
        ### and a vectorised rewrite that is subtly wrong would produce a smooth,
        ### plausible, publishable curve with no symptom at all.
        if check:
            ref = M.ap_by_bins(ev, bins)
            for bname, (ref_ap, ref_n) in ref.items():
                got = _ap(tp_by_bin[bname], fp, n_by_bin[bname])
                if abs(got - ref_ap) > 1e-9 or ref_n != n_by_bin[bname]:
                    raise SystemExit(
                        f"self-check FAILED on {gp.stem}/{bname}: "
                        f"fast {got:.12f} n={n_by_bin[bname]} vs "
                        f"reference {ref_ap:.12f} n={ref_n}")

        per_seq[gp.stem] = {"tp": tp_by_bin, "fp": fp, "n_gt": n_by_bin}
    return per_seq


def pooled_bins(per_seq, seqs, bins) -> dict[str, tuple[float, int]]:
    """Pool the named sequences into one PR curve per bin.

    A false positive has no size, so it is charged to EVERY bin -- otherwise a method
    could look strong on tiny targets by flooding the frame with large spurious boxes
    that no small-target bin ever pays for.
    """
    fp = np.concatenate([per_seq[s]["fp"] for s in seqs]) if seqs else np.empty(0)
    out: dict[str, tuple[float, int]] = {}
    for bname, _, _ in bins:
        n = sum(per_seq[s]["n_gt"][bname] for s in seqs)
        if n == 0:
            continue
        tp = np.concatenate([per_seq[s]["tp"][bname] for s in seqs])
        out[bname] = (_ap(tp, fp, n), n)
    return out


def main() -> int:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs=3, action="append", required=True,
                    metavar=("NAME", "GT_DIR", "DET_DIR"),
                    help="repeatable; NAME may repeat across seeds and is averaged")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--bins", default="mission", choices=tuple(BIN_SETS))
    ap.add_argument("--rule", default="iou", choices=("iou", "centre"))
    ap.add_argument("--tau", type=float, default=12.0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--min-gt", type=int, default=50,
                    help="bins with fewer GT instances are flagged as underpowered")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-self-check", action="store_true",
                    help="skip verifying the vectorised AP against "
                         "dronedet.metrics.ap_by_bins on every sequence")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    bins = BIN_SETS[a.bins]
    by_name: dict[str, list] = defaultdict(list)
    for name, gt_dir, det_dir in a.arm:
        gd, dd = Path(gt_dir), Path(det_dir)
        if not gd.is_dir() or not dd.is_dir():
            print(f"  skip {name}: missing {gd if not gd.is_dir() else dd}")
            continue
        per_seq = eval_arm(gd, dd, rule=a.rule, tau=a.tau, iou=a.iou, conf=a.conf,
                           bins=bins, check=not a.no_self_check)
        if per_seq:
            by_name[name].append(per_seq)
        else:
            print(f"  skip {name}: no overlapping sequences in {dd}")
    if not by_name:
        raise SystemExit("no arms evaluated")

    # Declared here, not inside the paired block: a single-sequence dataset skips that
    # block entirely, and the JSON must still carry the key so a figure can tell
    # "no test was possible" from "the test found nothing".
    paired_out: list[dict] = []
    rng = np.random.default_rng(a.seed)
    results: dict[str, dict] = {}
    for name, seed_runs in by_name.items():
        seqs = sorted(set.intersection(*(set(r) for r in seed_runs)))
        per_seed = [pooled_bins(r, seqs, bins) for r in seed_runs]

        # Bootstrap over SEQUENCES, once, computing EVERY bin per resample. Doing it
        # inside the per-bin loop instead re-pooled all bins for each bin -- bins x
        # resamples full passes over ~200k records, for results that are identical.
        #
        # With ONE sequence this is degenerate: every resample draws the same sequence, so
        # the interval collapses to a point. It printed CI[0.493, 0.493] on the one-flight
        # task, which reads as extreme precision and is the opposite -- a sample size of
        # one. A single-sequence task needs a moving-block bootstrap WITHIN the sequence
        # (benchmarks/block_bootstrap.py), which is what tools/make_summary.py uses there.
        boot: dict[str, list[float]] = defaultdict(list)
        if len(seqs) >= 2:
            for _ in range(a.resamples):
                pick = [seqs[i] for i in rng.integers(0, len(seqs), len(seqs))]
                for bn, (v, _) in pooled_bins(seed_runs[0], pick, bins).items():
                    boot[bn].append(v)

        cell: dict[str, dict] = {}
        for bname, _, _ in bins:
            vals = [d[bname][0] for d in per_seed if bname in d]
            if not vals:
                continue
            n_gt = next(d[bname][1] for d in per_seed if bname in d)
            bs = boot.get(bname, [])
            lo, hi = ((float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
                      if len(bs) > 1 else (float("nan"), float("nan")))
            cell[bname] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "seeds": [round(v, 4) for v in vals], "n_seeds": len(vals),
                "n_gt": n_gt, "ci95": [lo, hi], "underpowered": n_gt < a.min_gt,
            }
        results[name] = {"bins": cell, "sequences": seqs, "n_sequences": len(seqs)}

    L = [f"# Accuracy vs target size -- {a.dataset}", "",
         f"Bins on sqrt(area) in pixels ({a.bins}). Evaluator: rule={a.rule}, "
         f"IoU={a.iou}, conf>={a.conf}, pooled over {len(next(iter(results.values()))['sequences'])} "
         "sequences. Every arm is scored by this one evaluator; the arms differ only in "
         "which detections they read.", "",
         "`n` is GT instances in the bin. `+-` is the sample standard deviation over "
         "training seeds. `CI` is a 95 % bootstrap interval over SEQUENCES.", ""]

    names = list(results)
    L += ["| bin | n | " + " | ".join(names) + " |",
          "|---|---|" + "---|" * len(names)]
    for bname, _, _ in bins:
        if not any(bname in results[n]["bins"] for n in names):
            continue
        n_gt = next(results[n]["bins"][bname]["n_gt"] for n in names
                    if bname in results[n]["bins"])
        flag = " *" if n_gt < a.min_gt else ""
        cells = []
        for n in names:
            c = results[n]["bins"].get(bname)
            cells.append("--" if c is None else
                         f"{c['mean']:.3f} ± {c['std']:.3f}")
        L.append(f"| {bname}{flag} | {n_gt} | " + " | ".join(cells) + " |")
    L += ["", "`*` = fewer than %d GT instances; treat as underpowered." % a.min_gt, ""]

    # The crossover, stated only if the data actually shows one.
    if len(names) >= 2:
        ours = next((n for n in names if "our" in n.lower() or n == "ours"), names[0])
        other = next(n for n in names if n != ours)
        L += [f"## Crossover: {ours} minus {other}", "",
              "| bin | n | delta | leader |", "|---|---|---|---|"]
        deltas = []
        for bname, _, _ in bins:
            ca = results[ours]["bins"].get(bname)
            cb = results[other]["bins"].get(bname)
            if not ca or not cb:
                continue
            d = ca["mean"] - cb["mean"]
            deltas.append((bname, d, ca["n_gt"]))
            lead = "tie" if abs(d) < 1e-3 else (ours if d > 0 else other)
            flag = " *" if ca["n_gt"] < a.min_gt else ""
            L.append(f"| {bname}{flag} | {ca['n_gt']} | {d:+.3f} | {lead} |")
        L.append("")

        # A crossover is a SIGN CHANGE walking from the smallest bin upward -- but only
        # across bins that can carry one. The first version of this considered every bin
        # and duly announced a crossover on NPS between "16-25 px" and ">25 px", where the
        # larger bin holds 28 instances, both arms score 0.005, and the delta is +0.000.
        # That is a rounding artifact in an underpowered bin reported as a regime boundary,
        # which is exactly the kind of claim this file exists to prevent. Underpowered bins
        # and ties are excluded, and their exclusion is stated rather than silent.
        usable = [(b, d) for b, d, n in deltas if n >= a.min_gt and abs(d) >= 1e-3]
        dropped = [b for b, d, n in deltas if n < a.min_gt or abs(d) < 1e-3]
        if dropped:
            L += [f"Bins excluded from the crossover test: {', '.join('`'+b+'`' for b in dropped)}"
                  f" — fewer than {a.min_gt} GT instances, or a delta below 0.001.", ""]
        flips = [i for i in range(1, len(usable))
                 if (usable[i][1] > 0) != (usable[i - 1][1] > 0)]
        if flips:
            i = flips[0]
            lo_b, lo_d = usable[i - 1]
            hi_b, hi_d = usable[i]
            winner_lo = ours if lo_d > 0 else other
            L += [f"**Crossover between `{lo_b}` and `{hi_b}`.** {winner_lo} leads on the "
                  f"smaller side ({lo_d:+.3f}) and trails on the larger ({hi_d:+.3f}).", ""]
        elif usable:
            who = ours if usable[0][1] > 0 else other
            L += [f"**No crossover on {a.dataset}.** {who} leads in every bin with enough "
                  f"GT to judge ({', '.join('`'+b+'`' for b, _ in usable)}), so there is no "
                  "size regime that separates the methods here.", ""]
        else:
            L += [f"**No bin on {a.dataset} has enough GT to test for a crossover.**", ""]

        # State the range the dataset can actually speak to. NPS turned out to contain no
        # target below 10 px at all, so no arm can be credited or blamed there -- and a
        # reader scanning a table of five bin headings would not otherwise notice that two
        # of them are simply absent.
        present = [b for b, _, _ in deltas]
        missing = [b for b, _, _ in bins if b not in present]
        if missing:
            L += [f"`{a.dataset}` contains **no ground truth at all** in "
                  f"{', '.join('`'+b+'`' for b in missing)}. Nothing about those sizes can "
                  "be concluded from this dataset.", ""]

        # -------------------------------------------------------- significance, per bin
        # The marginal CIs in the table above are NOT the test. They are each arm's own
        # spread over sequences, and sequences differ enormously in difficulty, so the two
        # arms' intervals overlap heavily even where one arm wins on every sequence and
        # every seed. The paired test removes exactly that shared between-sequence
        # variance, which is the whole reason it is the one this project reports.
        runs_a, runs_b = by_name.get(ours, []), by_name.get(other, [])
        common = sorted(set.intersection(*(set(r) for r in runs_a + runs_b))) \
            if runs_a and runs_b else []
        if common and len(common) >= 2 and runs_a and runs_b:
            L += ["## Is the difference real? Paired, seed-matched, over sequences", "",
                  f"Paired bootstrap **and** permutation over the {len(common)} shared "
                  "sequences; a bin is called significant only when both agree, matching "
                  "`tools/make_summary.py`. Seeds are matched pairwise.", "",
                  "| bin | seed | d AP | 95% CI | p perm | verdict |",
                  "|---|---|---|---|---|---|"]
            sig_summary: dict[str, list[bool]] = {}
            for bname, _, _ in bins:
                if bname not in results[ours]["bins"]:
                    continue
                for s in range(min(len(runs_a), len(runs_b))):
                    r = paired_bin_test(runs_a[s], runs_b[s], common, bname,
                                        a.resamples, seed=s)
                    if not r:
                        continue
                    sig_summary.setdefault(bname, []).append(
                        bool(r["significant"]) and r["observed"] > 0)
                    paired_out.append({"bin": bname, "seed": s,
                                       "d_ap": r["observed"], "lo": r["lo"],
                                       "hi": r["hi"], "p_perm": r["p_perm"],
                                       "significant": bool(r["significant"]),
                                       "favours": (ours if r["observed"] > 0
                                                   else other)})
                    L.append(f"| {bname} | {s} | {r['observed']:+.3f} | "
                             f"[{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['p_perm']:.4f} | "
                             f"{'**significant**' if r['significant'] else 'no difference'} |")
            L.append("")
            won = [b for b, v in sig_summary.items() if v and all(v)]
            if won:
                L += [f"**{ours} wins significantly on every seed in: "
                      f"{', '.join('`'+b+'`' for b in won)}.**", ""]
            else:
                L += [f"**No bin where {ours} wins significantly on every seed.**", ""]

    out = "\n".join(L) + "\n"
    print(out)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"{a.dataset}_{a.bins}.md").write_text(out, encoding="utf-8")
        (a.out / f"{a.dataset}_{a.bins}.json").write_text(
            json.dumps({"dataset": a.dataset, "bins": a.bins, "rule": a.rule,
                        "iou": a.iou, "conf": a.conf, "resamples": a.resamples,
                        "min_gt": a.min_gt, "arms": results,
                        "paired": paired_out}, indent=2),
            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
