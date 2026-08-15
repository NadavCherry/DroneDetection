"""A pooled-AP bootstrap that finishes.

THE PROBLEM
-----------
`paired_bootstrap_diff` draws 10,000 resamples and evaluates the statistic twice per
resample. With `pooled_ap` that means 20,000 passes which each build a Python list of
every detection and sort it. Measured at the real ARD-MAV shape -- 15 sequences, 397,500
detections -- one `pooled_ap` call takes 354 ms, so one seed's bootstrap is 118 minutes
and three seeds with a permutation test on top is most of a day. A statistic nobody can
afford to run is a statistic that does not get reported.

This is a scaling problem that our own correctness fix created: both arms are now scored
at a conf floor of 0.001 to match the competitor's, which is right, and which produces
roughly ten times the detections the old 0.05 floor did.

THE FIX
-------
The resampling unit is the SEQUENCE, and there are 15 of them. Every resample is therefore
just a multiset of the same sequences, so the scores never change -- only how many times
each detection is counted. So sort once, globally, and afterwards a resample is a weighted
cumulative sum over that fixed order:

    weights   = counts[seq_of_detection]        (0, 1, 2 ... times drawn)
    tp_cum    = cumsum(weights * is_tp)
    fp_cum    = cumsum(weights * ~is_tp)
    n_gt      = counts . n_gt_per_sequence

No per-resample sort, no Python-level list building, three numpy passes. Same arithmetic,
same answer -- `test_fast_bootstrap.py` asserts equality against `pooled_ap` on random
resamples rather than trusting that claim.

Ties: `average_precision` sorts by score alone and leaves equal scores in whatever order
the sort produced. A stable descending sort here reproduces that, and since the cumulative
sums are taken over the whole array the interpolated envelope is unaffected by tie order
anyway.
"""

from __future__ import annotations

import numpy as np


class PooledAPResampler:
    """Pre-sorted detections for one arm, ready to be re-weighted per bootstrap draw."""

    def __init__(self, sequences, ap_style: str = "all-point"):
        if ap_style not in ("all-point", "11pt"):
            raise ValueError(f"unknown ap_style {ap_style!r}")
        self.ap_style = ap_style
        self.n_units = len(sequences)
        self.n_gt = np.array([s.n_gt for s in sequences], dtype=np.float64)

        scores, is_tp, unit = [], [], []
        for i, s in enumerate(sequences):
            for sc, outcome in s.detections:
                if outcome not in ("tp", "fp"):
                    continue                      # distractors are dropped, not counted
                scores.append(sc)
                is_tp.append(outcome == "tp")
                unit.append(i)

        if not scores:
            self.empty = True
            return
        self.empty = False
        order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
        self.is_tp = np.asarray(is_tp, dtype=np.float64)[order]
        self.is_fp = 1.0 - self.is_tp
        self.unit = np.asarray(unit, dtype=np.intp)[order]

    def ap_for_counts(self, counts: np.ndarray) -> float:
        """AP over the multiset of sequences described by `counts`."""
        if self.empty:
            return 0.0
        n_gt = float(counts @ self.n_gt)
        if n_gt <= 0:
            return 0.0

        w = counts.astype(np.float64)[self.unit]
        tp = np.cumsum(w * self.is_tp)
        fp = np.cumsum(w * self.is_fp)
        keep = (tp + fp) > 0
        if not keep.any():
            return 0.0
        tp, fp = tp[keep], fp[keep]

        recall = tp / n_gt
        precision = tp / np.maximum(tp + fp, 1e-12)
        precision = np.maximum.accumulate(precision[::-1])[::-1]

        if self.ap_style == "11pt":
            # Mean of max precision at recall >= t. Zero above the highest recall reached,
            # which is the whole reason 11-point and all-point disagree on a truncated curve.
            grid = np.linspace(0.0, 1.0, 11)
            idx = np.searchsorted(recall, grid, side="left")
            vals = np.where(idx < precision.size, precision[np.minimum(idx, precision.size - 1)], 0.0)
            return float(vals.mean())

        # All-point: integrate precision over the recall increments.
        prev = np.concatenate(([0.0], recall[:-1]))
        return float(np.sum((recall - prev) * precision))


class PairedPooledAP:
    """Both arms' detections in ONE globally sorted order, selectable per unit.

    The bootstrap only ever needs one arm at a time, but the permutation test swaps
    individual units BETWEEN arms -- under the null "the A/B labels are arbitrary" -- so it
    needs a structure that can answer "AP over unit 0 from arm A, unit 1 from arm B, ...".
    Carrying an arm tag alongside the unit tag makes both tests the same operation:

        weight(detection) = counts[unit] * (arm == pick[unit])

    with `pick` all-zeros for arm A, all-ones for arm B, and a random 0/1 vector for a
    permutation draw.
    """

    def __init__(self, seqs_a, seqs_b, ap_style: str = "all-point"):
        if len(seqs_a) != len(seqs_b):
            raise ValueError(f"paired comparison needs equal lengths, got "
                             f"{len(seqs_a)} and {len(seqs_b)}")
        self.ap_style = ap_style
        self.n_units = n = len(seqs_a)
        # n_gt[arm, unit] -- the two arms score the same sequences, so these normally
        # agree, but they are kept separately rather than assumed equal.
        self.n_gt = np.array([[s.n_gt for s in seqs_a],
                              [s.n_gt for s in seqs_b]], dtype=np.float64)

        scores, is_tp, unit, arm = [], [], [], []
        for arm_id, seqs in ((0, seqs_a), (1, seqs_b)):
            for i, s in enumerate(seqs):
                for sc, outcome in s.detections:
                    if outcome not in ("tp", "fp"):
                        continue
                    scores.append(sc)
                    is_tp.append(outcome == "tp")
                    unit.append(i)
                    arm.append(arm_id)

        self.empty = not scores
        if self.empty:
            return
        order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
        self.is_tp = np.asarray(is_tp, dtype=np.float64)[order]
        self.is_fp = 1.0 - self.is_tp
        self.unit = np.asarray(unit, dtype=np.intp)[order]
        self.arm = np.asarray(arm, dtype=np.intp)[order]
        self._ones = np.ones(n, dtype=np.float64)

    def ap(self, counts: np.ndarray, pick: np.ndarray) -> float:
        """AP over `counts[i]` copies of unit i, taken from arm `pick[i]`."""
        if self.empty:
            return 0.0
        n_gt = float(np.sum(counts * self.n_gt[pick, np.arange(self.n_units)]))
        if n_gt <= 0:
            return 0.0

        keep_arm = (self.arm == pick[self.unit])
        w = counts[self.unit] * keep_arm
        tp = np.cumsum(w * self.is_tp)
        fp = np.cumsum(w * self.is_fp)
        keep = (tp + fp) > 0
        if not keep.any():
            return 0.0
        tp, fp = tp[keep], fp[keep]

        recall = tp / n_gt
        precision = tp / np.maximum(tp + fp, 1e-12)
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        if self.ap_style == "11pt":
            grid = np.linspace(0.0, 1.0, 11)
            idx = np.searchsorted(recall, grid, side="left")
            vals = np.where(idx < precision.size,
                            precision[np.minimum(idx, precision.size - 1)], 0.0)
            return float(vals.mean())
        prev = np.concatenate(([0.0], recall[:-1]))
        return float(np.sum((recall - prev) * precision))


def paired_permutation_pooled_ap(seqs_a, seqs_b, *, n_resamples: int = 10000,
                                 seed: int = 0, ap_style: str = "all-point") -> float:
    """Two-sided paired permutation p-value for pooled AP. Same result as
    `dronedet.stats.paired_permutation_test(..., pooled_ap)`, computed without re-sorting.
    """
    p = PairedPooledAP(seqs_a, seqs_b, ap_style)
    n = p.n_units
    if n == 0:
        raise ValueError("no units to compare")
    ones = np.ones(n, dtype=np.float64)
    zeros_pick = np.zeros(n, dtype=np.intp)
    ones_pick = np.ones(n, dtype=np.intp)
    observed = abs(p.ap(ones, zeros_pick) - p.ap(ones, ones_pick))

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_resamples):
        swap = (rng.random(n) < 0.5).astype(np.intp)
        if abs(p.ap(ones, swap) - p.ap(ones, 1 - swap)) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_resamples + 1)      # add-one keeps p strictly positive


def paired_bootstrap_pooled_ap(seqs_a, seqs_b, *, n_resamples: int = 10000,
                               alpha: float = 0.05, seed: int = 0,
                               ap_style: str = "all-point"):
    """Paired bootstrap of (AP_a - AP_b) over sequences. Returns the same shape of result
    as `dronedet.stats.paired_bootstrap_diff`, computed the fast way.

    `seqs_a[i]` and `seqs_b[i]` must be the two arms' results on the SAME sequence: the
    indices are drawn jointly, which is what removes the between-sequence variance both
    arms share and stops an easy video counting as evidence for either.
    """
    if len(seqs_a) != len(seqs_b):
        raise ValueError(f"paired comparison needs equal lengths, got "
                         f"{len(seqs_a)} and {len(seqs_b)}")
    n = len(seqs_a)
    if n == 0:
        raise ValueError("no units to compare")

    ra = PooledAPResampler(seqs_a, ap_style)
    rb = PooledAPResampler(seqs_b, ap_style)
    ones = np.ones(n, dtype=np.float64)
    observed = ra.ap_for_counts(ones) - rb.ap_for_counts(ones)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        counts = np.bincount(idx, minlength=n).astype(np.float64)
        diffs[i] = ra.ap_for_counts(counts) - rb.ap_for_counts(counts)

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    frac_le = float(np.mean(diffs <= 0.0))
    frac_ge = float(np.mean(diffs >= 0.0))
    p = float(min(1.0, 2.0 * min(frac_le, frac_ge)))
    return {"observed": float(observed), "lo": lo, "hi": hi, "p": p,
            "n_resamples": n_resamples, "n_units": n}


__all__ = ["PooledAPResampler", "paired_bootstrap_pooled_ap"]
