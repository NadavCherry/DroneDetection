"""The fast pooled-AP resampler must agree with the slow one exactly, not approximately.

`PooledAPResampler` exists only because `pooled_ap` is too slow to bootstrap at the scale
we now evaluate at (354 ms per call, 397k detections, 20,000 calls per seed). A faster
statistic that quietly computes something slightly different would be far worse than a
slow one: it would move published intervals for a reason invisible in the output.

So every test here compares against `benchmarks.scorecard.pooled_ap` on the same data,
including on the awkward inputs -- multiset resamples where a sequence is drawn twice or
not at all, which is exactly what a bootstrap does and what a naive implementation gets
wrong by concatenating instead of re-weighting.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.fast_bootstrap import PooledAPResampler, paired_bootstrap_pooled_ap
from benchmarks.scorecard import SequenceResult, pooled_ap


def _seqs(n_seq=6, n_det=400, seed=0, tp_rate=0.25):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_seq):
        dets = [(float(rng.random()),
                 "tp" if rng.random() < tp_rate else "fp") for _ in range(n_det)]
        n_gt = max(1, int(sum(1 for _, o in dets if o == "tp") * 1.4))
        out.append(SequenceResult(sequence=f"v{i}", n_gt=n_gt, n_frames=100,
                                  detections=dets))
    return out


def test_matches_pooled_ap_on_the_full_set():
    seqs = _seqs()
    r = PooledAPResampler(seqs)
    got = r.ap_for_counts(np.ones(len(seqs)))
    assert got == pytest.approx(pooled_ap(seqs), abs=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_matches_pooled_ap_on_bootstrap_multisets(seed):
    """The case a concatenation-based implementation gets wrong: a resample draws some
    sequences twice and others zero times, and the duplicates must count twice in BOTH
    the detections and n_gt."""
    seqs = _seqs(seed=seed)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(seqs), size=len(seqs))

    counts = np.bincount(idx, minlength=len(seqs)).astype(float)
    fast = PooledAPResampler(seqs).ap_for_counts(counts)
    slow = pooled_ap([seqs[j] for j in idx])          # the resampled collection, literally

    assert fast == pytest.approx(slow, abs=1e-12), f"counts={counts}"


def test_a_sequence_drawn_twice_is_not_the_same_as_drawn_once():
    """Guards against silently ignoring the weights -- which would still pass the
    full-set test above, since there every count is 1."""
    seqs = _seqs(n_seq=3, seed=3)
    r = PooledAPResampler(seqs)
    once = r.ap_for_counts(np.array([1.0, 1.0, 1.0]))
    twice = r.ap_for_counts(np.array([2.0, 1.0, 0.0]))
    assert once != pytest.approx(twice), "weights are being ignored"
    assert twice == pytest.approx(pooled_ap([seqs[0], seqs[0], seqs[1]]), abs=1e-12)


def test_distractors_are_dropped_not_scored():
    seqs = _seqs(n_seq=3, seed=5)
    seqs[0].detections.append((0.99, "distractor"))
    r = PooledAPResampler(seqs)
    assert r.ap_for_counts(np.ones(3)) == pytest.approx(pooled_ap(seqs), abs=1e-12)


def test_empty_and_degenerate_inputs_do_not_raise():
    empty = [SequenceResult(sequence="v0", n_gt=0, n_frames=0, detections=[])]
    assert PooledAPResampler(empty).ap_for_counts(np.ones(1)) == 0.0

    no_gt = [SequenceResult(sequence="v0", n_gt=0, n_frames=10,
                            detections=[(0.5, "fp")])]
    assert PooledAPResampler(no_gt).ap_for_counts(np.ones(1)) == 0.0


def test_paired_bootstrap_is_paired_and_reproducible():
    a, b = _seqs(seed=1, tp_rate=0.35), _seqs(seed=1, tp_rate=0.20)
    r1 = paired_bootstrap_pooled_ap(a, b, n_resamples=200, seed=7)
    r2 = paired_bootstrap_pooled_ap(a, b, n_resamples=200, seed=7)

    assert r1 == r2, "same seed must give the same interval"
    assert r1["observed"] == pytest.approx(pooled_ap(a) - pooled_ap(b), abs=1e-12)
    assert r1["lo"] <= r1["observed"] <= r1["hi"] or r1["p"] < 1.0
    assert r1["n_units"] == len(a)


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="equal lengths"):
        paired_bootstrap_pooled_ap(_seqs(n_seq=3), _seqs(n_seq=4), n_resamples=10)


def test_paired_selection_matches_pooled_ap_on_a_mixed_arm_draw():
    """The permutation test's core operation: some units taken from arm A, others from
    arm B. Getting the arm mask wrong would let BOTH arms' detections into one curve --
    which does not crash and makes every p-value meaningless."""
    from benchmarks.fast_bootstrap import PairedPooledAP

    a, b = _seqs(seed=2, tp_rate=0.35), _seqs(seed=2, tp_rate=0.15)
    p = PairedPooledAP(a, b)
    rng = np.random.default_rng(4)
    for _ in range(6):
        pick = (rng.random(len(a)) < 0.5).astype(np.intp)
        fast = p.ap(np.ones(len(a)), pick)
        slow = pooled_ap([(b[i] if pick[i] else a[i]) for i in range(len(a))])
        assert fast == pytest.approx(slow, abs=1e-12), f"pick={pick}"


def test_fast_permutation_matches_the_slow_permutation_test():
    from benchmarks.fast_bootstrap import paired_permutation_pooled_ap
    from dronedet.stats import paired_permutation_test

    a, b = _seqs(seed=6, tp_rate=0.34), _seqs(seed=6, tp_rate=0.21)
    fast = paired_permutation_pooled_ap(a, b, n_resamples=300, seed=11)
    slow = paired_permutation_test(a, b, pooled_ap, n_resamples=300, seed=11)
    assert fast == pytest.approx(slow, abs=1e-12), \
        "the fast permutation test must draw the same swaps and reach the same p"


def test_11pt_style_differs_from_all_point_on_a_truncated_curve():
    """Both styles exist because GLAD uses 11-point; they must not be silently aliased."""
    seqs = _seqs(n_seq=4, seed=11, tp_rate=0.1)      # low recall -> truncated curve
    r_all = PooledAPResampler(seqs, "all-point").ap_for_counts(np.ones(4))
    r_11 = PooledAPResampler(seqs, "11pt").ap_for_counts(np.ones(4))
    assert r_all != pytest.approx(r_11), "11-point and all-point should differ here"
    assert 0.0 <= r_11 <= 1.0
