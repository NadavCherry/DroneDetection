"""Tests for dronedet.stats.

The property worth guarding hardest is the one the module exists to enforce: a
comparison against a *published scalar* must never produce a p-value, because there is
no second sample to test against. `test_published_comparison_never_produces_a_p_value`
is that guard.
"""

from __future__ import annotations

import pytest

from dronedet import stats as S


# ------------------------------------------------------------------ wilson
def test_wilson_does_not_claim_certainty_from_a_perfect_score():
    p, lo, hi = S.wilson(31, 31)
    assert p == 1.0
    assert lo < 0.95, "31/31 must not imply near-certainty"
    assert hi <= 1.0


def test_wilson_handles_zero_samples():
    assert S.wilson(0, 0) == (0.0, 0.0, 0.0)


def test_wilson_interval_narrows_with_more_samples():
    _, lo_small, hi_small = S.wilson(8, 10)
    _, lo_big, hi_big = S.wilson(800, 1000)
    assert (hi_big - lo_big) < (hi_small - lo_small)


# ------------------------------------------------------------------ holm
def test_holm_inflates_p_values_and_is_monotone():
    out = S.holm([("a", 0.01), ("b", 0.02), ("c", 0.04)])
    names = [n for n, _, _ in out]
    assert names == ["a", "b", "c"]                # sorted by raw p
    for _, raw, adj in out:
        assert adj >= raw
    adjusted = [adj for _, _, adj in out]
    assert adjusted == sorted(adjusted), "adjusted p must be non-decreasing"


def test_holm_can_kill_a_borderline_finding():
    """The point of the correction: best-of-eight at p=0.04 is not a finding."""
    out = S.holm([(f"f{i}", p) for i, p in enumerate([0.04, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])])
    assert dict((n, adj) for n, _, adj in out)["f0"] > 0.05


# ------------------------------------------------------------------ mcnemar
def test_mcnemar_ignores_agreements():
    """Items both detectors get right carry no information about which is better."""
    assert S.mcnemar(0, 0) == 1.0


def test_mcnemar_is_symmetric():
    assert S.mcnemar(9, 1) == pytest.approx(S.mcnemar(1, 9))


def test_mcnemar_detects_a_lopsided_split():
    assert S.mcnemar(20, 2) < 0.001
    assert S.mcnemar(6, 5) > 0.5


def test_mcnemar_never_exceeds_one():
    assert S.mcnemar(1, 1) <= 1.0
    assert S.mcnemar(2, 2) <= 1.0


# ------------------------------------------------------------------ paired bootstrap
def _mean(units):
    return sum(units) / len(units) if units else 0.0


def test_paired_bootstrap_finds_a_consistent_difference():
    a = [0.90, 0.92, 0.88, 0.91, 0.89, 0.93, 0.90, 0.92]
    b = [x - 0.10 for x in a]                       # A better on every single unit
    r = S.paired_bootstrap_diff(a, b, _mean, n_resamples=2000, seed=1)
    assert r.observed == pytest.approx(0.10, abs=1e-9)
    assert r.significant and r.lo > 0
    assert r.p_value < 0.05


def test_paired_bootstrap_reports_no_difference_when_there_is_none():
    a = [0.5, 0.7, 0.3, 0.9, 0.4, 0.6, 0.8, 0.2]
    b = list(reversed(a))                           # same values, different order
    r = S.paired_bootstrap_diff(a, b, _mean, n_resamples=2000, seed=1)
    assert not r.significant
    assert r.lo <= 0 <= r.hi


def test_paired_bootstrap_pairing_removes_shared_unit_variance():
    """Two easy and two hard videos, A always +0.05. Unpaired the spread would swamp
    the effect; paired it must not."""
    a = [0.95, 0.94, 0.35, 0.34]
    b = [0.90, 0.89, 0.30, 0.29]
    r = S.paired_bootstrap_diff(a, b, _mean, n_resamples=3000, seed=2)
    assert r.observed == pytest.approx(0.05, abs=1e-9)
    assert r.lo > 0.0, "a constant per-unit gain must survive the huge between-unit spread"


def test_paired_bootstrap_is_deterministic_for_a_seed():
    a, b = [0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3, 0.2]
    r1 = S.paired_bootstrap_diff(a, b, _mean, n_resamples=500, seed=7)
    r2 = S.paired_bootstrap_diff(a, b, _mean, n_resamples=500, seed=7)
    assert (r1.lo, r1.hi, r1.p_value) == (r2.lo, r2.hi, r2.p_value)


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        S.paired_bootstrap_diff([1.0, 2.0], [1.0], _mean)


def test_paired_bootstrap_rejects_empty_input():
    with pytest.raises(ValueError, match="no units"):
        S.paired_bootstrap_diff([], [], _mean)


def test_bootstrap_describe_flags_an_interval_that_includes_zero():
    a = [0.5, 0.6, 0.4, 0.55]
    r = S.paired_bootstrap_diff(a, list(reversed(a)), _mean, n_resamples=500, seed=3)
    assert "includes 0" in r.describe()


# ------------------------------------------------------------------ permutation
def test_permutation_test_agrees_with_bootstrap_on_a_clear_effect():
    a = [0.9] * 10
    b = [0.5] * 10
    assert S.paired_permutation_test(a, b, _mean, n_resamples=2000, seed=1) < 0.05


def test_permutation_p_is_never_zero():
    """The add-one correction: p=0 would claim more certainty than n resamples allow."""
    a, b = [1.0] * 8, [0.0] * 8
    assert S.paired_permutation_test(a, b, _mean, n_resamples=100, seed=1) > 0.0


def test_permutation_test_on_identical_inputs_is_not_significant():
    a = [0.4, 0.5, 0.6, 0.7, 0.8]
    assert S.paired_permutation_test(a, list(a), _mean, n_resamples=500, seed=1) > 0.5


# ------------------------------------------------------------------ effect size
def test_cliffs_delta_sign_and_magnitude():
    d, label = S.cliffs_delta([5, 6, 7, 8], [1, 2, 3, 4])
    assert d == pytest.approx(1.0) and label == "large"
    d2, _ = S.cliffs_delta([1, 2, 3, 4], [5, 6, 7, 8])
    assert d2 == pytest.approx(-1.0)


def test_cliffs_delta_calls_a_tiny_consistent_gain_negligible():
    """A difference can be perfectly consistent and still not matter."""
    xs = [0.500, 0.501, 0.502, 0.503]
    ys = [0.4995, 0.4996, 0.4997, 0.4998]
    d, label = S.cliffs_delta(xs, ys)
    assert d == pytest.approx(1.0)          # ranks fully separated...
    assert label == "large"                 # ...so rank-based effect size is large
    # which is exactly why an effect size must be read next to the raw difference:
    assert (sum(xs) / 4) - (sum(ys) / 4) < 0.005


def test_cliffs_delta_handles_empty_input():
    assert S.cliffs_delta([], [1, 2]) == (0.0, "undefined")


# ------------------------------------------------------------------ published comparison
def test_published_comparison_never_produces_a_p_value():
    """The load-bearing guarantee of this module. A published AP is one scalar; there is
    no sample to test against, so no significance claim may be manufactured."""
    c = S.compare_with_published(0.85, (0.80, 0.90), 0.83, "YOLOMG")
    assert not hasattr(c, "p_value")
    text = c.describe()
    assert "p=" not in text, "no p-value may be emitted against a published scalar"
    # ...and the absence must be stated, not merely implied by omission
    assert "no significance test is possible" in text


def test_published_comparison_reports_coverage_not_victory():
    c = S.compare_with_published(0.85, (0.80, 0.90), 0.83, "YOLOMG")
    assert c.covers_published
    assert c.verdict() == "indistinguishable at this sample size"


def test_published_comparison_admits_when_ours_is_higher():
    c = S.compare_with_published(0.90, (0.88, 0.92), 0.83, "YOLOMG")
    assert not c.covers_published and c.verdict() == "ours higher"


def test_published_comparison_admits_when_ours_is_lower():
    c = S.compare_with_published(0.70, (0.68, 0.72), 0.83, "YOLOMG")
    assert c.verdict() == "ours lower"


def test_protocol_mismatch_blocks_the_comparison_entirely():
    """Different metric or different split means the point estimates are not even
    on the same axis, whatever the intervals say."""
    c = S.compare_with_published(
        0.99, (0.98, 1.00), 0.55, "MGMD",
        protocol_mismatches=["centre-distance tau=12 px vs their IoU 0.25",
                             "one clip vs the official 15-video test split"])
    assert not c.comparable
    assert c.verdict() == "NOT COMPARABLE"
    text = c.describe()
    assert "NOT COMPARABLE" in text
    assert "IoU 0.25" in text and "15-video" in text


def test_published_comparison_surfaces_the_source_url():
    c = S.compare_with_published(0.8, (0.7, 0.9), 0.85, "YOLOMG",
                                 published_url="https://arxiv.org/abs/2503.07115")
    assert "arxiv.org/abs/2503.07115" in c.describe()
