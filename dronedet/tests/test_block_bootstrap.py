"""The within-sequence bootstrap must be honest about being within-sequence.

Two videos are the whole corpus for the project's own task, so the test set is one
sequence and the usual sequence-level bootstrap is undefined. The temptation is to
resample FRAMES instead, which would be wrong in a way that flatters: consecutive frames
of one flight are near-duplicates, so treating 250 of them as independent produces an
interval several times too narrow and a p-value to match.

These tests pin the two properties that keep the block bootstrap from becoming that:
blocks are contiguous and cut identically for both arms, and the resulting interval is
WIDER than the frame-level one it replaces.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.block_bootstrap import (blocks_from_detections,
                                        paired_block_bootstrap)


def _synthetic(n_frames=240, seed=0, hit_rate=0.7, corr=True):
    """A sequence with ONE ground-truth box per frame and strongly autocorrelated
    outcomes -- long runs of hits and long runs of misses, as a real flight has."""
    rng = np.random.default_rng(seed)
    gt = {f: 1 for f in range(n_frames)}
    det = {}
    state = True
    for f in range(n_frames):
        if corr:
            if rng.random() < 0.02:          # rare switch -> long runs
                state = not state
            hit = state if rng.random() < 0.95 else not state
        else:
            hit = rng.random() < hit_rate
        det[f] = [(float(rng.uniform(0.3, 0.99)), bool(hit))]
    return gt, det


def test_blocks_are_contiguous_and_cover_every_frame():
    gt, det = _synthetic(n_frames=250)
    blocks = blocks_from_detections(gt, det, block_frames=30, n_frames=250)

    assert len(blocks) == 9, "250 frames in 30-frame blocks is 8 full + 1 partial"
    assert sum(g for g, _ in blocks) == 250, "every frame's ground truth must appear once"
    assert sum(len(d) for _, d in blocks) == 250, "every detection must appear once"


def test_both_arms_are_cut_at_the_same_boundaries():
    """If the two arms were blocked differently, a paired draw would compare block k of
    one flight against a different stretch of the same flight."""
    gt, det_a = _synthetic(seed=1)
    _, det_b = _synthetic(seed=2)
    a = blocks_from_detections(gt, det_a, block_frames=30, n_frames=240)
    b = blocks_from_detections(gt, det_b, block_frames=30, n_frames=240)

    assert len(a) == len(b)
    assert [g for g, _ in a] == [g for g, _ in b], "ground truth per block must match"


def test_the_interval_is_wider_than_a_frame_level_one_would_be():
    """The whole point. Frame-level resampling of autocorrelated data understates the
    spread; blocking must widen it."""
    gt, det_a = _synthetic(seed=3)
    _, det_b = _synthetic(seed=4)

    wide = paired_block_bootstrap(
        blocks_from_detections(gt, det_a, block_frames=30, n_frames=240),
        blocks_from_detections(gt, det_b, block_frames=30, n_frames=240),
        block_frames=30, n_resamples=600, seed=0)
    narrow = paired_block_bootstrap(
        blocks_from_detections(gt, det_a, block_frames=1, n_frames=240),
        blocks_from_detections(gt, det_b, block_frames=1, n_frames=240),
        block_frames=1, n_resamples=600, seed=0)

    assert (wide.hi - wide.lo) > (narrow.hi - narrow.lo), (
        f"30-frame blocks gave width {wide.hi - wide.lo:.4f}, single frames "
        f"{narrow.hi - narrow.lo:.4f} -- blocking must not narrow the interval")


def test_identical_arms_give_a_difference_of_zero_and_no_significance():
    gt, det = _synthetic(seed=5)
    blocks = blocks_from_detections(gt, det, block_frames=30, n_frames=240)
    r = paired_block_bootstrap(blocks, blocks, n_resamples=400, seed=0)

    assert r.observed == pytest.approx(0.0, abs=1e-12)
    assert not r.significant
    assert r.p_value == pytest.approx(1.0, abs=1e-9)


def test_the_description_says_within_one_sequence():
    """A caption must not be able to quote the interval without the caveat."""
    gt, det_a = _synthetic(seed=6)
    _, det_b = _synthetic(seed=7)
    r = paired_block_bootstrap(
        blocks_from_detections(gt, det_a, block_frames=30, n_frames=240),
        blocks_from_detections(gt, det_b, block_frames=30, n_frames=240),
        n_resamples=200, seed=0)

    text = r.describe("temporal", "singleframe")
    assert "WITHIN ONE SEQUENCE" in text
    assert "not generalisation" in text
    assert "blocks" in text


def test_too_few_blocks_is_refused_rather_than_reported():
    gt, det = _synthetic(n_frames=20)
    one = blocks_from_detections(gt, det, block_frames=30, n_frames=20)
    with pytest.raises(ValueError, match="not enough to resample"):
        paired_block_bootstrap(one, one, n_resamples=100)


def test_mismatched_block_counts_are_refused():
    gt, det = _synthetic()
    a = blocks_from_detections(gt, det, block_frames=30, n_frames=240)
    b = blocks_from_detections(gt, det, block_frames=60, n_frames=240)
    with pytest.raises(ValueError, match="equal block counts"):
        paired_block_bootstrap(a, b, n_resamples=100)
