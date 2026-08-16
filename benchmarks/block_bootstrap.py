"""Significance inside ONE sequence, by resampling contiguous blocks of frames.

WHY A DIFFERENT BOOTSTRAP
-------------------------
The paired bootstrap this project uses everywhere else resamples SEQUENCES, because
sequences are the independent units: two flights on different days share nothing, while
two frames 33 ms apart share almost everything. That is the right unit and it is why the
ARD-MAV comparison is over 15 videos rather than 28,160 frames.

But the project's own task has TWO videos in total -- train on 07_05, test on 10_06 -- so
the test set is a single sequence and n = 1. A sequence-level bootstrap is undefined, and
resampling individual frames would be flatly wrong: consecutive frames of one flight are
near-duplicates, so it would treat ~250 correlated observations as 250 independent ones
and report an interval several times too narrow.

The moving-block bootstrap is the standard answer for exactly this. Cut the sequence into
contiguous blocks long enough to contain the correlation, then resample BLOCKS with
replacement. Within a block the temporal structure is preserved; between blocks the
resampling assumes independence, which is approximately true once the block is longer than
the correlation length.

WHAT IT DOES AND DOES NOT LICENSE
---------------------------------
It answers: *is this difference stable across the segments of this flight?*
It does NOT answer: *does this difference generalise to another flight?*

Only more sequences answer the second, and no amount of resampling one video substitutes
for them. Every number this module produces must be labelled as within-sequence, and
`describe()` does that in the string itself so a figure caption cannot quietly drop it.

BLOCK LENGTH
------------
Default 30 frames (1 second at 30 fps). Long enough that a drone's appearance and the
camera's motion have both changed substantially across a block boundary; short enough that
250 annotated frames still yield ~8 blocks. The choice matters -- too short and the
interval is too narrow, too long and there are too few blocks to resample -- so it is a
parameter, it is recorded in the result, and `n_blocks` is reported alongside every
interval so a reader can see how thin the evidence is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BlockBootstrapResult:
    observed: float
    lo: float
    hi: float
    p_value: float
    n_blocks: int
    block_frames: int
    n_resamples: int
    statistic_a: float
    statistic_b: float

    @property
    def significant(self) -> bool:
        return (self.lo > 0.0) or (self.hi < 0.0)

    def describe(self, name_a: str = "A", name_b: str = "B") -> str:
        verdict = "excludes 0" if self.significant else "**includes 0**"
        return (f"{name_a} {self.statistic_a:.3f} vs {name_b} {self.statistic_b:.3f} — "
                f"difference {self.observed:+.3f}, 95% CI [{self.lo:+.3f}, {self.hi:+.3f}] "
                f"({verdict}), p={self.p_value:.4f}, moving-block bootstrap over "
                f"{self.n_blocks} blocks of {self.block_frames} frames "
                f"**WITHIN ONE SEQUENCE** — this is stability across segments of one "
                f"flight, not generalisation to another flight")


def split_into_blocks(seq, block_frames: int = 30):
    """One SequenceResult -> a list of per-block SequenceResults, in time order.

    Detections carry no frame index in `SequenceResult` (it stores `(score, outcome)`
    pairs), so blocks are formed over the FRAME axis and the ground truth is apportioned
    by frame while detections are apportioned proportionally by position. That is an
    approximation, and it is the reason this returns blocks rather than pretending to be
    exact: use `blocks_from_detections` when frame-indexed detections are available.
    """
    raise NotImplementedError(
        "SequenceResult does not retain per-detection frame indices; build blocks with "
        "blocks_from_detections() from the detection JSON and GT instead")


def blocks_from_detections(gt_frames: dict, det_frames: dict, *, block_frames: int = 30,
                           n_frames: int | None = None):
    """Cut one sequence into contiguous frame blocks. -> list of (n_gt, [(score, is_tp)]).

    `gt_frames`  : {frame -> n_gt_in_that_frame}
    `det_frames` : {frame -> [(score, is_tp), ...]}

    Both come from the evaluator's per-frame records, so a block's ground truth and its
    detections are cut at the same boundary and no detection is scored against a block
    that does not contain its target.
    """
    if n_frames is None:
        n_frames = (max(list(gt_frames) + list(det_frames)) + 1
                    if (gt_frames or det_frames) else 0)
    if n_frames <= 0:
        return []

    blocks = []
    for start in range(0, n_frames, block_frames):
        stop = min(start + block_frames, n_frames)
        n_gt = sum(gt_frames.get(f, 0) for f in range(start, stop))
        dets = [d for f in range(start, stop) for d in det_frames.get(f, [])]
        blocks.append((n_gt, dets))
    return blocks


def _ap(blocks_idx, blocks) -> float:
    """Pooled all-point AP over a multiset of blocks."""
    n_gt = 0
    scores, is_tp = [], []
    for j in blocks_idx:
        g, dets = blocks[j]
        n_gt += g
        for s, tp in dets:
            scores.append(s)
            is_tp.append(tp)
    if n_gt == 0 or not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    tp = np.cumsum(np.asarray(is_tp, dtype=np.float64)[order])
    fp = np.cumsum(1.0 - np.asarray(is_tp, dtype=np.float64)[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - prev) * precision))


def paired_block_bootstrap(blocks_a, blocks_b, *, block_frames: int = 30,
                           n_resamples: int = 10000, alpha: float = 0.05,
                           seed: int = 0) -> BlockBootstrapResult:
    """Paired moving-block bootstrap of (AP_a - AP_b) within one sequence.

    `blocks_a[i]` and `blocks_b[i]` must be the two methods' results on the SAME block of
    the SAME video; indices are drawn jointly so a hard stretch of the flight that hurts
    both does not count as evidence for either.
    """
    if len(blocks_a) != len(blocks_b):
        raise ValueError(f"paired comparison needs equal block counts, got "
                         f"{len(blocks_a)} and {len(blocks_b)}")
    n = len(blocks_a)
    if n < 2:
        raise ValueError(f"{n} block(s) is not enough to resample; shorten --block-frames "
                         f"or accept that this sequence cannot carry an interval")

    full = list(range(n))
    sa, sb = _ap(full, blocks_a), _ap(full, blocks_b)
    observed = sa - sb

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = _ap(idx, blocks_a) - _ap(idx, blocks_b)

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    p = float(min(1.0, 2.0 * min(float(np.mean(diffs <= 0.0)),
                                 float(np.mean(diffs >= 0.0)))))
    return BlockBootstrapResult(observed=observed, lo=lo, hi=hi, p_value=p,
                                n_blocks=n, block_frames=block_frames,
                                n_resamples=n_resamples,
                                statistic_a=sa, statistic_b=sb)


__all__ = ["BlockBootstrapResult", "blocks_from_detections", "paired_block_bootstrap"]
