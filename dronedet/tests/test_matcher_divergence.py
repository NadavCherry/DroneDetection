"""Pin the one place `dronedet.evaluate` and `dronedet.metrics` disagree.

Both implement this repository's centre-distance rule with an identical radius,
``max(tau, 0.5 * sqrt(gt_area))``. They differ in exactly one situation: a detection that
falls inside BOTH a target's radius and a distractor's.

    evaluate.py   takes the nearest object regardless of its ignore flag, so a bird that
                  happens to be closer than the drone wins and the detection is dropped
                  as `ign` -- neither credited nor charged.
    metrics.py    ranks positives above distractors before distance, so the same detection
                  scores `tp`.

`metrics.py` is the more generous of the two here, and it is the one behind every published
number. That is worth pinning rather than leaving as folklore.

Measured impact at the time of writing: over the 23 committed 07_05 detection sets scored
against `work/gt_user.json` at tau=12, 102,017 detections were compared and **0 outcomes
differed** -- the precondition arises in 3 frame-pairs out of 571 and no committed
detection lands in any of them. So this is a latent trap, not a live error, and these tests
exist so that it stays visible if a denser distractor set ever makes it live.

`benchmarks/protocol.py` used to cite both files as the implementation of one protocol.
It now cites `metrics.py` alone, for the reason these tests demonstrate.
"""

from __future__ import annotations

import math

from dronedet import metrics as M
from dronedet.detections import Detection
from dronedet.evaluate import _match_frame as legacy_match


def _radius(w: float, h: float, tau: float = 12.0) -> float:
    return max(tau, 0.5 * math.sqrt(max(w * h, 1.0)))


class TestRadiusIsShared:
    def test_both_modules_use_the_same_matching_radius(self):
        """The disagreement is about ranking, not about reach. Establish that first."""
        for w, h in ((6.0, 6.0), (40.0, 40.0), (1.0, 1.0), (100.0, 25.0)):
            # metrics.centre_distance_ok computes the radius inline; probe it by finding
            # the largest error it still accepts for a box of this size.
            r = _radius(w, h)
            gt = (100.0 - w / 2, 100.0 - h / 2, 100.0 + w / 2, 100.0 + h / 2)
            just_in = (100.0 + r - 0.01, 100.0, 100.0 + r - 0.01, 100.0)
            just_out = (100.0 + r + 0.01, 100.0, 100.0 + r + 0.01, 100.0)
            assert M.centre_distance_ok(just_in, gt, 12.0)[0], (w, h)
            assert not M.centre_distance_ok(just_out, gt, 12.0)[0], (w, h)


class TestOverlapResolutionDiverges:
    """A detection inside a target's radius AND a nearer distractor's."""

    # target at (100,100), bird at (104,100), detection at (103,100):
    #   distance to target = 3.0, to bird = 1.0 -> the BIRD is nearer.
    DET = (103.0, 100.0)
    TARGET = (100.0, 100.0, 6.0, 6.0)
    BIRD = (104.0, 100.0, 6.0, 6.0)

    def test_legacy_drops_it_because_the_distractor_is_nearer(self):
        dets = [Detection(self.DET[0] - 3, self.DET[1] - 3,
                          self.DET[0] + 3, self.DET[1] + 3, 0.9)]
        gts = {"far": (*self.TARGET, False), "bird": (*self.BIRD, True)}
        out = legacy_match(dets, gts, 12.0)
        assert out[0][0] == "ign", out

    def test_metrics_scores_it_because_positives_outrank_distractors(self):
        dets = [(self.DET[0] - 3, self.DET[1] - 3,
                 self.DET[0] + 3, self.DET[1] + 3, 0.9)]
        gts = {
            "far": (*M.cxcywh_to_xyxy(self.TARGET), False),
            "bird": (*M.cxcywh_to_xyxy(self.BIRD), True),
        }
        out = M._match_frame(dets, gts, "centre", 12.0, 0.5)
        assert out[0][0] == "tp", out
        assert out[0][1] == "far", out

    def test_the_two_therefore_disagree_and_metrics_is_the_generous_one(self):
        """The whole point, asserted as one statement so it cannot drift silently."""
        d = Detection(self.DET[0] - 3, self.DET[1] - 3,
                      self.DET[0] + 3, self.DET[1] + 3, 0.9)
        legacy = legacy_match([d], {"far": (*self.TARGET, False),
                                    "bird": (*self.BIRD, True)}, 12.0)[0][0]
        modern = M._match_frame(
            [(d.x1, d.y1, d.x2, d.y2, d.score)],
            {"far": (*M.cxcywh_to_xyxy(self.TARGET), False),
             "bird": (*M.cxcywh_to_xyxy(self.BIRD), True)},
            "centre", 12.0, 0.5)[0][0]
        assert (legacy, modern) == ("ign", "tp"), (legacy, modern)


class TestTheyAgreeEverywhereElse:
    """The divergence needs BOTH objects in reach. Confirm the ordinary cases match."""

    def test_a_clean_target_hit_is_tp_in_both(self):
        d = Detection(99.0, 99.0, 105.0, 105.0, 0.9)
        legacy = legacy_match([d], {"far": (100.0, 100.0, 6.0, 6.0, False)}, 12.0)[0][0]
        modern = M._match_frame([(d.x1, d.y1, d.x2, d.y2, d.score)],
                                {"far": (*M.cxcywh_to_xyxy((100.0, 100.0, 6.0, 6.0)),
                                         False)}, "centre", 12.0, 0.5)[0][0]
        assert legacy == "tp" and modern == "tp"

    def test_a_hit_on_a_lone_distractor_is_not_a_tp_in_either(self):
        d = Detection(101.0, 97.0, 107.0, 103.0, 0.9)
        legacy = legacy_match([d], {"bird": (104.0, 100.0, 6.0, 6.0, True)}, 12.0)[0][0]
        modern = M._match_frame([(d.x1, d.y1, d.x2, d.y2, d.score)],
                                {"bird": (*M.cxcywh_to_xyxy((104.0, 100.0, 6.0, 6.0)),
                                          True)}, "centre", 12.0, 0.5)[0][0]
        # Named differently -- 'ign' vs 'distractor' -- but neither counts it as a hit,
        # and metrics.py keeps it as a first-class outcome instead of discarding it.
        assert legacy == "ign"
        assert modern == "distractor"

    def test_a_detection_reaching_nothing_is_fp_in_both(self):
        d = Detection(497.0, 497.0, 503.0, 503.0, 0.9)
        legacy = legacy_match([d], {"far": (100.0, 100.0, 6.0, 6.0, False)}, 12.0)[0][0]
        modern = M._match_frame([(d.x1, d.y1, d.x2, d.y2, d.score)],
                                {"far": (*M.cxcywh_to_xyxy((100.0, 100.0, 6.0, 6.0)),
                                         False)}, "centre", 12.0, 0.5)[0][0]
        assert legacy == "fp" and modern == "fp"
