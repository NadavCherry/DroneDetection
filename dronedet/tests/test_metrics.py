"""Tests for dronedet.metrics.

The coordinate convention is the thing worth guarding: ground truth is stored
``(cx, cy, w, h)`` while detections are ``xyxy``. Confusing the two costs about
w/2 px of phantom localisation error, which on a 6 px target is larger than the
signal being measured -- and it is silent, because every number still looks
plausible. `test_gt_is_centre_convention` is the regression test for exactly that.
"""

from __future__ import annotations

import math

import pytest

from dronedet import metrics as M
from dronedet.detections import Detection, DetectionSet
from dronedet.gt import GroundTruth, GTObject


def make_gt(objects, meta=None):
    """objects: {name: (ignore, {frame: (cx, cy, w, h)})}"""
    return GroundTruth(
        video="t.mp4",
        objects={n: GTObject(name=n, ignore=ig, frames=fr) for n, (ig, fr) in objects.items()},
        meta=meta or {},
    )


def make_dets(frames):
    """frames: {frame: [(x1, y1, x2, y2, score), ...]}"""
    ds = DetectionSet(video="t.mp4", method="test")
    for f, dl in frames.items():
        ds.add(f, [Detection(*d) for d in dl])
    return ds


# ---------------------------------------------------------------- geometry
def test_cxcywh_to_xyxy():
    assert M.cxcywh_to_xyxy((100.0, 50.0, 8.0, 8.0)) == (96.0, 46.0, 104.0, 54.0)


def test_iou_identical_and_disjoint():
    b = (0.0, 0.0, 10.0, 10.0)
    assert M.iou(b, b) == pytest.approx(1.0)
    assert M.iou(b, (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_iou_half_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    assert M.iou(a, b) == pytest.approx(50 / 150)


def test_centre_distance_uses_tau_floor_for_tiny_boxes():
    # a 6 px box has 0.5*sqrt(area) = 3, so tau=12 is the binding radius
    gt = M.cxcywh_to_xyxy((100.0, 100.0, 6.0, 6.0))
    det = M.cxcywh_to_xyxy((110.0, 100.0, 6.0, 6.0))
    ok, err = M.centre_distance_ok(det, gt, tau=12.0)
    assert ok and err == pytest.approx(10.0)
    far = M.cxcywh_to_xyxy((113.0, 100.0, 6.0, 6.0))
    assert not M.centre_distance_ok(far, gt, tau=12.0)[0]


def test_centre_distance_radius_grows_with_large_boxes():
    # a 100 px box gets radius 50, well past tau
    gt = M.cxcywh_to_xyxy((100.0, 100.0, 100.0, 100.0))
    det = M.cxcywh_to_xyxy((140.0, 100.0, 10.0, 10.0))
    assert M.centre_distance_ok(det, gt, tau=12.0)[0]


def test_size_bin_boundaries():
    assert M.size_bin(4.0, 4.0) == "very-tiny"     # sqrt(area) = 4
    assert M.size_bin(8.0, 8.0) == "tiny"          # 8 is the lower edge of tiny
    assert M.size_bin(20.0, 20.0) == "small"
    assert M.size_bin(64.0, 64.0) == "medium"


# ---------------------------------------------------------------- conventions
def test_gt_is_centre_convention():
    """A detection centred exactly on the GT centre must have zero error.

    If GT were read as a top-left corner this would report w/2*sqrt(2) px instead.
    """
    gt = make_gt({"far": (False, {0: (500.0, 300.0, 8.0, 8.0)})})
    ds = make_dets({0: [(488.0, 288.0, 512.0, 312.0, 0.9)]})   # 24 px box, same centre
    ev = M.evaluate(gt, ds, rule="centre", tau=12.0)
    assert [r.outcome for r in ev.records] == ["tp"]
    assert ev.records[0].error == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- outcomes
def test_distractor_is_neither_tp_nor_fp():
    gt = make_gt({
        "far": (False, {0: (100.0, 100.0, 8.0, 8.0)}),
        "bird": (True, {0: (400.0, 400.0, 6.0, 6.0)}),
    })
    ds = make_dets({0: [(96.0, 96.0, 104.0, 104.0, 0.9),      # on the drone
                        (397.0, 397.0, 403.0, 403.0, 0.8)]})  # on the bird
    ev = M.evaluate(gt, ds)
    outcomes = sorted(r.outcome for r in ev.records)
    assert outcomes == ["distractor", "tp"]
    s = M.summarise(ev, 0.0)
    assert s.precision == 1.0          # the bird hit is not charged as a FP...
    assert s.distractor_hits == 1      # ...but it is counted and reportable
    assert s.distractor_instances == 1
    assert s.fp_per_frame == 0.0


def test_confuser_hits_separate_birds_from_benign_distractors():
    """`near` is the same drone landed and huge -- hitting it is correct. A bird hit is
    the failure the pipeline exists to prevent. One total would hide the result."""
    gt = make_gt({
        "far": (False, {0: (100.0, 100.0, 8.0, 8.0)}),
        "near": (True, {0: (400.0, 400.0, 140.0, 140.0)}),
        "bird": (True, {0: (800.0, 800.0, 6.0, 6.0)}),
    })
    ds = make_dets({0: [(96.0, 96.0, 104.0, 104.0, 0.9),
                        (395.0, 395.0, 405.0, 405.0, 0.8),     # on the near drone
                        (797.0, 797.0, 803.0, 803.0, 0.7)]})   # on the bird
    s = M.summarise(M.evaluate(gt, ds, targets={"far"}), 0.0)
    assert s.distractor_hits == 2
    assert s.distractor_hits_by_object == {"near": 1, "bird": 1}
    assert s.confuser_hits(("bird",)) == (1, 1)
    assert s.confuser_hits(("near",)) == (1, 1)


def test_confuser_hits_are_zero_when_the_bird_is_rejected():
    gt = make_gt({
        "far": (False, {0: (100.0, 100.0, 8.0, 8.0)}),
        "bird": (True, {0: (800.0, 800.0, 6.0, 6.0)}),
    })
    ds = make_dets({0: [(96.0, 96.0, 104.0, 104.0, 0.9)]})
    s = M.summarise(M.evaluate(gt, ds, targets={"far"}), 0.0)
    assert s.confuser_hits(("bird",)) == (0, 1)     # 0 hits, 1 bird instance on offer


def test_confuser_hits_respect_the_threshold():
    gt = make_gt({
        "far": (False, {0: (100.0, 100.0, 8.0, 8.0)}),
        "bird": (True, {0: (800.0, 800.0, 6.0, 6.0)}),
    })
    ds = make_dets({0: [(96.0, 96.0, 104.0, 104.0, 0.9),
                        (797.0, 797.0, 803.0, 803.0, 0.3)]})
    assert M.summarise(M.evaluate(gt, ds, targets={"far"}), 0.5).confuser_hits()[0] == 0
    assert M.summarise(M.evaluate(gt, ds, targets={"far"}), 0.1).confuser_hits()[0] == 1


def test_detection_on_nothing_is_a_false_positive():
    gt = make_gt({"far": (False, {0: (100.0, 100.0, 8.0, 8.0)})})
    ds = make_dets({0: [(700.0, 700.0, 710.0, 710.0, 0.9)]})
    ev = M.evaluate(gt, ds)
    assert [r.outcome for r in ev.records] == ["fp"]
    assert M.summarise(ev, 0.0).precision == 0.0


def test_positive_outranks_distractor_when_both_match():
    """A drone sitting near a bird must be credited to the drone, not consumed
    by the distractor -- otherwise recall silently drops wherever clutter is dense."""
    gt = make_gt({
        "far": (False, {0: (100.0, 100.0, 8.0, 8.0)}),
        "bird": (True, {0: (104.0, 100.0, 6.0, 6.0)}),
    })
    ds = make_dets({0: [(96.0, 96.0, 104.0, 104.0, 0.9)]})
    ev = M.evaluate(gt, ds)
    assert [r.outcome for r in ev.records] == ["tp"]


def test_each_gt_claimed_once_so_duplicates_are_false_positives():
    gt = make_gt({"far": (False, {0: (100.0, 100.0, 8.0, 8.0)})})
    ds = make_dets({0: [(96.0, 96.0, 104.0, 104.0, 0.9),
                        (97.0, 97.0, 105.0, 105.0, 0.8)]})
    ev = M.evaluate(gt, ds)
    assert sorted(r.outcome for r in ev.records) == ["fp", "tp"]


def test_excluded_frames_are_skipped():
    gt = make_gt({"far": (False, {0: (100.0, 100.0, 8.0, 8.0)},)},
                 meta={"exclude_frames": [0]})
    ds = make_dets({0: [(700.0, 700.0, 710.0, 710.0, 0.9)]})
    ev = M.evaluate(gt, ds)
    assert ev.n_frames == 0 and ev.records == []


def test_targets_argument_demotes_other_objects_to_distractors():
    gt = make_gt({
        "far": (False, {0: (100.0, 100.0, 8.0, 8.0)}),
        "near": (False, {0: (400.0, 400.0, 100.0, 100.0)}),
    })
    ds = make_dets({0: [(390.0, 390.0, 410.0, 410.0, 0.9)]})
    ev = M.evaluate(gt, ds, targets={"far"})
    assert [r.outcome for r in ev.records] == ["distractor"]
    assert ev.n_gt == 1                 # only 'far' counts toward recall


# ---------------------------------------------------------------- AP
def test_ap_is_one_for_perfect_detections():
    gt = make_gt({"far": (False, {f: (100.0, 100.0, 8.0, 8.0) for f in range(10)})})
    ds = make_dets({f: [(96.0, 96.0, 104.0, 104.0, 0.9)] for f in range(10)})
    ev = M.evaluate(gt, ds)
    assert M.average_precision(ev.records, ev.n_gt) == pytest.approx(1.0)


def test_ap_is_zero_when_nothing_is_found():
    gt = make_gt({"far": (False, {f: (100.0, 100.0, 8.0, 8.0) for f in range(10)})})
    ev = M.evaluate(gt, make_dets({f: [] for f in range(10)}))
    assert M.average_precision(ev.records, ev.n_gt) == 0.0


def test_ap_penalises_high_scoring_false_positives():
    """Same recall, but a FP ranked above the TP must score lower."""
    gt = make_gt({"far": (False, {0: (100.0, 100.0, 8.0, 8.0)})})
    clean = M.evaluate(gt, make_dets({0: [(96.0, 96.0, 104.0, 104.0, 0.9)]}))
    dirty = M.evaluate(gt, make_dets({0: [(96.0, 96.0, 104.0, 104.0, 0.5),
                                          (700.0, 700.0, 710.0, 710.0, 0.99)]}))
    assert M.average_precision(clean.records, clean.n_gt) > \
        M.average_precision(dirty.records, dirty.n_gt)


def test_coco_ap_zero_when_boxes_have_wrong_extent():
    """The repo's shipped situation: 8 px GT, 24 px predictions, centres aligned.

    Max achievable IoU is (8*8)/(24*24) = 0.111, so every COCO threshold misses and
    AP is structurally 0 no matter how good the localisation is. This test exists to
    document that the zero is a box-extent problem, not a detection failure.
    """
    gt = make_gt({"far": (False, {f: (100.0, 100.0, 8.0, 8.0) for f in range(5)})})
    ds = make_dets({f: [(88.0, 88.0, 112.0, 112.0, 0.9)] for f in range(5)})
    assert M.coco_ap(gt, ds)["AP50"] == 0.0
    ev = M.evaluate(gt, ds, rule="centre")               # ...while centre-distance
    assert M.average_precision(ev.records, ev.n_gt) == pytest.approx(1.0)  # is perfect


def test_coco_ap_one_for_exact_boxes():
    gt = make_gt({"far": (False, {f: (100.0, 100.0, 20.0, 20.0) for f in range(5)})})
    ds = make_dets({f: [(90.0, 90.0, 110.0, 110.0, 0.9)] for f in range(5)})
    c = M.coco_ap(gt, ds)
    assert c["AP"] == pytest.approx(1.0) and c["AP50"] == pytest.approx(1.0)


def test_ap_by_size_charges_false_positives_to_every_bin():
    """A method must not look strong on very-tiny targets by flooding the frame."""
    gt = make_gt({
        "a": (False, {0: (100.0, 100.0, 4.0, 4.0)}),     # very-tiny
        "b": (False, {0: (300.0, 300.0, 20.0, 20.0)}),   # small
    })
    ds = make_dets({0: [(98.0, 98.0, 102.0, 102.0, 0.9),
                        (290.0, 290.0, 310.0, 310.0, 0.9),
                        (700.0, 700.0, 710.0, 710.0, 0.95)]})   # top-ranked FP
    by = M.ap_by_size(M.evaluate(gt, ds))
    assert by["very-tiny"] < 1.0 and by["small"] < 1.0


# ---------------------------------------------------------------- thresholds
def test_pick_threshold_separates_tp_from_fp():
    gt = make_gt({"far": (False, {f: (100.0, 100.0, 8.0, 8.0) for f in range(5)})})
    frames = {f: [(96.0, 96.0, 104.0, 104.0, 0.9),
                  (700.0, 700.0, 710.0, 710.0, 0.2)] for f in range(5)}
    ev = M.evaluate(gt, make_dets(frames))
    thr = M.pick_threshold(ev)
    s = M.summarise(ev, thr)
    assert s.precision == 1.0 and s.recall == 1.0 and s.f1 == pytest.approx(1.0)


def test_summarise_respects_a_threshold_from_elsewhere():
    """The point of splitting pick_threshold from summarise: a val-chosen threshold
    may be wrong on test, and the report must show that rather than hide it."""
    gt = make_gt({"far": (False, {f: (100.0, 100.0, 8.0, 8.0) for f in range(5)})})
    ev = M.evaluate(gt, make_dets({f: [(96.0, 96.0, 104.0, 104.0, 0.4)] for f in range(5)}))
    assert M.summarise(ev, 0.9).recall == 0.0     # threshold too high for this set
    assert M.summarise(ev, 0.1).recall == 1.0


# ---------------------------------------------------------------- bootstrap
def test_bootstrap_ci_brackets_the_point_estimate():
    gt = make_gt({"far": (False, {f: (100.0, 100.0, 8.0, 8.0) for f in range(200)})})
    frames = {f: [(96.0, 96.0, 104.0, 104.0, 0.9)] for f in range(200)}
    for f in range(0, 200, 10):
        frames[f] = [(700.0, 700.0, 710.0, 710.0, 0.95)]     # a periodic miss + FP
    ev = M.evaluate(gt, make_dets(frames))
    ap = M.average_precision(ev.records, ev.n_gt)
    lo, hi = M.bootstrap_ci(ev, block=20, n_resamples=200, seed=1)
    assert lo <= ap <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed():
    gt = make_gt({"far": (False, {f: (100.0, 100.0, 8.0, 8.0) for f in range(100)})})
    frames = {f: [(96.0, 96.0, 104.0, 104.0, 0.9)] for f in range(100)}
    frames[7] = [(700.0, 700.0, 710.0, 710.0, 0.99)]
    ev = M.evaluate(gt, make_dets(frames))
    a = M.bootstrap_ci(ev, block=10, n_resamples=100, seed=3)
    b = M.bootstrap_ci(ev, block=10, n_resamples=100, seed=3)
    assert a == b


def test_evaluate_rejects_an_unknown_rule():
    gt = make_gt({"far": (False, {0: (1.0, 1.0, 2.0, 2.0)})})
    with pytest.raises(ValueError, match="rule"):
        M.evaluate(gt, make_dets({0: []}), rule="l2")
