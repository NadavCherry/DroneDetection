"""Tests for the label-extent control in tools/make_datasets_v3.py.

The defect these guard is the single most consequential one in the project, and it was
invisible for seven rounds: `LABEL = 24.0` wrote a fixed 24 px square for every target
regardless of its real size, so the model learned to predict 24 px boxes, so the maximum
achievable IoU against a real annotation was capped at a median of 0.110 and COCO AP was
not low but *arithmetically impossible* (verified-measurements-2026-08.md §6b).

Two things are pinned:

1. `--label-px 0` really writes true extents. An earlier attempt at this flag would have
   written zero-size boxes, because `objs_at` discarded w and h before any emit site could
   see them -- a fix that looks complete and silently produces degenerate labels.
2. The default still writes fixed squares, so the round-1..7 datasets regenerate
   unchanged. A reproducibility guarantee is worth nothing if a later fix quietly voids it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

m = pytest.importorskip("make_datasets_v3", reason="tools/ needs numpy + opencv")

from dronedet.gt import GTObject, GroundTruth  # noqa: E402


@pytest.fixture(autouse=True)
def restore_label():
    """LABEL is module state; a test that changes it must not leak into the next."""
    before = m.LABEL
    yield
    m.LABEL = before


def test_default_is_the_fixed_square_that_rounds_1_to_7_used():
    assert m.LABEL == 24.0


def test_label_wh_returns_the_fixed_square_by_default():
    m.LABEL = 24.0
    assert m.label_wh(7.0, 5.0) == (24.0, 24.0)
    assert m.label_wh(200.0, 150.0) == (24.0, 24.0)


def test_label_wh_returns_true_extent_at_zero():
    m.LABEL = 0.0
    assert m.label_wh(7.0, 5.0) == (7.0, 5.0)
    assert m.label_wh(200.0, 150.0) == (200.0, 150.0)


def test_label_wh_never_returns_zero_sized_boxes_for_a_real_target():
    """The failure mode of a half-done fix: `--label-px 0` writing degenerate labels
    because the true extent was discarded upstream."""
    m.LABEL = 0.0
    w, h = m.label_wh(9.85, 5.98)
    assert w > 0 and h > 0


def _gt(cx=100.0, cy=200.0, w=9.85, h=5.98, bird=True):
    objects = {"far": GTObject(name="far", ignore=False, frames={0: (cx, cy, w, h)})}
    if bird:
        objects["bird"] = GTObject(name="bird", ignore=True, frames={0: (300.0, 400.0, 6.1, 5.2)})
    return GroundTruth(video="t.mp4", objects=objects, meta={})


def test_objs_at_carries_the_true_extent():
    """The information must survive to the emit sites. It used to be dropped here."""
    objs = m.objs_at(_gt(), {0: (0.0, 0.0)}, 0)
    assert objs, "expected at least the drone"
    for _cls, box in objs:
        assert len(box) == 4, "every object must be (cx, cy, w, h)"
    drone = next(b for c, b in objs if c == "drone")
    assert drone[2] == pytest.approx(9.85)
    assert drone[3] == pytest.approx(5.98)


def test_objs_at_applies_the_stabilisation_shift_to_the_centre_only():
    """The shift moves the target; it must not resize it."""
    objs = m.objs_at(_gt(), {0: (5.0, -3.0)}, 0)
    cx, cy, w, h = next(b for c, b in objs if c == "drone")
    assert (cx, cy) == pytest.approx((105.0, 197.0))
    assert (w, h) == pytest.approx((9.85, 5.98))


def test_birds_are_carried_with_their_own_extent_too():
    objs = m.objs_at(_gt(), {0: (0.0, 0.0)}, 0)
    bird = next(b for c, b in objs if c == "bird")
    assert bird[2] == pytest.approx(6.1) and bird[3] == pytest.approx(5.2)


def test_true_extents_vary_where_fixed_squares_cannot():
    """The property that makes an IoU number possible at all: real annotations have a
    distribution of sizes, and a constant destroys it."""
    m.LABEL = 24.0
    fixed = {m.label_wh(w, 6.0) for w in (4.0, 9.0, 15.0)}
    assert len(fixed) == 1
    m.LABEL = 0.0
    true = {m.label_wh(w, 6.0) for w in (4.0, 9.0, 15.0)}
    assert len(true) == 3
