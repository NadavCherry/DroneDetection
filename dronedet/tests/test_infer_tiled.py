"""Tiling geometry and duplicate merging for tools/infer_tiled.py.

Both are places where a quiet mistake produces a plausible number. A grid that misses a
strip of the frame loses every target in it and simply reports lower recall; a merge that
uses IoU instead of centre distance keeps both copies of a target seen in two overlapping
tiles and manufactures false positives out of the tiling.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location("infer_tiled", REPO / "tools/infer_tiled.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()
from dronedet.detections import Detection  # noqa: E402


# ------------------------------------------------------------------ tiling geometry

@pytest.mark.parametrize("w,h,tile,overlap", [
    (1920, 1080, 640, 128),
    (1920, 1080, 640, 0),
    (1280, 720, 640, 128),
    (640, 640, 640, 128),
    (700, 500, 640, 64),
])
def test_the_grid_covers_every_pixel(w, h, tile, overlap):
    """No uncovered strip. A target in a gap is invisible and costs recall silently."""
    origins = M.tile_origins(w, h, tile, overlap)
    covered_x, covered_y = set(), set()
    for x, y in origins:
        covered_x.update(range(x, min(x + tile, w)))
        covered_y.update(range(y, min(y + tile, h)))
    assert covered_x == set(range(w)), f"columns uncovered for {w}x{h}"
    assert covered_y == set(range(h)), f"rows uncovered for {w}x{h}"


@pytest.mark.parametrize("w,h,tile", [(1920, 1080, 640), (1280, 720, 640)])
def test_tiles_stay_inside_the_frame(w, h, tile):
    """The last row and column are pulled back, never padded.

    A padded tile puts a hard black edge into the image. On a 6 px target that edge is a
    strong gradient the detector never saw in training, where every tile was cut from
    real pixels.
    """
    for x, y in M.tile_origins(w, h, tile, 128):
        assert 0 <= x and x + tile <= w
        assert 0 <= y and y + tile <= h


def test_the_last_tile_is_flush_with_the_edge():
    origins = M.tile_origins(1920, 1080, 640, 128)
    assert max(x for x, _ in origins) == 1920 - 640
    assert max(y for _, y in origins) == 1080 - 640


def test_a_frame_smaller_than_a_tile_still_yields_one_origin():
    assert M.tile_origins(400, 300, 640, 128) == [(0, 0)]


# ------------------------------------------------------------------ duplicate merging

def _d(cx, cy, score, side=6.0):
    h = side / 2
    return Detection(cx - h, cy - h, cx + h, cy + h, score)


def test_the_same_target_seen_in_two_tiles_becomes_one_detection():
    """The overlap duplicates targets; that is the tiling's own artefact to clean up."""
    merged = M.merge_by_centre([_d(100, 100, 0.9), _d(102, 101, 0.7)], dist=6.0)
    assert len(merged) == 1
    assert merged[0].score == 0.9, "the stronger detection must survive"


def test_two_genuinely_separate_targets_are_both_kept():
    merged = M.merge_by_centre([_d(100, 100, 0.9), _d(140, 100, 0.8)], dist=6.0)
    assert len(merged) == 2


def test_merging_is_by_centre_not_iou():
    """Two 6 px boxes 4 px apart overlap very little -- IoU here is about 0.11, far under
    any usual NMS threshold, so an IoU-based merge would keep both and invent a false
    positive. Centre distance is the only rule that survives few-pixel boxes.
    """
    a, b = _d(100, 100, 0.9), _d(104, 100, 0.8)
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    iou = (ix * iy) / (a.area + b.area - ix * iy)
    assert iou < 0.3, f"fixture broken: IoU {iou:.2f} is not the low-overlap case"
    assert len(M.merge_by_centre([a, b], dist=6.0)) == 1


def test_merging_keeps_the_highest_score_regardless_of_input_order():
    for order in ([_d(50, 50, 0.4), _d(51, 50, 0.95)], [_d(51, 50, 0.95), _d(50, 50, 0.4)]):
        merged = M.merge_by_centre(order, dist=6.0)
        assert len(merged) == 1 and merged[0].score == 0.95


def test_an_empty_frame_merges_to_nothing():
    assert M.merge_by_centre([], dist=6.0) == []


# --------------------------------------------------- the stabilisation border artefact

def test_a_detection_on_the_stabilisation_border_is_rejected():
    """Undoing the pan can place a box outside the frame; that box is our own artefact.

    Stabilisation warps each frame onto a same-sized canvas, so the region the pan vacates
    is filled with a constant and carries a hard straight edge a detector fires on. Undoing
    the shift puts those boxes beyond the image. Measured on phantom05 before this check:
    detections at x = 2091 on a 1920 px frame -- 171 px into a border containing no scene.
    Ground truth is inside the frame, so every one is a false positive we manufactured.
    """
    assert not M.centre_in_frame(_d(2091, 500, 0.9), 1920, 1080)
    assert not M.centre_in_frame(_d(-30, 500, 0.9), 1920, 1080)
    assert not M.centre_in_frame(_d(900, 1120, 0.9), 1920, 1080)


def test_a_detection_inside_the_frame_survives():
    assert M.centre_in_frame(_d(960, 540, 0.9), 1920, 1080)
    assert M.centre_in_frame(_d(0.5, 0.5, 0.9), 1920, 1080)


def test_the_bound_is_the_centre_not_the_corner():
    """A real target near the edge has a box that pokes past it; that is not a reason to
    drop it. Only the centre leaving the frame means the detection belongs to the border.
    """
    d = _d(1918, 540, 0.9, side=10)          # centre inside, right edge at 1923
    assert d.x2 > 1920
    assert M.centre_in_frame(d, 1920, 1080)
