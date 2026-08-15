"""Tests for the YOLOMG dataset builder.

The failure this file exists to prevent: YOLOMG pairs `train.txt` line N with `train2.txt`
line N. Nothing in its code checks that line N of each refers to the same frame. Drop one
mask and every later pair is off by one -- the model trains on one frame's appearance
against another frame's motion, converges to something mediocre, and reports no error at
all. We would then publish a win over a baseline we broke ourselves.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from tools.sota.build_yolomg import (MASK_DIR, _frames_needed, _write_pair,
                                     write_lists)


def _fake_build(root, n=6, drop_mask=None):
    """All three splits get frames: `write_lists` refuses a build with an empty split, and
    a real build always has all three."""
    for sub in ("images", MASK_DIR, "labels"):
        for split in ("train", "val", "test"):
            (root / sub / split).mkdir(parents=True, exist_ok=True)
    frame = np.full((64, 64, 3), 120, dtype=np.uint8)
    for split, k in (("train", n), ("val", 2), ("test", 2)):
        for i in range(k):
            _write_pair(root, split, f"clip_{i:06d}", frame,
                        np.zeros((64, 64)), [[10, 10, 20, 20]], 64, 64)
    if drop_mask is not None:
        (root / MASK_DIR / "train" / f"clip_{drop_mask:06d}.jpg").unlink()
    return root


def test_lists_are_line_aligned(tmp_path):
    root = _fake_build(tmp_path / "ds")
    counts = write_lists(root)

    assert counts["train"] == 6
    a = (root / "train.txt").read_text(encoding="utf-8").splitlines()
    b = (root / "train2.txt").read_text(encoding="utf-8").splitlines()
    assert [p.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for p in a] == \
           [p.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for p in b]
    assert "/images/" in a[0].replace("\\", "/")
    # `images2`, not `mask`: YOLOMG's img2label_paths2 only rewrites that exact substring.
    assert "/images2/" in b[0].replace("\\", "/")


def test_a_missing_mask_is_refused_rather_than_silently_shifting_the_pairing(tmp_path):
    """The whole point. Upstream would happily train on the shifted pairing."""
    root = _fake_build(tmp_path / "ds", drop_mask=3)
    with pytest.raises(RuntimeError, match="have no mask"):
        write_lists(root)


def test_labels_are_normalised_yolo_and_survive_the_round_trip(tmp_path):
    root = tmp_path / "ds"
    for sub in ("images", MASK_DIR, "labels"):
        (root / sub / "train").mkdir(parents=True, exist_ok=True)
    frame = np.full((100, 200, 3), 90, dtype=np.uint8)

    _write_pair(root, "train", "v_000000", frame, np.zeros((100, 200)),
                [[40, 30, 60, 50]], 200, 100)

    cls, cx, cy, w, h = (root / "labels" / "train" / "v_000000.txt").read_text(
        encoding="utf-8").split()
    assert cls == "0"
    assert float(cx) == pytest.approx(50 / 200)
    assert float(cy) == pytest.approx(40 / 100)
    assert float(w) == pytest.approx(20 / 200)
    assert float(h) == pytest.approx(20 / 100)


def test_degenerate_boxes_are_dropped_not_written_as_zero_area(tmp_path):
    root = tmp_path / "ds"
    for sub in ("images", MASK_DIR, "labels"):
        (root / sub / "train").mkdir(parents=True, exist_ok=True)
    frame = np.full((100, 100, 3), 90, dtype=np.uint8)

    _write_pair(root, "train", "v_000000", frame, np.zeros((100, 100)),
                [[10, 10, 10, 20], [30, 30, 40, 40]], 100, 100)

    lines = (root / "labels" / "train" / "v_000000.txt").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == 1, "the zero-width box should have been dropped"


def test_an_empty_build_is_refused_rather_than_declared_successful(tmp_path):
    """This one really happened: `Videos` instead of `videos` matched nothing on a
    case-sensitive filesystem, and the build reported exit 0 with a BUILD.json and three
    perfectly line-aligned lists of zero pairs. Every downstream check passed."""
    root = tmp_path / "ds"
    for sub in ("images", MASK_DIR, "labels"):
        for split in ("train", "val", "test"):
            (root / sub / split).mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="refusing to declare an empty dataset"):
        write_lists(root)


def test_taps_requested_match_the_documented_aperture():
    """dt=2 means the mask for frame t reads t-2 and t+2 -- five frames of span."""
    assert sorted(_frames_needed([100], 2)) == [98, 100, 102]
    assert sorted(_frames_needed([100], 6)) == [94, 100, 106]
