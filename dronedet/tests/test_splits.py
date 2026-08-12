"""Tests for the external-dataset splits.

These exist because of a real, shipped bug: `tools/make_dataset_external.py` defined the
published ARD-MAV test list at line 41 and then `combined_splits()` ignored it 168 lines
later, re-splitting by position. Rounds 5-7 trained on most of the official test videos,
so their ARD-MAV numbers cannot be compared with any published one. Nothing caught it,
because nothing tested it.

The split is a *claim about what a number means*. It gets a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

# The builder imports cv2/numpy at module scope but nothing heavier, so this is CI-safe.
make_dataset_external = pytest.importorskip(
    "make_dataset_external", reason="tools/ not importable (needs opencv + numpy)")

OFFICIAL_TEST = make_dataset_external.ARD_TEST_IDS
OFFICIAL_VAL = make_dataset_external.ARD_VAL_IDS

# The 60 ARD-MAV video ids, so these tests run without the 14.6 GB download present.
ALL_ARD = sorted({f"phantom{n:02d}" for n in range(1, 91)} | set(OFFICIAL_TEST) | set(OFFICIAL_VAL))


@pytest.fixture
def splits(monkeypatch):
    """combined_splits() without needing the dataset on disk."""
    monkeypatch.setattr(make_dataset_external, "_ard_all", lambda: list(ALL_ARD))
    monkeypatch.setattr(make_dataset_external, "NPS_ANN", Path("/nonexistent"))
    return make_dataset_external.combined_splits


def test_official_ard_test_list_matches_the_published_one():
    """Guo et al.'s 15 test videos. If this list ever changes, every ARD-MAV number
    in the repo silently stops meaning what its table says it means."""
    assert sorted(OFFICIAL_TEST) == sorted(
        f"phantom{n:02d}" for n in
        (5, 8, 9, 10, 19, 30, 41, 43, 46, 47, 58, 63, 65, 70, 86))
    assert len(OFFICIAL_TEST) == 15


def test_combined_split_holds_out_every_official_test_video(splits):
    sp = splits()["ardmav"]
    for vid in OFFICIAL_TEST:
        assert vid in sp["test"], f"{vid} is an official test video but is not in test"
        assert vid not in sp["train"], f"LEAK: official test video {vid} is in train"
        assert vid not in sp["val"], f"LEAK: official test video {vid} is in val"


def test_combined_split_partitions_cleanly(splits):
    sp = splits()["ardmav"]
    tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(ALL_ARD)


def test_val_videos_are_the_declared_ones_and_are_not_trained_on(splits):
    sp = splits()["ardmav"]
    assert sorted(sp["val"]) == sorted(OFFICIAL_VAL)
    assert not (set(sp["train"]) & set(OFFICIAL_VAL))


def test_legacy_split_is_reachable_and_is_the_leaky_one(splits):
    """The old behaviour must stay available to regenerate round-5..7 artifacts --
    and must remain provably leaky, so nobody mistakes it for the official split."""
    legacy = splits(legacy=True)["ardmav"]
    leaked = [v for v in OFFICIAL_TEST if v in legacy["train"] or v in legacy["val"]]
    assert leaked, "legacy split is supposed to leak; if it stopped, this test is stale"
    assert len(leaked) >= 10, f"expected most of the 15 official test videos to leak, got {len(leaked)}"
    assert len(legacy["test"]) < len(OFFICIAL_TEST)


def test_user_videos_keep_10_06_out_of_training(splits):
    """10_06 is the held-out test video for the repo's own data (CLAUDE.md)."""
    user = splits()["user"]
    assert user["test"] == ["10_06"]
    assert "10_06" not in user["train"] and "10_06" not in user["val"]
