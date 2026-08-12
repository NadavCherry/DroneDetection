"""Tests for tools/evaluate.py and tools/compare.py.

Two behaviours are load-bearing and get the most attention here:

* **A sequence with no detections is scored as a total miss, not skipped.** Skipping it
  would quietly raise the number, which is the kind of silent favour that makes a
  benchmark result untrustworthy.
* **A protocol mismatch blocks the comparison.** `tools/compare.py` must refuse to print
  a difference between a centre-distance AP on one clip and a published IoU AP on a full
  split -- the exact comparison this project used to make.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import compare as C  # noqa: E402
import evaluate as E  # noqa: E402
from benchmarks.scorecard import Scorecard, SequenceResult, pooled_ap  # noqa: E402


# ------------------------------------------------------------------ fixtures
def write_gt(path: Path, name: str, n_frames: int, *, birds: int = 0) -> None:
    """GT boxes are (cx, cy, w, h) -- centre format."""
    objects = {"far": {"ignore": False,
                       "frames": {str(f): [100.0 + f, 100.0, 8.0, 8.0] for f in range(n_frames)}}}
    if birds:
        objects["bird"] = {"ignore": True,
                           "frames": {str(f): [500.0, 500.0, 6.0, 6.0] for f in range(birds)}}
    (path / f"{name}.json").write_text(json.dumps(
        {"video": f"{name}.mp4", "meta": {}, "objects": objects}), encoding="utf-8")


def write_dets(path: Path, name: str, n_frames: int, *, hit: bool = True,
               bird_hits: int = 0) -> None:
    """Detections are xyxy."""
    frames = {}
    for f in range(n_frames):
        dl = []
        if hit:
            cx = 100.0 + f
            dl.append([cx - 4, 96.0, cx + 4, 104.0, 0.9, "drone"])
        if f < bird_hits:
            dl.append([497.0, 497.0, 503.0, 503.0, 0.8, "drone"])
        frames[str(f)] = dl
    (path / f"{name}.json").write_text(json.dumps(
        {"video": f"{name}.mp4", "method": "test", "meta": {}, "frames": frames}), encoding="utf-8")


@pytest.fixture
def bench(tmp_path):
    gt, det = tmp_path / "gt", tmp_path / "det"
    gt.mkdir(); det.mkdir()
    return gt, det


# ------------------------------------------------------------------ evaluate
def test_perfect_detections_score_one(bench):
    gt, det = bench
    write_gt(gt, "s01", 20)
    write_dets(det, "s01", 20)
    card = E.build_scorecard("ardmav", "m", gt, det, protocol_key="specklock-centre",
                             targets={"far"})
    assert card.n_sequences == 1
    assert pooled_ap(card.sequences) == pytest.approx(1.0)


def test_a_missing_detection_file_is_a_total_miss_not_a_skip(bench):
    """The property that stops a benchmark quietly flattering itself."""
    gt, det = bench
    write_gt(gt, "s01", 20)
    write_gt(gt, "s02", 20)
    write_dets(det, "s01", 20)          # s02 has no detections at all
    card = E.build_scorecard("ardmav", "m", gt, det, protocol_key="specklock-centre",
                             targets={"far"})
    assert card.n_sequences == 2, "the missing sequence must still appear"
    assert card.n_gt == 40, "its ground truth must still count toward recall"
    assert pooled_ap(card.sequences) == pytest.approx(0.5, abs=0.02)


def test_bird_hits_are_recorded_as_distractors_not_false_positives(bench):
    gt, det = bench
    write_gt(gt, "s01", 20, birds=20)
    write_dets(det, "s01", 20, bird_hits=5)
    card = E.build_scorecard("ardmav", "m", gt, det, protocol_key="specklock-centre",
                             targets={"far"})
    hits, total = card.distractor_hits(0.0, ("bird",))
    assert hits == 5 and total == 20
    assert pooled_ap(card.sequences) == pytest.approx(1.0), \
        "a bird hit must not be charged as a false positive against the drone AP"


def test_conditions_travel_into_the_scorecard(bench):
    gt, det = bench
    write_gt(gt, "s01", 10); write_dets(det, "s01", 10)
    write_gt(gt, "s02", 10); write_dets(det, "s02", 10)
    card = E.build_scorecard("ardmav", "m", gt, det, protocol_key="specklock-centre",
                             targets={"far"},
                             conditions_map={"s01": ("night",), "s02": ("clear",)})
    assert card.conditions_present() == ["clear", "night"]
    assert card.with_condition("night").n_sequences == 1


def test_official_split_filter_selects_only_published_test_sequences(bench):
    gt, det = bench
    for name in ("phantom05", "phantom16"):     # 05 is official test, 16 is not
        write_gt(gt, name, 10)
        write_dets(det, name, 10)
    card = E.build_scorecard("ardmav", "m", gt, det, protocol_key="ardmav-official",
                             targets={"far"}, only={"phantom05"})
    assert [s.sequence for s in card.sequences] == ["phantom05"]


def test_scorecard_records_provenance(bench):
    gt, det = bench
    write_gt(gt, "s01", 5); write_dets(det, "s01", 5)
    card = E.build_scorecard("ardmav", "m", gt, det, protocol_key="specklock-centre",
                             targets={"far"}, seed=7)
    assert card.created and card.command
    assert card.seed == 7
    assert isinstance(card.git_dirty, bool)


def test_target_px_median_is_measured_from_the_ground_truth(bench):
    gt, det = bench
    write_gt(gt, "s01", 10); write_dets(det, "s01", 10)
    card = E.build_scorecard("ardmav", "m", gt, det, protocol_key="specklock-centre",
                             targets={"far"})
    assert card.sequences[0].target_px_median == pytest.approx(8.0)


# ------------------------------------------------------------------ compare
def _card(model, quality, n_seq=6, protocol="ardmav-official", split="official-test-15",
          bird_hits=0, seed=0):
    import random
    r = random.Random(seed)
    seqs = []
    for i in range(n_seq):
        n_gt = 50
        dets = [(r.uniform(0.6, 0.99), "tp") for _ in range(int(n_gt * quality))]
        dets += [(r.uniform(0.5, 0.9), "distractor:bird") for _ in range(bird_hits)]
        seqs.append(SequenceResult(sequence=f"s{i:02d}", n_gt=n_gt, n_frames=100,
                                   conditions=["night"] if i % 2 else ["clear"],
                                   detections=dets, distractor_instances={"bird": 100}))
    return Scorecard(model=model, dataset_key="ardmav", protocol_key=protocol,
                     split=split, sequences=seqs)


def test_compare_pair_detects_a_real_improvement():
    r = C.compare_pair(_card("better", 0.9, seed=1), _card("worse", 0.6, seed=1),
                       n_resamples=500, seed=0)
    assert r["diff"] > 0 and r["significant"]
    assert r["wins_a"] > r["wins_b"]


def test_compare_pair_reports_no_difference_for_identical_models():
    r = C.compare_pair(_card("a", 0.8, seed=3), _card("b", 0.8, seed=3),
                       n_resamples=500, seed=0)
    assert r["diff"] == pytest.approx(0.0, abs=1e-9)
    assert not r["significant"]


def test_compare_pair_refuses_when_no_sequences_are_shared():
    a = _card("a", 0.8)
    b = _card("b", 0.8)
    for s in b.sequences:
        s.sequence = "other-" + s.sequence
    with pytest.raises(ValueError, match="share no sequences"):
        C.compare_pair(a, b, n_resamples=100)


def test_ours_table_applies_holm_correction():
    cards = [_card("base", 0.7, seed=1), _card("x", 0.71, seed=2), _card("y", 0.72, seed=3)]
    table = C.table_ours(cards, "base", n_resamples=300)
    assert "Holm" in table
    assert "| **base** (baseline)" in table


def test_published_table_blocks_a_protocol_mismatch():
    """The regression test for this project's original sin."""
    card = _card("old-style", 0.9, protocol="specklock-centre", split="single-clip-phantom16")
    table = C.table_published(card, "ardmav", n_resamples=200)
    assert "NOT COMPARABLE" in table
    assert "different matcher" in table
    assert "different split" in table


def test_published_table_allows_a_matching_protocol():
    card = _card("ours", 0.9, protocol="ardmav-official", split="official-test-15")
    table = C.table_published(card, "ardmav", n_resamples=200)
    assert "✅ yes" in table
    assert "NOT COMPARABLE" not in table


def test_published_table_never_prints_a_p_value():
    card = _card("ours", 0.9, protocol="ardmav-official", split="official-test-15")
    table = C.table_published(card, "ardmav", n_resamples=200)
    assert "p=" not in table
    assert "No p-value appears in this table" in table


def test_condition_table_stratifies():
    table = C.table_conditions([_card("m", 0.8)], threshold=0.5)
    assert "night" in table and "clear" in table


def test_condition_table_says_so_when_there_are_no_labels():
    card = _card("m", 0.8)
    for s in card.sequences:
        s.conditions = []
    table = C.table_conditions([card], threshold=0.5)
    assert "no condition labels" in table


def test_confuser_table_reports_hits_with_an_interval():
    table = C.table_confusers([_card("m", 0.8, bird_hits=3)], 0.5, ("bird",))
    assert "confuser hits" in table and "95% CI" in table


def test_confuser_table_says_so_when_no_confusers_are_labelled():
    card = _card("m", 0.8)
    for s in card.sequences:
        s.distractor_instances = {}
    table = C.table_confusers([card], 0.5, ("bird",))
    assert "no distractor objects" in table


def test_small_sample_verdict_is_inconclusive_not_better():
    """3 sequences cannot support a significance claim: the paired permutation test has
    only 2^3 = 8 arrangements, so its p-value floors at 0.125. The bootstrap will happily
    say 0.0000; the table must not turn that into '**better**'."""
    a = _card("big-gain", 0.95, n_seq=3, seed=1)
    b = _card("baseline", 0.50, n_seq=3, seed=1)
    table = C.table_ours([b, a], "baseline", n_resamples=400)
    assert "inconclusive (too few sequences)" in table
    assert "too few for the permutation column" in table


def test_ample_sample_can_reach_a_verdict():
    a = _card("big-gain", 0.95, n_seq=16, seed=1)
    b = _card("baseline", 0.50, n_seq=16, seed=1)
    table = C.table_ours([b, a], "baseline", n_resamples=400)
    assert "**better**" in table
    assert "too few for the permutation column" not in table
