"""Tests for the benchmark harness: protocol, catalog, published registry, scorecard.

The behaviour worth guarding hardest is `Protocol.mismatches_with`. It is the mechanism
that stops this project repeating its own worst mistake — a table that compared a
centre-distance AP on one dataset's single easiest clip against a published IoU AP on a
different dataset's full test split, with nothing in the code able to notice.
"""

from __future__ import annotations

import pytest

from benchmarks import catalog, published
from benchmarks.protocol import AP50, ARDMAV_OFFICIAL, COCO, SPECKLOCK_CENTRE, Protocol
from benchmarks.scorecard import (
    Scorecard, SequenceResult, average_precision, pooled_ap, pooled_precision, pooled_recall)


# ------------------------------------------------------------------ protocol
def test_identical_protocols_have_no_mismatches():
    p = Protocol(matcher="iou", ap_style="ap50", iou_threshold=0.5, split="official-test-15")
    assert p.mismatches_with(p) == []


def test_matcher_difference_is_caught():
    ours = Protocol(matcher="centre", ap_style="voc-all-point", tau_px=12.0, split="s")
    theirs = Protocol(matcher="iou", ap_style="ap50", iou_threshold=0.5, split="s")
    mm = ours.mismatches_with(theirs)
    assert any("matcher" in m for m in mm)


def test_iou_threshold_difference_is_caught():
    a = Protocol(matcher="iou", ap_style="ap50", iou_threshold=0.5, split="s")
    b = Protocol(matcher="iou", ap_style="ap50", iou_threshold=0.25, split="s")
    assert any("IoU threshold" in m for m in a.mismatches_with(b))


def test_ap_definition_difference_is_caught():
    """COCO AP and AP50 are different quantities; conflating them flatters whoever
    reports AP50."""
    assert any("AP definition" in m for m in COCO.mismatches_with(AP50))


def test_split_difference_is_caught():
    a = Protocol(matcher="iou", ap_style="ap50", iou_threshold=0.5, split="official-test-15")
    b = Protocol(matcher="iou", ap_style="ap50", iou_threshold=0.5, split="single-clip-phantom16")
    assert any("different split" in m for m in a.mismatches_with(b))


def test_missing_split_makes_a_comparison_unverifiable():
    a = Protocol(matcher="iou", ap_style="ap50", iou_threshold=0.5, split="")
    b = Protocol(matcher="iou", ap_style="ap50", iou_threshold=0.5, split="official-test-15")
    assert any("not stated" in m for m in a.mismatches_with(b))


def test_label_inflation_caps_an_iou_comparison():
    """24 px labels make COCO AP arithmetically impossible; the protocol must say so
    rather than letting a 0.000 look like a detector failure."""
    inflated = Protocol(matcher="iou", ap_style="ap50", iou_threshold=0.5,
                        split="s", label_inflation_px=24.0)
    true_extent = Protocol(matcher="iou", ap_style="ap50", iou_threshold=0.5, split="s")
    mm = inflated.mismatches_with(true_extent)
    assert any("inflated labels" in m and "caps achievable IoU" in m for m in mm)


def test_the_repos_native_protocol_is_not_comparable_to_papers():
    """The whole reason this module exists."""
    assert SPECKLOCK_CENTRE.mismatches_with(AP50) != []


def test_protocol_describe_is_human_readable():
    assert "τ=12" in SPECKLOCK_CENTRE.describe()
    assert "IoU≥0.25" in ARDMAV_OFFICIAL.describe()


# ------------------------------------------------------------------ catalog
def test_ardmav_carries_the_official_split():
    d = catalog.DATASETS["ardmav"]
    assert len(d.official_test) == 15
    assert "phantom16" not in d.official_test, "phantom16 is a TRAIN video, not test"
    assert d.official_protocol is ARDMAV_OFFICIAL


def test_every_dataset_key_matches_its_registry_key():
    for key, d in catalog.DATASETS.items():
        assert d.key == key


def test_stills_datasets_are_not_marked_as_video():
    """A stills set cannot support a temporal claim, and the property is what stops a
    caller accidentally training the 3-moment stack on one."""
    assert not catalog.DATASETS["uav_smid"].is_video
    assert catalog.DATASETS["ardmav"].is_video


def test_gated_datasets_are_not_marked_offline_usable():
    assert not catalog.DATASETS["dvb"].usable_offline      # needs a signed agreement
    assert not catalog.DATASETS["ard100"].usable_offline   # BaiduYun
    assert catalog.DATASETS["ardmav"].usable_offline


def test_bird_datasets_are_discoverable():
    keys = {d.key for d in catalog.with_birds()}
    assert {"halmstad", "uav_smid"} <= keys


def test_condition_query_finds_the_weather_sets():
    fog = {d.key for d in catalog.with_condition(catalog.Condition.FOG)}
    assert {"extremetrack", "tricross"} <= fog
    night = {d.key for d in catalog.with_condition(catalog.Condition.NIGHT)}
    assert "halmstad" in night


def test_priority_ordering_puts_acquired_data_first():
    assert catalog.by_priority(1)[0].key == "ardmav"


# ------------------------------------------------------------------ published
def test_published_results_all_carry_a_source():
    for r in published.RESULTS:
        assert r.source_url.startswith("http"), f"{r.method} has no source"


def test_best_for_dataset_excludes_competitor_reported_numbers():
    """TransVisDrone's 0.15 on ARD100 comes from YOLOMG's authors. It must never be the
    number we claim to beat."""
    best = published.best_for_dataset("ard100")
    assert best is not None and not best.reported_by_competitor
    assert best.method.startswith("YOLOMG")


def test_the_ardmav_bar_is_recorded_with_its_forgiving_threshold():
    bar = published.best_for_dataset("ardmav")
    assert bar.value == pytest.approx(0.55)
    assert bar.protocol.iou_threshold == 0.25


def test_competitor_reported_numbers_are_flagged():
    tvd = [r for r in published.for_dataset("ard100") if r.method == "TransVisDrone"]
    assert tvd and tvd[0].reported_by_competitor


# ------------------------------------------------------------------ scorecard
def _seq(name, n_gt, dets, conditions=(), distractors=None):
    return SequenceResult(sequence=name, n_gt=n_gt, n_frames=100,
                          conditions=list(conditions), detections=list(dets),
                          distractor_instances=dict(distractors or {}))


def test_average_precision_is_one_for_perfect_detections():
    dets = [(0.9, "tp")] * 10
    assert average_precision(dets, 10) == pytest.approx(1.0)


def test_average_precision_penalises_a_top_ranked_false_positive():
    clean = average_precision([(0.9, "tp")], 1)
    dirty = average_precision([(0.99, "fp"), (0.5, "tp")], 1)
    assert dirty < clean


def test_average_precision_is_zero_with_no_ground_truth():
    assert average_precision([(0.9, "fp")], 0) == 0.0


def test_pooled_ap_weights_sequences_by_size_not_equally():
    """A 3,000-frame sequence must not count the same as a 30-frame one."""
    big = _seq("big", 100, [(0.9, "tp")] * 100)
    small_bad = _seq("small", 1, [(0.9, "fp")])
    pooled = pooled_ap([big, small_bad])
    mean_of_aps = (pooled_ap([big]) + pooled_ap([small_bad])) / 2
    assert pooled > mean_of_aps


def test_distractor_hits_are_counted_not_discarded():
    s = _seq("s", 10, [(0.9, "tp"), (0.8, "distractor:bird"), (0.7, "distractor:near")],
             distractors={"bird": 500, "near": 100})
    assert s.distractor_hits() == 2
    assert s.distractor_hits(prefixes=("bird",)) == 1
    assert s.distractor_total(prefixes=("bird",)) == 500


def test_distractor_hits_respect_the_threshold():
    s = _seq("s", 10, [(0.8, "distractor:bird"), (0.3, "distractor:bird")],
             distractors={"bird": 100})
    assert s.distractor_hits(0.5, ("bird",)) == 1
    assert s.distractor_hits(0.1, ("bird",)) == 2


def test_distractors_never_count_toward_precision():
    s = _seq("s", 1, [(0.9, "tp"), (0.8, "distractor:bird")])
    assert pooled_precision([s], 0.0) == 1.0


def test_condition_view_selects_the_right_sequences():
    card = Scorecard(model="m", dataset_key="d", protocol_key="ap50", split="test",
                     sequences=[_seq("a", 5, [(0.9, "tp")] * 5, conditions=["night"]),
                                _seq("b", 5, [(0.9, "tp")] * 5, conditions=["clear"])])
    night = card.with_condition("night")
    assert night.n_sequences == 1 and night.sequences[0].sequence == "a"
    assert "night" in night.split
    assert card.conditions_present() == ["clear", "night"]


def test_condition_view_is_empty_when_the_condition_is_absent():
    card = Scorecard(model="m", dataset_key="d", protocol_key="ap50", split="test",
                     sequences=[_seq("a", 5, [(0.9, "tp")] * 5, conditions=["clear"])])
    assert card.with_condition("fog").n_sequences == 0


def test_pooled_recall_and_precision_at_a_threshold():
    s = _seq("s", 10, [(0.9, "tp")] * 4 + [(0.2, "tp")] * 2 + [(0.9, "fp")])
    assert pooled_recall([s], 0.5) == pytest.approx(0.4)
    assert pooled_precision([s], 0.5) == pytest.approx(4 / 5)


def test_scorecard_round_trips_through_disk(tmp_path):
    card = Scorecard(model="m", dataset_key="ardmav", protocol_key="ardmav-official",
                     split="official-test-15", git_sha="abc123",
                     sequences=[_seq("phantom05", 10, [(0.9, "tp"), (0.4, "distractor:bird")],
                                     conditions=["clear"], distractors={"bird": 3})])
    p = tmp_path / "card.json"
    card.save(p)
    back = Scorecard.load(p)
    assert back.model == "m" and back.git_sha == "abc123"
    assert back.sequences[0].detections == [(0.9, "tp"), (0.4, "distractor:bird")]
    assert back.sequences[0].distractor_instances == {"bird": 3}
    assert pooled_ap(back.sequences) == pytest.approx(pooled_ap(card.sequences))


def test_loading_a_scorecard_from_a_different_schema_is_refused(tmp_path):
    """Silently reading an old artifact with new semantics is how a number changes
    meaning without anyone noticing."""
    import json
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"model": "m", "dataset_key": "d", "protocol_key": "p",
                             "split": "s", "sequences": [], "schema_version": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        Scorecard.load(p)
