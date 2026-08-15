"""Resolving a GT stem to its video file, and refusing to proceed when it cannot.

This is the bug that produced six NPS scorecards reading AP = 0.000. `infer_tiled` looked
for `<stem>.mp4`; NPS ships `.mov`. Nothing crashed. Every video was skipped with a
printed warning, ten empty detection files were written, and `tools/evaluate.py` then did
exactly the right thing -- scored each missing sequence as a total miss rather than
skipping it, because skipping would flatter the result.

The outcome was a complete, well-formed, entirely believable scorecard reporting that our
detector finds nothing at all. The models were fine. The path was wrong.

So there are two separate properties to pin: the resolver handles the containers these
corpora actually use, and a failure to resolve is fatal rather than cosmetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.infer_tiled import _resolve_video


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00")
    return p


@pytest.mark.parametrize("filename,stem", [
    ("phantom05.mp4", "phantom05"),      # ARD-MAV
    ("Clip_041.mov", "Clip_041"),        # NPS, the case that broke
    ("Clip_041.MOV", "Clip_041"),        # case varies between NPS releases
    ("Clip_041.avi", "Clip_041"),
    ("Clip_041.m4v", "Clip_041"),
])
def test_resolves_the_containers_these_corpora_actually_ship(tmp_path, filename, stem):
    _touch(tmp_path / filename)
    got = _resolve_video(tmp_path, stem)
    # Compared case-insensitively on purpose: Windows resolves Clip_041.mov to a file named
    # Clip_041.MOV and hands back the string we asked for. The cluster is Linux, where the
    # two are genuinely different files -- which is why both cases stay in _VIDEO_EXTS.
    assert got is not None and got.name.lower() == filename.lower()


def test_resolves_an_unpadded_clip_number(tmp_path):
    """Purdue's NPS release names them Clip_5.mov; Dogfight's annotations say Clip_005."""
    _touch(tmp_path / "Clip_5.mov")
    got = _resolve_video(tmp_path, "Clip_005")
    assert got is not None and got.name == "Clip_5.mov"


def test_resolves_a_padded_file_from_an_unpadded_stem(tmp_path):
    _touch(tmp_path / "Clip_005.mov")
    got = _resolve_video(tmp_path, "Clip_5")
    assert got is not None and got.name == "Clip_005.mov"


def test_returns_none_when_there_is_genuinely_no_video(tmp_path):
    _touch(tmp_path / "something_else.mp4")
    assert _resolve_video(tmp_path, "Clip_041") is None


def test_a_stem_prefix_does_not_masquerade_as_a_match(tmp_path):
    """Clip_04 must not be satisfied by Clip_041.mov -- that would score one sequence's
    detections against another sequence's ground truth."""
    _touch(tmp_path / "Clip_041.mov")
    assert _resolve_video(tmp_path, "Clip_04") is None


def test_unresolved_videos_make_the_run_fail_rather_than_report_zero():
    """The property that matters most. Pinned against the source because reaching it
    requires a GPU and a model; what must never regress is that the exit path exists."""
    src = Path(__file__).resolve().parents[2] / "tools" / "infer_tiled.py"
    text = src.read_text(encoding="utf-8")
    assert "ABORT:" in text and "return 2" in text, (
        "a missing video must abort, not warn: evaluate.py will happily score the gap as "
        "a total miss and hand you a believable AP of 0")
