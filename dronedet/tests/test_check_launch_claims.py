"""The launch checker must be able to FAIL.

This file exists because of a specific defect, not a hypothetical one. The forbidden-phrase
list in `tools/check_launch_claims.py` was first written through a non-raw Python string, so
every ``\b`` word-boundary was interpreted at write time and stored as byte 0x08. The bytes
are invisible in cat, in an editor, and in review. The patterns searched for a backspace
character, matched nothing, and the tool printed five PASS lines for a post that contained
all five forbidden phrases.

That is the worst failure mode available to a verification tool: it does not go quiet, it
actively reports success. The launch package is the one artefact in this repository that
cannot be quietly corrected after publication, and this tool is the only thing standing
between it and a stale claim, so the tool itself gets tested.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools.check_launch_claims import FORBIDDEN, publishable

REPO = Path(__file__).resolve().parents[2]
CHECKER = REPO / "tools" / "check_launch_claims.py"

# One dishonest sentence per pattern: the sentence that pattern exists to catch.
DISHONEST = {
    'claims SOTA': "SpeckLock is state of the art on tiny-drone detection.",
    'score-weighted per-frame metric only': "Bird rejection gives zero false positives.",
    'not supported: it leads overall': "It beats YOLOMG on small targets.",
    'the sweep does not establish this': "The sweep shows dt = 6 is optimal.",
    '24/24 without naming the perfect sensor': "Interception success was 24/24 in the sim.",
}


def test_checker_source_has_no_interpreted_escapes() -> None:
    """The original bug, asserted directly on the bytes."""
    raw = CHECKER.read_bytes()
    assert b"\x08" not in raw, (
        "tools/check_launch_claims.py contains byte 0x08 -- a \b word-boundary was "
        "interpreted when the file was written. Every regex using it silently matches "
        "nothing and every check built on it passes vacuously."
    )


def test_every_forbidden_pattern_still_fires() -> None:
    """A pattern that matches nothing is a check that cannot fail."""
    assert len(FORBIDDEN) == len(DISHONEST)
    for pat, why in FORBIDDEN:
        assert why in DISHONEST, f"no dishonest fixture for {why!r}"
        assert re.search(pat, DISHONEST[why], re.I), (
            f"pattern for {why!r} no longer matches the sentence it exists to catch: "
            f"{DISHONEST[why]!r}"
        )


def test_publishable_keeps_the_post_and_drops_the_commentary() -> None:
    """Scope correctness: the editorial sections legitimately contain forbidden phrases."""
    doc = "\n".join([
        "## 1 - Headline options",
        "| 5 | My detector loses to the state of the art. |",
        "",
        "## 2 - The post",
        "> Stack three moments as colour channels.",
        "> AP 0.159 to 0.895, same network.",
        "",
        "## 6 - Sentences that must not appear",
        '| "Zero false positives" | "Zero bird false alarms at track level" |',
    ])
    out = publishable(doc)
    assert "Stack three moments" in out
    assert "0.159" in out
    # Both of these are legitimate text that a whole-document scan would flag.
    assert "state of the art" not in out.lower()
    assert "zero false positives" not in out.lower()


@pytest.mark.skipif(not (REPO / "docs/launch/linkedin.md").is_file(),
                    reason="launch package not present")
def test_shipped_launch_package_is_clean() -> None:
    """The live post, through the same path the tool uses."""
    published = publishable((REPO / "docs/launch/linkedin.md").read_text(encoding="utf-8"))
    assert len(published) > 1500, (
        f"only {len(published)} chars of blockquote found -- if the post stops being "
        "blockquoted the scan silently covers nothing"
    )
    for pat, why in FORBIDDEN:
        hits = re.findall(pat, published, re.I)
        assert not hits, f"launch package contains forbidden phrasing ({why}): {hits[:2]}"
