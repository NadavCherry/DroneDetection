"""The scorecard's split label must come from the protocol, not from a second literal.

`Protocol.mismatches_with` compares split NAMES, and rightly refuses to subtract two APs
whose splits differ. So a scorecard that describes the same 15 ARD-MAV videos as
"official-test" while the protocol calls them "official-test-15" makes every published
comparison read NOT COMPARABLE -- and it did, for the whole GLAD/TPH/MGMD table, with the
guard behaving exactly as designed and nothing actually wrong but the label.

Verified before this was changed: the GT directory, `ARD_TEST_IDS` and the scorecard's own
sequence list are the same 15 videos, set-for-set. Only the string differed.

`build_scorecard` already defaulted to `protocol.split`; `main()` passed a literal that
overrode it. That is the failure mode worth pinning -- not "is there a default" but "does
the caller respect it".
"""

from __future__ import annotations

import re
from pathlib import Path

from benchmarks.protocol import BY_KEY as PROTOCOLS
from tools.evaluate import _declared_split_size

SRC = Path(__file__).resolve().parents[2] / "tools" / "evaluate.py"


def test_main_does_not_hardcode_a_split_name_for_official_splits():
    """The literal "official-test" must not be what lands in the scorecard."""
    src = SRC.read_text(encoding="utf-8")
    body = src[src.index("if a.official_split:"):src.index("conditions_map = {}")]
    assert "proto.split" in body, \
        "the split label must be taken from the protocol, which is the thing compare.py " \
        "matches against"


def test_the_ardmav_protocol_and_its_catalogue_entry_agree_on_size():
    """The label asserts a count; the catalogue must actually contain it."""
    from benchmarks.catalog import DATASETS

    proto = PROTOCOLS["ardmav-official"]
    want = _declared_split_size(proto.split)
    assert want == 15, f"ardmav-official declares {proto.split!r}"

    entry = DATASETS["ardmav"]
    assert entry.official_test is not None
    assert len(set(entry.official_test)) == want, (
        f"protocol says {want} sequences, catalogue lists "
        f"{len(set(entry.official_test))}")


def test_declared_split_size_reads_a_trailing_count_only():
    assert _declared_split_size("official-test-15") == 15
    assert _declared_split_size("official-test-15/small") == 15   # condition subset
    assert _declared_split_size("nps-dogfight-10") == 10
    assert _declared_split_size("official-test") is None          # no claim, no check
    assert _declared_split_size("mgmd-self-chosen-UNENUMERATED") is None


def test_every_protocol_that_states_a_count_states_a_plausible_one():
    """A protocol naming zero or one sequence is a typo, not a benchmark."""
    for key, proto in PROTOCOLS.items():
        n = _declared_split_size(proto.split or "")
        if n is not None:
            assert n >= 2, f"{key}: split {proto.split!r} claims {n} sequences"


def test_the_split_label_survives_a_round_trip_through_the_scorecard(tmp_path):
    """End to end on the real dataclass rather than on a string in a source file."""
    from benchmarks.scorecard import Scorecard

    proto = PROTOCOLS["ardmav-official"]
    card = Scorecard(model="m", dataset_key="ardmav",
                     protocol_key="ardmav-official", split=proto.split)
    p = tmp_path / "c.json"
    card.save(p)

    import json
    assert json.loads(p.read_text(encoding="utf-8"))["split"] == proto.split
    assert re.search(r"-\d+$", proto.split), \
        "ardmav-official's split should carry its sequence count"
