"""The --official-split path must resolve a protocol without inventing attributes.

This branch runs only for datasets that publish an official test split, so a mistake in it
is invisible on every other corpus. `Dataset` has `official_protocol` (a Protocol object)
and no `protocol_key`; reaching for the latter raised AttributeError on every ARD-MAV
evaluation while all six NPS evaluations passed, and it did so *after* 40-90 minutes of
inference per run.

The test exercises the resolution both ways -- explicit --protocol and dataset default --
against the real catalogue, because the bug was a field name and only real objects have
real field names.
"""

from __future__ import annotations

import pytest

from benchmarks.catalog import DATASETS
from benchmarks.protocol import BY_KEY as PROTOCOLS


def test_dataset_has_official_protocol_and_not_protocol_key():
    e = DATASETS["ardmav"]
    assert e.official_protocol is not None
    assert not hasattr(e, "protocol_key"), \
        "if Dataset grows a protocol_key, tools/evaluate.py must be revisited"


def test_the_official_protocol_carries_the_split_name_the_table_matches_on():
    e = DATASETS["ardmav"]
    assert e.official_protocol.split == "official-test-15"
    assert PROTOCOLS["ardmav-official"].split == e.official_protocol.split, \
        "the keyed protocol and the catalogue's must be the same object's split"


def test_the_split_size_claim_matches_the_catalogue():
    e = DATASETS["ardmav"]
    from tools.evaluate import _declared_split_size
    assert _declared_split_size(e.official_protocol.split) == len(set(e.official_test))


@pytest.mark.parametrize("key", [k for k, v in DATASETS.items()
                                 if v.official_test and v.official_protocol])
def test_every_dataset_with_an_official_split_can_resolve_one(key):
    """Whatever the --official-split branch reads must exist for all of them, not just
    the one that happened to be tested."""
    e = DATASETS[key]
    proto = e.official_protocol
    assert proto is not None and proto.split, f"{key}: no usable official protocol"
