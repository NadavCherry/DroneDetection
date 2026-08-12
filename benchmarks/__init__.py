"""Benchmark harness: what to measure on, how it must be measured, and against what.

Three modules, deliberately data-first and dependency-light (stdlib only, so they import
in the torch-free CI job):

* `protocol`  -- an evaluation protocol as an object, and `mismatches_with`, which
                 *derives* whether two numbers may be subtracted instead of trusting
                 whoever writes the table to remember.
* `catalog`   -- the datasets: official splits, official protocols, licences, download
                 gates, and which operating **conditions** each actually contains.
* `published` -- other people's numbers, each carrying the protocol that produced it.

The rule the package exists to enforce: **you may only subtract two numbers that share a
protocol.** Everything else is reported as a non-comparable pair of point estimates with
the differences listed.
"""

from .catalog import DATASETS, Condition, Dataset, Gate, by_priority, with_birds, with_condition
from .protocol import BY_KEY as PROTOCOLS, Protocol
from .published import RESULTS, PublishedResult, best_for_dataset, for_dataset

__all__ = [
    "DATASETS", "Dataset", "Condition", "Gate", "by_priority", "with_birds", "with_condition",
    "Protocol", "PROTOCOLS",
    "RESULTS", "PublishedResult", "for_dataset", "best_for_dataset",
]
