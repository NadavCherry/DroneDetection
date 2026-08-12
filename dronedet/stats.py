"""Statistical comparison for detection results.

This module exists because of a specific criticism: the pursuit half of this project
reports Wilson intervals, Holm-corrected p-values and Mann-Whitney tests, while the
detection half reported bare point estimates from single runs. A number without a
spread is an anecdote, and a comparison between two anecdotes is not evidence.

**The central honesty rule, which the API enforces.** There are two very different
comparisons, and conflating them is the most common way detection papers mislead:

* **Paired** — both methods were run *by us*, on the *same* test units, under the *same*
  protocol. Here a difference can be tested: `paired_bootstrap_diff`,
  `paired_permutation_test`, `mcnemar`. This is the only comparison that supports the
  word "significant".
* **Unpaired against a published scalar** — the other method's number came from its own
  paper, on its own split, under its own metric. **No significance test is possible**,
  because there is no second sample: a published AP is one number, not a distribution.
  `compare_with_published` therefore refuses to emit a p-value. It reports our interval,
  states whether the published point estimate falls inside it, and carries the protocol
  mismatches forward so they appear in the output rather than being quietly dropped.

Everything here is numpy-and-stdlib only, so it runs in the torch-free CI job.

Resampling unit
---------------
For video detection the resampling unit is a **sequence**, not a frame. Frames within a
flight are strongly correlated, so a per-frame bootstrap reports an interval far tighter
than the evidence supports -- on 10_06 it produces [1.000, 1.000] from what is really a
single flight. Pass whole sequences as units, and if a benchmark has few sequences, say
so: an honest wide interval is worth more than a precise-looking wrong one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

# Re-exported so there is exactly one implementation in the repo. `wilson` was
# previously copy-pasted into pursuit/tools/analyze.py and pursuit/tools/city_report.py;
# a fix to one silently diverged from the other.
__all__ = [
    "wilson", "holm", "mcnemar", "paired_bootstrap_diff", "paired_permutation_test",
    "cliffs_delta", "compare_with_published", "BootstrapResult", "PublishedComparison",
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Binomial confidence interval that behaves at the edges.

    The textbook ``p ± z·sqrt(p(1-p)/n)`` gives ``[1.0, 1.0]`` for 31/31, asserting
    certainty from 31 samples. Wilson returns about [0.89, 1.00], which is the honest
    statement. Returns ``(point, lo, hi)``.
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def holm(pairs: Sequence[tuple[str, float]]) -> list[tuple[str, float, float]]:
    """Holm-Bonferroni: control the family-wise error over several comparisons.

    Ablation tables test many hypotheses at once. Reporting a raw p < 0.05 from the best
    of eight comparisons is how a null result is dressed up as a finding. Returns
    ``(name, raw_p, adjusted_p)`` sorted by raw p.
    """
    ordered = sorted(pairs, key=lambda kv: kv[1])
    m = len(ordered)
    out, running = [], 0.0
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out.append((name, p, adj))
    return out


def _binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def mcnemar(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value for two detectors on the *same* items.

    ``only_a`` = items A got right and B did not; ``only_b`` = the reverse. Items both
    got right, or both wrong, carry no information about which is better and are
    correctly ignored -- which is the whole point of the test, and why it is the right
    tool for "does my detector find frames the baseline misses".

    Uses the exact binomial rather than the chi-square approximation, because the
    discordant count is often small on a benchmark with few sequences.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = max(only_a, only_b)
    return min(1.0, 2.0 * _binom_sf(k, n))


@dataclass
class BootstrapResult:
    """A difference between two methods, with the spread that makes it meaningful."""
    observed: float          # statistic(A) - statistic(B) on the real data
    lo: float
    hi: float
    p_value: float           # two-sided, proportion of resamples on the far side of 0
    n_units: int
    n_resamples: int
    statistic_a: float
    statistic_b: float

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero. Not a licence to say 'proven'."""
        return (self.lo > 0.0) or (self.hi < 0.0)

    def describe(self, name_a: str = "A", name_b: str = "B", unit: str = "sequences") -> str:
        verdict = "excludes 0" if self.significant else "**includes 0**"
        return (f"{name_a} {self.statistic_a:.3f} vs {name_b} {self.statistic_b:.3f} — "
                f"difference {self.observed:+.3f}, 95% CI [{self.lo:+.3f}, {self.hi:+.3f}] "
                f"({verdict}), p={self.p_value:.4f}, "
                f"paired bootstrap over {self.n_units} {unit}")


def paired_bootstrap_diff(units_a: Sequence, units_b: Sequence,
                          statistic: Callable[[list], float], *,
                          n_resamples: int = 10000, alpha: float = 0.05,
                          seed: int = 0) -> BootstrapResult:
    """Bootstrap the difference between two methods measured on the *same* units.

    ``units_a[i]`` and ``units_b[i]`` must be the two methods' results on the **same**
    test unit (normally one video). Resampling unit indices *jointly* is what makes this
    paired: it removes the between-unit variance that both methods share, so an easy
    video helping both does not count as evidence for either.

    ``statistic`` maps a list of unit-results to a scalar (e.g. pooled AP, mean recall).
    It is applied to the resampled *collection*, not averaged over units, so pooled
    metrics stay pooled.
    """
    if len(units_a) != len(units_b):
        raise ValueError(f"paired comparison needs equal lengths, "
                         f"got {len(units_a)} and {len(units_b)}")
    n = len(units_a)
    if n == 0:
        raise ValueError("no units to compare")
    if n < 5:
        # Not fatal -- some benchmarks genuinely have few sequences -- but the caller
        # should be reporting that alongside the interval.
        pass

    a, b = list(units_a), list(units_b)
    sa, sb = statistic(a), statistic(b)
    observed = sa - sb

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = statistic([a[j] for j in idx]) - statistic([b[j] for j in idx])

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    # Two-sided bootstrap p: how far into the tail zero sits.
    frac_le = float(np.mean(diffs <= 0.0))
    frac_ge = float(np.mean(diffs >= 0.0))
    p = min(1.0, 2.0 * min(frac_le, frac_ge))

    return BootstrapResult(observed=observed, lo=lo, hi=hi, p_value=p, n_units=n,
                           n_resamples=n_resamples, statistic_a=sa, statistic_b=sb)


def paired_permutation_test(units_a: Sequence, units_b: Sequence,
                            statistic: Callable[[list], float], *,
                            n_resamples: int = 10000, seed: int = 0) -> float:
    """Two-sided paired permutation p-value under the null 'the labels A/B are arbitrary'.

    Swaps each unit's two results independently with probability 0.5. Makes no
    distributional assumption, and unlike the bootstrap it tests a null hypothesis
    directly rather than inverting an interval. Report both when they disagree -- that
    disagreement is itself information about how few units there are.
    """
    if len(units_a) != len(units_b):
        raise ValueError("paired test needs equal lengths")
    n = len(units_a)
    if n == 0:
        raise ValueError("no units to compare")
    a, b = list(units_a), list(units_b)
    observed = abs(statistic(a) - statistic(b))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_resamples):
        swap = rng.random(n) < 0.5
        pa = [b[i] if swap[i] else a[i] for i in range(n)]
        pb = [a[i] if swap[i] else b[i] for i in range(n)]
        if abs(statistic(pa) - statistic(pb)) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_resamples + 1)      # add-one keeps p strictly positive


def cliffs_delta(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, str]:
    """Non-parametric effect size in [-1, 1], with the conventional magnitude label.

    A p-value says "the difference is probably not zero"; it says nothing about whether
    the difference matters. On a benchmark with 15 sequences, an AP gain of 0.003 can be
    consistent and irrelevant at the same time. Thresholds are Romano et al.'s:
    negligible < 0.147, small < 0.33, medium < 0.474.
    """
    xs, ys = list(xs), list(ys)
    if not xs or not ys:
        return 0.0, "undefined"
    gt = sum(1 for x in xs for y in ys if x > y)
    lt = sum(1 for x in xs for y in ys if x < y)
    d = (gt - lt) / (len(xs) * len(ys))
    a = abs(d)
    label = ("negligible" if a < 0.147 else "small" if a < 0.33
             else "medium" if a < 0.474 else "large")
    return d, label


@dataclass
class PublishedComparison:
    """Our measured result set beside a number quoted from someone else's paper.

    Deliberately carries **no p-value**. A published AP is a single scalar with no
    distribution behind it, so nothing can be tested against it. What can honestly be
    said is: here is our interval, here is their point estimate, here is whether ours
    covers it, and here is every way the two protocols differ.
    """
    ours: float
    ours_lo: float
    ours_hi: float
    published: float
    published_by: str
    published_url: str = ""
    protocol_mismatches: list[str] = field(default_factory=list)
    n_units: int = 0

    @property
    def comparable(self) -> bool:
        return not self.protocol_mismatches

    @property
    def covers_published(self) -> bool:
        return self.ours_lo <= self.published <= self.ours_hi

    def verdict(self) -> str:
        if not self.comparable:
            return "NOT COMPARABLE"
        if self.covers_published:
            return "indistinguishable at this sample size"
        return "ours higher" if self.ours > self.published else "ours lower"

    def describe(self, metric: str = "AP") -> str:
        head = (f"{metric}: ours {self.ours:.3f} [{self.ours_lo:.3f}, {self.ours_hi:.3f}] "
                f"(n={self.n_units}) vs {self.published_by} {self.published:.3f} "
                f"→ {self.verdict()}")
        if self.protocol_mismatches:
            head += ("\n  ⚠ no significance test is possible against a published scalar, and "
                     "these protocol differences make even the point estimates non-comparable:")
            head += "".join(f"\n    - {m}" for m in self.protocol_mismatches)
        else:
            head += ("\n  note: no significance test is possible against a published scalar — "
                     "the interval is ours alone.")
        if self.published_url:
            head += f"\n  source: {self.published_url}"
        return head


def compare_with_published(ours: float, ours_ci: tuple[float, float], published: float,
                           published_by: str, *, published_url: str = "",
                           protocol_mismatches: Sequence[str] = (),
                           n_units: int = 0) -> PublishedComparison:
    """Build the honest form of a 'we beat X' claim. See `PublishedComparison`."""
    return PublishedComparison(ours=ours, ours_lo=ours_ci[0], ours_hi=ours_ci[1],
                               published=published, published_by=published_by,
                               published_url=published_url,
                               protocol_mismatches=list(protocol_mismatches),
                               n_units=n_units)
