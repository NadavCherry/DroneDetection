#!/usr/bin/env python3
"""Build the comparison table — against our own baselines, and against published work.

Two comparisons, and the tool refuses to blur them:

**Ours vs ours.** Both scorecards came from this code, on the same sequences, under the
same protocol. A difference can be tested. You get a paired bootstrap over *sequences*
(not frames — frames within a flight are not independent), a paired permutation p-value,
McNemar on per-sequence wins, and Holm correction across the whole table because an
ablation with eight rows is eight hypotheses.

**Ours vs published.** Their number came from their paper, their split, their metric.
There is no second sample, so **no p-value is possible** and none is printed. What is
printed: our interval, their point estimate, whether ours covers it, and every protocol
difference — derived by `Protocol.mismatches_with`, not remembered by whoever writes the
table. If the protocols differ the row is marked NOT COMPARABLE and the difference is
not shown at all, because subtracting those two numbers is the specific error that made
this project's old comparison table indefensible.

    python tools/compare.py --scorecards work/scorecards/*.json
    python tools/compare.py --scorecards A.json B.json --baseline A --by-condition
    python tools/compare.py --scorecards ours.json --vs-published --dataset ardmav
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.protocol import BY_KEY as PROTOCOLS  # noqa: E402
from benchmarks.published import for_dataset  # noqa: E402
from benchmarks.fast_bootstrap import (  # noqa: E402
    PooledAPResampler, paired_bootstrap_pooled_ap, paired_permutation_pooled_ap)
from benchmarks.scorecard import (  # noqa: E402
    Scorecard, pooled_ap, pooled_precision, pooled_recall)
from dronedet.stats import (  # noqa: E402
    BootstrapResult, compare_with_published, holm, mcnemar,
    paired_bootstrap_diff, paired_permutation_test,
    wilson)
from dronedet.console import use_utf8_stdio  # noqa: E402


# --------------------------------------------------------------------------- ours vs ours
def compare_pair(a: Scorecard, b: Scorecard, *, n_resamples: int = 10000,
                 seed: int = 0) -> dict:
    """Paired comparison of two scorecards over their common sequences."""
    common = sorted({s.sequence for s in a.sequences} & {s.sequence for s in b.sequences})
    if not common:
        raise ValueError(f"{a.model} and {b.model} share no sequences — nothing to pair")
    ia = {s.sequence: s for s in a.sequences}
    ib = {s.sequence: s for s in b.sequences}
    ua = [ia[k] for k in common]
    ub = [ib[k] for k in common]

    # The fast path, not a different statistic: benchmarks/fast_bootstrap sorts the
    # detections once and re-weights them per draw instead of rebuilding and re-sorting a
    # 400k-element Python list 20,000 times. dronedet/tests/test_fast_bootstrap.py asserts
    # equality with pooled_ap and with the slow permutation test to 1e-12.
    #
    # This is not an optimisation for its own sake. At the conf floor both arms are now
    # scored at (0.001, matching the competitor's), one pooled_ap call takes 354 ms, so the
    # old path needed 118 minutes per seed for the bootstrap alone. A statistic nobody can
    # afford to run does not get reported, and then the table has no significance column.
    fb = paired_bootstrap_pooled_ap(ua, ub, n_resamples=n_resamples, seed=seed)
    boot = BootstrapResult(observed=fb["observed"], lo=fb["lo"], hi=fb["hi"],
                           p_value=fb["p"], n_units=fb["n_units"],
                           n_resamples=fb["n_resamples"],
                           statistic_a=pooled_ap(ua), statistic_b=pooled_ap(ub))
    perm = paired_permutation_pooled_ap(ua, ub, n_resamples=n_resamples, seed=seed)

    # McNemar on per-sequence wins: how many sequences did each win outright? This asks a
    # different question from AP — "is it better more often" rather than "is it better on
    # average" — and the two disagree exactly when one big sequence carries the result.
    wins_a = wins_b = 0
    for sa, sb in zip(ua, ub):
        pa, pb = pooled_ap([sa]), pooled_ap([sb])
        wins_a += pa > pb
        wins_b += pb > pa
    mcn = mcnemar(wins_a, wins_b)

    return {
        "a": a.model, "b": b.model, "n_sequences": len(common),
        "ap_a": boot.statistic_a, "ap_b": boot.statistic_b,
        "diff": boot.observed, "lo": boot.lo, "hi": boot.hi,
        "p_bootstrap": boot.p_value, "p_permutation": perm,
        "wins_a": wins_a, "wins_b": wins_b, "p_mcnemar": mcn,
        "significant": boot.significant, "describe": boot.describe(a.model, b.model),
    }


def table_ours(cards: list[Scorecard], baseline: str | None, *, seed: int = 0,
               n_resamples: int = 10000) -> str:
    if not cards:
        return "_no scorecards_"
    base = next((c for c in cards if c.model == baseline), cards[0])
    others = [c for c in cards if c.model != base.model]

    lines = [
        f"### Ours vs ours — paired over sequences, baseline `{base.model}`",
        "",
        f"Dataset `{base.dataset_key}`, split `{base.split}`, protocol "
        f"`{base.protocol_key}`. Resampling unit is the **sequence** "
        f"({base.n_sequences} of them, {base.n_gt:,} labelled instances).",
        "",
        "| model | AP | Δ vs baseline | 95% CI on Δ | p (boot) | p (perm) | p (Holm) | seq wins | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| **{base.model}** (baseline) | {pooled_ap(base.sequences):.3f} | — | — | — | — | — | — | — |",
    ]

    rows, pvals = [], []
    for c in others:
        try:
            r = compare_pair(c, base, seed=seed, n_resamples=n_resamples)
        except ValueError as e:
            lines.append(f"| {c.model} | — | — | — | — | — | — | — | {e} |")
            continue
        rows.append(r)
        pvals.append((c.model, r["p_bootstrap"]))

    adjusted = {n: adj for n, _, adj in holm(pvals)} if pvals else {}
    for r in rows:
        adj = adjusted.get(r["a"], float("nan"))
        # Both tests must agree. The bootstrap alone will call a 3-sequence difference
        # significant; the permutation test cannot, because 3 units admit only 8
        # arrangements. Requiring both means a small benchmark reports "inconclusive"
        # rather than manufacturing confidence it has not earned.
        agreed = r["significant"] and adj < 0.05 and r["p_permutation"] < 0.05
        if agreed:
            verdict = "**better**" if r["diff"] > 0 else "**worse**"
        elif r["significant"] and adj < 0.05:
            verdict = "inconclusive (too few sequences)"
        else:
            verdict = "no difference"
        lines.append(
            f"| {r['a']} | {r['ap_a']:.3f} | {r['diff']:+.3f} | "
            f"[{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['p_bootstrap']:.4f} | "
            f"{r['p_permutation']:.4f} | {adj:.4f} | {r['wins_a']}–{r['wins_b']} | {verdict} |")

    lines += [
        "",
        "Holm-corrected across the whole table: an ablation with N rows is N hypotheses, and "
        "the best of eight raw p-values is not a finding. 'seq wins' is how many sequences "
        "each model won outright — it disagrees with AP exactly when one long sequence "
        "carries the result, which is worth knowing.",
    ]

    # With few sequences the paired permutation test cannot go below 1/2^n however large
    # the effect, while the bootstrap happily reports 0.0000. That disagreement is not a
    # bug in either test -- it is the sample size showing through, and it must be said out
    # loud or the bootstrap column will be quoted on its own.
    n = base.n_sequences
    if n and n < 8:
        floor = 1.0 / (2 ** n)
        lines += [
            "",
            f"⚠ **{n} sequences is too few for the permutation column to mean much.** A paired "
            f"permutation test over {n} units has only 2^{n} = {2 ** n} arrangements, so its "
            f"p-value cannot fall below ≈{floor:.3f} no matter how large the effect. Where the "
            "bootstrap reports 0.0000 and the permutation test does not, believe the "
            "permutation test: the bootstrap is resampling {n} units and inheriting their "
            "optimism. Report the effect size and the sequence count, not the p-value.".format(n=n),
        ]
    return "\n".join(lines)


# ----------------------------------------------------------------------- ours vs published
def table_published(card: Scorecard, dataset_key: str, *, n_resamples: int = 10000,
                    seed: int = 0) -> str:
    base_protocol = PROTOCOLS.get(card.protocol_key)
    if base_protocol is None:
        return f"_unknown protocol key `{card.protocol_key}` — cannot compare_"
    # The scorecard knows which split it actually ran on; the named protocol carries only a
    # default. Without this override a single-clip result would silently claim the protocol's
    # split, which is precisely how "one clip" got reported as a dataset-level number.
    ours_protocol = replace(base_protocol, split=card.split or base_protocol.split)

    ap = pooled_ap(card.sequences)
    # There is nothing to pair against, so this is a one-sample interval on our own
    # statistic, not a difference. Resampling unit is still the sequence.
    lo, hi = _bootstrap_own(card, n_resamples=n_resamples, seed=seed)

    published = for_dataset(dataset_key)
    if not published:
        return f"_no published results recorded for `{dataset_key}`_"

    lines = [
        f"### Ours vs published — `{dataset_key}`",
        "",
        f"Ours: **{ap:.3f}** [{lo:.3f}, {hi:.3f}] under `{ours_protocol.describe()}`, over "
        f"{card.n_sequences} sequence{'s' if card.n_sequences != 1 else ''} / "
        f"{card.n_gt:,} instances.",
        "",
        "| method | their number | their protocol | comparable? | verdict |",
        "|---|---|---|---|---|",
    ]
    for r in published:
        mismatches = ours_protocol.mismatches_with(r.protocol)
        cmp = compare_with_published(
            ap, (lo, hi), r.value, r.method, published_url=r.source_url,
            protocol_mismatches=mismatches, n_units=card.n_sequences)
        flag = "✅ yes" if cmp.comparable else "❌ **no**"
        note = r.method + (" ⚠ competitor-reported" if r.reported_by_competitor else "")
        lines.append(f"| {note} | {r.value:.3f} ({r.metric}) | {r.protocol.describe()} | "
                     f"{flag} | {cmp.verdict()} |")

    lines += [
        "",
        "**No p-value appears in this table and none can.** A published AP is a single "
        "scalar with no distribution behind it, so nothing can be tested against it. The "
        "interval is ours alone; 'indistinguishable' means our interval covers their point "
        "estimate, not that a test was run.",
    ]
    reasons = [(r, ours_protocol.mismatches_with(r.protocol)) for r in published]
    reasons = [(r, mm) for r, mm in reasons if mm]
    if reasons:
        lines += [
            "",
            "Rows marked ❌ differ in matcher, AP definition or split. Those two numbers are "
            "not on the same axis, and subtracting them is the error that made this project's "
            "earlier comparison table indefensible. Reasons, per row:",
        ]
        lines += [f"- **{r.method}**: " + "; ".join(mm) for r, mm in reasons]
    return "\n".join(lines)


def _bootstrap_own(card: Scorecard, *, n_resamples: int, seed: int) -> tuple[float, float]:
    """Percentile interval on our own AP, resampling sequences with replacement."""
    import numpy as np
    rng = np.random.default_rng(seed)
    seqs = card.sequences
    n = len(seqs)
    if n == 0:
        return (float("nan"), float("nan"))
    # Same fast path as the paired test: sort the detections once, then a draw is a
    # re-weighted cumulative sum. Left on the slow path this was the reason a comparison
    # job still took over an hour after the paired tests were made fast -- 10,000 calls to
    # pooled_ap at 354 ms each, hidden one function below the one I had already fixed.
    r = PooledAPResampler(seqs)
    vals = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        vals[i] = r.ap_for_counts(np.bincount(idx, minlength=n).astype(float))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ------------------------------------------------------------------------ conditions & birds
def table_conditions(cards: list[Scorecard], threshold: float) -> str:
    """The stratified table: does it still work at night, in rain, against cloud?"""
    conditions = sorted({c for card in cards for c in card.conditions_present()})
    if not conditions:
        return ("_no condition labels on these sequences — the dataset adapter did not supply "
                "them, so nothing can be said about night/rain/fog. This is a data gap, not a "
                "result._")
    lines = [
        "### By operating condition",
        "",
        "| model | condition | sequences | AP | recall | precision | n_gt |",
        "|---|---|---|---|---|---|---|",
    ]
    for card in cards:
        for cond in conditions:
            view = card.with_condition(cond)
            if not view.sequences:
                continue
            lines.append(
                f"| {card.model} | {cond} | {view.n_sequences} | "
                f"{pooled_ap(view.sequences):.3f} | "
                f"{pooled_recall(view.sequences, threshold):.3f} | "
                f"{pooled_precision(view.sequences, threshold):.3f} | {view.n_gt:,} |")
    lines += [
        "",
        "A condition row with few sequences is a hint, not a result — read the sequence count "
        "before the AP.",
    ]
    return "\n".join(lines)


def table_confusers(cards: list[Scorecard], threshold: float,
                    prefixes: tuple[str, ...]) -> str:
    """The bird/plane false-alarm table — the number nobody in this field publishes."""
    lines = [
        f"### Confuser rejection at score ≥ {threshold:g}",
        "",
        "| model | confuser hits | confuser instances | hits/instance | recall on drone | 95% CI on rate |",
        "|---|---|---|---|---|---|",
    ]
    any_rows = False
    for card in cards:
        hits, total = card.distractor_hits(threshold, prefixes)
        if total == 0:
            continue
        any_rows = True
        rate, lo, hi = wilson(hits, total)
        lines.append(
            f"| {card.model} | {hits} | {total:,} | {rate:.4f} | "
            f"{pooled_recall(card.sequences, threshold):.3f} | [{lo:.4f}, {hi:.4f}] |")
    if not any_rows:
        return (f"_no distractor objects matching {prefixes} in these scorecards — either the "
                "dataset has no labelled birds/planes, or the adapter did not carry them "
                "through. Without them, no false-alarm claim is measurable._")
    lines += [
        "",
        "Read this beside the recall column: a model with zero confuser hits and 0.2 recall has "
        "not solved anything. The Wilson interval is on the hit *rate*; with a few hundred "
        "confuser instances it stays wide, which is the honest state of the evidence.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------ cli
def main(argv=None) -> int:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorecards", nargs="+", required=True, type=Path)
    ap.add_argument("--baseline", help="model name to compare the others against")
    ap.add_argument("--vs-published", action="store_true")
    ap.add_argument("--dataset", help="dataset key for --vs-published (else taken from the card)")
    ap.add_argument("--by-condition", action="store_true")
    ap.add_argument("--confusers", nargs="*", default=["bird", "plane", "airplane", "helicopter"],
                    help="distractor name prefixes that must never be called a drone")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="operating threshold; take it from a VAL run, not this one")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    cards = [Scorecard.load(p) for p in a.scorecards]
    keys = {c.dataset_key for c in cards}
    if len(keys) > 1 and not a.vs_published:
        print(f"warning: scorecards span several datasets {sorted(keys)} — a paired "
              "comparison across different data is meaningless", file=sys.stderr)

    parts = [f"# Comparison report", "",
             f"Scorecards: {', '.join(c.model for c in cards)}", ""]

    if len(cards) > 1:
        parts += [table_ours(cards, a.baseline, seed=a.seed, n_resamples=a.resamples), ""]
    if a.by_condition:
        parts += [table_conditions(cards, a.threshold), ""]
    parts += [table_confusers(cards, a.threshold, tuple(a.confusers)), ""]
    if a.vs_published:
        key = a.dataset or cards[0].dataset_key
        parts += [table_published(cards[0], key, n_resamples=a.resamples, seed=a.seed), ""]

    report = "\n".join(parts)
    print(report)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(report, encoding="utf-8")
        print(f"\nwrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
