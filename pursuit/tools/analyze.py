"""Statistical analysis of recorded engagements: is the system actually good?

A hit rate on its own is a number without an error bar, and with 30-60
engagements per environment the error bar is wide enough to matter -- 27/31 and
29/31 are not distinguishable, and treating them as if they were is how a
regression gets shipped. So every rate here carries a Wilson score interval, and
every claimed difference between groups carries a test rather than an eyebrow.

What it answers:

* **How good, with what uncertainty** -- hit rate and CI, and the distribution of
  the true closest approach (not the scored miss, which is mostly un-flown
  forward distance).
* **Does anything predict failure** -- environment, arrival direction, evasion
  policy, start range, detection rate. Each gets a Fisher exact test (categorical)
  or a Mann-Whitney U (continuous) against the hit/miss outcome, with Holm
  correction, because testing six factors at p<0.05 finds one by luck about a
  quarter of the time.
* **Where the misses live** -- separated into acquisition failures (never locked)
  and closure failures (locked and did not arrive), because they have different
  causes and only the second is the guidance law's fault.
* **Timing** -- per-stage cost and the frame rate the loop can hold.

Everything is computed from ``results.json`` files, and the tests are implemented
here rather than pulled from scipy so the numbers can be checked by hand.

    python -m pursuit.tools.analyze --search work/pursuit/final
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from itertools import combinations
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pursuit.tools.report import load  # noqa: E402


# ---------------------------------------------------------------- statistics

# wilson and holm live in dronedet.stats so there is exactly one implementation in the
# repo -- they were previously copy-pasted here and into city_report.py, where a fix to
# one would silently diverge from the other. Both are numpy/stdlib only, so importing
# them keeps this module inside the torch-free CI contract.
from dronedet.stats import holm, wilson  # noqa: E402,F401


def _logf(n: int) -> float:
    return math.lgamma(n + 1)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on a 2x2 table.

    Exact rather than chi-square because the cells here are routinely small
    (one environment, four failures), which is precisely where chi-square's
    normal approximation stops being trustworthy.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1, row2, col1, col2 = a + b, c + d, a + c, b + d

    def prob(x):
        y = col1 - x
        z_ = row1 - x
        w = row2 - y
        if min(x, y, z_, w) < 0:
            return 0.0
        return math.exp(_logf(row1) + _logf(row2) + _logf(col1) + _logf(col2)
                        - _logf(n) - _logf(x) - _logf(y) - _logf(z_) - _logf(w))

    observed = prob(a)
    total = 0.0
    for x in range(max(0, col1 - row2), min(col1, row1) + 1):
        p = prob(x)
        if p <= observed * (1 + 1e-9):
            total += p
    return min(1.0, total)


def mann_whitney(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Two-sided rank-sum test, normal approximation with tie correction.

    Rank-based on purpose: miss distances and times are visibly skewed, and a
    t-test on skewed data with n < 40 answers a question nobody asked.
    """
    n1, n2 = len(xs), len(ys)
    if n1 == 0 or n2 == 0:
        return 1.0
    merged = sorted([(v, 0) for v in xs] + [(v, 1) for v in ys])
    ranks, i, ties = {}, 0, 0.0
    vals = [v for v, _ in merged]
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    r1 = sum(ranks[k] for k, (_v, g) in enumerate(merged) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return 1.0
    z = (abs(u1 - mu) - 0.5) / math.sqrt(var)
    return max(0.0, min(1.0, math.erfc(z / math.sqrt(2.0))))


# ------------------------------------------------------------------ analysis

def collect(search: Path) -> list:
    rows = []
    for run in load(search):
        for r in run["payload"]["results"]:
            cfg = r.get("config") or {}
            sm = r.get("stage_ms") or {}
            rows.append({
                "scene": run["scene"], "name": r.get("name", "?"),
                "entry": cfg.get("entry") or "-", "policy": r.get("policy", "-"),
                "r0": cfg.get("start_range_m"),
                "ingress": bool(cfg.get("ingress")),
                "hit": bool(r.get("success")), "outcome": r.get("outcome", "?"),
                "cpa": r.get("pass_cpa_m"), "scored": r.get("miss_distance_m"),
                "vert": r.get("pass_vertical_m"), "lat": r.get("pass_lateral_m"),
                "t": r.get("time_to_intercept_s"), "acq": r.get("acquire_time_s"),
                "det": r.get("detect_rate"), "trk": r.get("track_rate"),
                "det_ms": sm.get("detect_ms"), "trk_ms": sm.get("track_ms"),
                "gui_ms": sm.get("guidance_ms"),
                "p95": sm.get("pipeline_p95_ms"),
            })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", default="work/pursuit/final")
    ap.add_argument("--out", default="")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)

    rows = collect(ROOT / a.search)
    if not rows:
        print(f"no results.json under {ROOT / a.search}")
        return 1

    L = []
    def out(s=""):
        L.append(s)
        print(s)

    n = len(rows)
    hits = [r for r in rows if r["hit"]]
    p, lo, hi = wilson(len(hits), n)
    out(f"# End-to-end analysis — {n} engagements\n")
    out("## 1. Hit rate\n")
    out(f"**{len(hits)}/{n} = {100 * p:.1f}%**   95% CI [{100 * lo:.1f}%, "
        f"{100 * hi:.1f}%]  (Wilson)\n")

    # -- failure taxonomy ---------------------------------------------------
    miss = [r for r in rows if not r["hit"]]
    if miss:
        acq = [r for r in miss if r["outcome"] == "never_acquired"
               or (r["acq"] is None)]
        clo = [r for r in miss if r not in acq]
        out("Failures split by cause — only the second kind is the guidance "
            "law's:\n")
        out("| failure | n | meaning |")
        out("|---|---|---|")
        out(f"| never acquired | {len(acq)} | the detector never produced a "
            f"confirmed track |")
        out(f"| acquired, did not close | {len(clo)} | locked on and still "
            f"failed to arrive |")
        out("")

    # -- miss distance ------------------------------------------------------
    cpas = [r["cpa"] for r in hits if r["cpa"] is not None]
    if cpas:
        s = sorted(cpas)
        out("## 2. Where the intercepts land\n")
        out("| statistic | value |")
        out("|---|---|")
        out(f"| mean true closest approach | **{st.fmean(s):.3f} m** |")
        out(f"| median | {st.median(s):.3f} m |")
        out(f"| p90 / p95 / max | {s[int(0.9 * len(s))]:.3f} / "
            f"{s[min(len(s) - 1, int(0.95 * len(s)))]:.3f} / {max(s):.3f} m |")
        out(f"| best | {min(s):.3f} m |")
        out(f"| inside 0.5 m | {100 * sum(1 for v in s if v <= 0.5) / len(s):.0f}% |")
        out("")
        vs = [r["vert"] for r in hits if r["vert"] is not None]
        ls = [r["lat"] for r in hits if r["lat"] is not None]
        if len(vs) > 2:
            for label, xs in (("vertical", vs), ("lateral", ls)):
                m, sd = st.fmean(xs), st.pstdev(xs)
                sem = sd / math.sqrt(len(xs)) if xs else 0.0
                tstat = m / sem if sem > 1e-12 else 0.0
                verdict = ("biased" if abs(tstat) > 2.0 else "centred")
                out(f"- **{label}**: {m * 100:+.1f} cm ± {sd * 100:.1f} "
                    f"(t={tstat:+.2f}) — {verdict}")
            out("")

    # -- what predicts failure ---------------------------------------------
    out("## 3. Does anything predict failure?\n")
    tests = []
    for factor in ("scene", "entry", "policy"):
        groups = sorted({r[factor] for r in rows})
        if len(groups) < 2:
            continue
        worst = None
        for g1, g2 in combinations(groups, 2):
            a1 = sum(1 for r in rows if r[factor] == g1 and r["hit"])
            b1 = sum(1 for r in rows if r[factor] == g1 and not r["hit"])
            c1 = sum(1 for r in rows if r[factor] == g2 and r["hit"])
            d1 = sum(1 for r in rows if r[factor] == g2 and not r["hit"])
            pv = fisher_exact(a1, b1, c1, d1)
            if worst is None or pv < worst[1]:
                worst = (f"{factor}: {g1} vs {g2}", pv)
        if worst:
            tests.append(worst)
    for factor in ("r0", "det", "trk"):
        xs = [r[factor] for r in rows if r["hit"] and r[factor] is not None]
        ys = [r[factor] for r in rows if not r["hit"] and r[factor] is not None]
        if len(xs) > 2 and len(ys) > 2:
            tests.append((f"{factor} (hit vs miss)", mann_whitney(xs, ys)))

    if tests:
        out("Holm-corrected across all factors — testing six things at p<0.05 "
            "finds one by luck about a quarter of the time.\n")
        out("| factor | p (raw) | p (Holm) | significant |")
        out("|---|---|---|---|")
        for name, raw, adj in holm(tests):
            out(f"| {name} | {raw:.4f} | {adj:.4f} | "
                f"{'**yes**' if adj < a.alpha else 'no'} |")
        out("")
        if all(adj >= a.alpha for _n, _r, adj in holm(tests)):
            out("No factor survives correction: the failures are not "
                "concentrated in any environment, arrival direction or evasion "
                "policy that the sample can resolve.\n")

    # -- detection vs outcome ----------------------------------------------
    dh = [r["det"] for r in hits if r["det"] is not None]
    dm = [r["det"] for r in rows if not r["hit"] and r["det"] is not None]
    if dh and dm:
        out("## 4. Perception versus guidance\n")
        out(f"- detection rate on **hits**: {st.fmean(dh):.3f} "
            f"(median {st.median(dh):.3f}, n={len(dh)})")
        out(f"- detection rate on **misses**: {st.fmean(dm):.3f} "
            f"(median {st.median(dm):.3f}, n={len(dm)})")
        out(f"- Mann-Whitney p = {mann_whitney(dh, dm):.4f}")
        overlap = (max(dm) >= min(dh))
        out(f"- ranges {'overlap' if overlap else '**do not overlap**'}: "
            f"hits [{min(dh):.2f}, {max(dh):.2f}], "
            f"misses [{min(dm):.2f}, {max(dm):.2f}]")
        out("")

    # -- timing -------------------------------------------------------------
    det = [r["det_ms"] for r in rows if r["det_ms"]]
    if det:
        trk = [r["trk_ms"] for r in rows if r["trk_ms"]] or [0.0]
        gui = [r["gui_ms"] for r in rows if r["gui_ms"]] or [0.0]
        loop = st.fmean(det) + st.fmean(trk) + st.fmean(gui)
        p95 = [r["p95"] for r in rows if r["p95"]]
        out("## 5. Frame rate\n")
        out("| stage | ms | share |")
        out("|---|---|---|")
        for label, xs in (("detector", det), ("tracker", trk), ("guidance", gui)):
            out(f"| {label} | {st.fmean(xs):.2f} | "
                f"{100 * st.fmean(xs) / loop:.0f}% |")
        out(f"| **total** | **{loop:.2f}** | |")
        out("")
        out(f"- **{1000 / loop:.1f} FPS** mean"
            + (f", **{1000 / st.fmean(p95):.1f} FPS** at p95" if p95 else ""))
        out(f"- 20 Hz control budget is 50 ms — "
            f"{'**met**' if loop <= 50 else '**NOT met**'} ({loop:.1f} ms)")
        out("")

    tt = [r["t"] for r in hits if r["t"]]
    if tt:
        s = sorted(tt)
        out("## 6. Time to intercept\n")
        out(f"median **{st.median(s):.2f} s**, p90 {s[int(0.9 * len(s))]:.2f} s, "
            f"max {max(s):.2f} s\n")

    if a.out:
        pth = ROOT / a.out
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_text("\n".join(L))
        print(f"\nwrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
