#!/usr/bin/env python3
"""Compare the dt sweep on FULL-FRAME HELD-OUT TEST AP, paired over sequences.

Why this and not the validation table
-------------------------------------
The dt sweep's validation numbers are clean and orderly: an inverted U peaking at dt=6,
seed ranges that do not overlap. But this project has already measured that its
tile-level validation metric does not predict its full-frame test result -- the same
weights score val mAP50 0.941 and test AP 0.487 (`docs/reports/yolomg-nps-discrepancy.md`).
A spacing chosen on validation is chosen on the weaker of the two metrics, so
`docs/reports/dt-ablation.md` fixed, before these numbers existed, that the deployed
protocol decides and that a disagreement between the curves gets analysed rather than
resolved in validation's favour.

This reads the scorecards both arms already wrote and applies the test that rule names.

Why paired, and why it will probably still say "inconclusive"
-------------------------------------------------------------
Seed noise on the test metric is enormous relative to the effect: dt=6's three seeds are
0.4819 / 0.5441 / 0.4347, an SD of 0.055 against between-setting differences of 0.005 to
0.030. An unpaired three-seed comparison cannot resolve that and never could have.

The paired bootstrap and permutation over sequences remove the between-sequence variance
both arms share, which is the only thing that makes a comparison this small tractable at
all -- and even then, ARD-MAV's size curve showed a consistent-on-every-seed effect fail
significance over 15 sequences, where NPS test has 10. Both tests must agree before
anything is called significant, matching `tools/make_summary.py`.

    PYTHONPATH=. python tools/dt_compare.py
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.fast_bootstrap import (paired_bootstrap_pooled_ap,  # noqa: E402
                                       paired_permutation_pooled_ap)
from dronedet import metrics as M  # noqa: E402
from dronedet.console import use_utf8_stdio  # noqa: E402


class _Unit:
    """One sequence's scored detections, shaped for benchmarks.fast_bootstrap."""

    __slots__ = ("sequence", "n_gt", "detections")

    def __init__(self, sequence, n_gt, detections):
        self.sequence, self.n_gt, self.detections = sequence, n_gt, detections


def load(path: str) -> dict:
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    return {s["sequence"]: _Unit(s["sequence"], s["n_gt"],
                                 [(float(sc), o) for sc, o in s["detections"]])
            for s in j["sequences"]}


def pooled_ap(u: dict) -> float:
    recs = [M.Record(frame=0, score=sc, outcome=o)
            for unit in u.values() for sc, o in unit.detections]
    return M.average_precision(recs, sum(x.n_gt for x in u.values()))


def main() -> int:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorecards", default="work/scorecards")
    ap.add_argument("--baseline-dt", type=int, default=6)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    sc = Path(a.scorecards)
    arms: dict[int, dict[int, dict]] = {}
    for p in sorted(glob.glob(str(sc / "dt*_nps-s*.json"))):
        dt = int(re.search(r"dt(\d+)_", p).group(1))
        seed = int(re.search(r"-s(\d+)\.json", p).group(1))
        arms.setdefault(dt, {})[seed] = load(p)
    # dt=6 is the shipped configuration; its scorecards predate this sweep and were
    # written by cluster/eval_e100.sbatch through the identical pipeline.
    for p in sorted(glob.glob(str(sc / "temporal_nps-e100-s*.json"))):
        seed = int(re.search(r"-s(\d+)\.json", p).group(1))
        arms.setdefault(a.baseline_dt, {})[seed] = load(p)
    if a.baseline_dt not in arms:
        raise SystemExit(f"no scorecards for the dt={a.baseline_dt} baseline")

    L = ["# dt ablation on full-frame held-out test AP", "",
         "10 NPS test clips, tiled full-frame inference, unified evaluator, conf 0.001 — "
         "the same path every headline number in this repository uses.", "",
         "| dt | n | test AP | sd | per-seed | vs dt=%d |" % a.baseline_dt,
         "|---|---|---|---|---|---|"]
    # Every mean first: the table's delta column references the baseline, and building it
    # inside the same loop read means[6] before dt=2 and dt=4 had passed over it.
    per_seed = {dt: [pooled_ap(arms[dt][s]) for s in sorted(arms[dt])] for dt in arms}
    means = {dt: st.fmean(v) for dt, v in per_seed.items()}
    for dt in sorted(arms):
        vals = per_seed[dt]
        sd = st.stdev(vals) if len(vals) > 1 else 0.0
        delta = "—" if dt == a.baseline_dt else "%+.4f" % (means[dt] - means[a.baseline_dt])
        L.append("| %s%d%s | %d | **%.4f** | %.4f | %s | %s |"
                 % ("**" if dt == a.baseline_dt else "", dt,
                    "**" if dt == a.baseline_dt else "", len(vals), means[dt], sd,
                    " / ".join("%.4f" % v for v in vals), delta))

    rank = sorted(means, key=lambda d: -means[d])
    L += ["", "**Test ranking:** " + " > ".join("dt%d" % d for d in rank), ""]

    L += ["## Paired, seed-matched, over the 10 shared sequences", "",
          "Positive delta favours dt=%d. Significant only when the bootstrap CI excludes "
          "zero **and** the permutation p < 0.05, matching `tools/make_summary.py`." % a.baseline_dt,
          "", "| vs | seed | d AP | 95% CI | p perm | verdict |", "|---|---|---|---|---|---|"]
    sig_total = 0
    for dt in sorted(k for k in arms if k != a.baseline_dt):
        for seed in sorted(arms[a.baseline_dt]):
            if seed not in arms[dt]:
                continue
            base, other = arms[a.baseline_dt][seed], arms[dt][seed]
            common = sorted(set(base) & set(other))
            ua = [base[k] for k in common]
            ub = [other[k] for k in common]
            r = paired_bootstrap_pooled_ap(ua, ub, n_resamples=a.resamples, seed=seed)
            pp = paired_permutation_pooled_ap(ua, ub, n_resamples=a.resamples, seed=seed)
            sig = (r["lo"] > 0 or r["hi"] < 0) and pp < 0.05
            sig_total += sig
            L.append("| dt%d | %d | %+.4f | [%+.4f, %+.4f] | %.4f | %s |"
                     % (dt, seed, r["observed"], r["lo"], r["hi"], pp,
                        "**significant**" if sig else "no difference"))
    L += ["", "**%d of %d paired comparisons reached significance.**"
          % (sig_total, 3 * (len(arms) - 1)), ""]

    out = "\n".join(L) + "\n"
    print(out)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
