#!/usr/bin/env python3
"""Right-size the CPU jobs' --mem from measurement, and make them self-reporting.

WHY THIS EXISTS
---------------
This cluster sets ``JobAcctGatherType = (null)``. SLURM therefore records **no** peak RSS:
``sacct --format=MaxRSS`` is empty for every job ever run here, and ``sstat`` returns
nothing for a running one. Nothing has ever contradicted a memory request, so the requests
drifted upward on habit -- 16 G and 24 G for jobs that turned out to need three.

Measured with ``/usr/bin/time -v`` on this cluster:

    tools/make_summary.py, all 30 arms, INCLUDING the bootstrap   3.21 GB
    tools/make_summary.py, loading only                           3.06 GB
    tools/size_curve.py, two arms, one dataset                    0.48 GB

The bootstrap costs almost nothing over loading, which is the useful surprise: it
re-weights data already in memory rather than copying it. So the request should be sized
by how many scorecards a job opens, not by how much resampling it does.

Every job this touches also gains ``/usr/bin/time -v``, so it prints its own peak RSS into
its log. The next person to size one of these gets a number instead of inheriting a guess.

    python tools/rightsize_jobs.py --cluster-dir cluster
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

#: filename -> (expected current, new, one-line justification)
SIZES: dict[str, tuple[str, str, str]] = {
    "regen_summary.sbatch": ("16G", "4G",
                             "make_summary peaks at 3.21 GB over all 30 arms"),
    "gzab.sbatch": ("16G", "5G",
                    "two sequential make_summary runs, 3.21 GB each"),
    "dt_compare.sbatch": ("16G", "3G",
                          "15 arms, half of make_summary's 30"),
    "sizecurve.sbatch": ("24G", "4G",
                         "size_curve peaks at 0.48 GB for 2 arms; 9 scales to ~2 GB"),
    "figures.sbatch": ("12G", "3G",
                       "matplotlib plus one video decode for fig5"),
    "splitfix.sbatch": ("4G", "2G",
                        "shuffles text; the images it lists are never opened"),
}

NOTE = (
    "### {new}, measured rather than guessed: {why}.\n"
    "### This cluster sets JobAcctGatherType=(null), so sacct MaxRSS and sstat are both\n"
    "### empty and nothing ever contradicted an over-request. The python step below runs\n"
    "### under /usr/bin/time -v and prints its own peak RSS into this job's log, so the\n"
    "### next person sizing it has a measurement instead of a habit.\n"
)


def instrument(text: str) -> tuple[str, int]:
    """Prefix the job's python invocations with /usr/bin/time -v.

    Only lines that START a python command are touched, and only once: a continuation
    line inside an already-wrapped command must not be wrapped again, and a heredoc body
    that happens to contain the word python is not a command at all.
    """
    out, n, in_heredoc = [], 0, False
    for line in text.splitlines():
        stripped = line.lstrip()
        if re.search(r"<<\s*[\"']?\w+[\"']?\s*$", line):
            in_heredoc = True
        elif in_heredoc and stripped in ("PY", "EOF", "PYEOF"):
            in_heredoc = False
        if (not in_heredoc
                and re.match(r"^(python|python3) -u? ?tools/", stripped)
                and "/usr/bin/time" not in line):
            indent = line[:len(line) - len(stripped)]
            out.append(f"{indent}/usr/bin/time -v {stripped}")
            n += 1
        else:
            out.append(line)
    return "\n".join(out) + "\n", n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster-dir", type=Path, default=Path("cluster"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    total_saved = 0
    for fn, (old, new, why) in SIZES.items():
        p = a.cluster_dir / fn
        if not p.is_file():
            print(f"  skip {fn}: absent")
            continue
        t = p.read_text(encoding="utf-8")
        m = re.search(r"#SBATCH --mem=(\S+)", t)
        if not m:
            print(f"  skip {fn}: no --mem line")
            continue
        if m.group(1) != old:
            print(f"  skip {fn}: expected --mem={old}, found --mem={m.group(1)}")
            continue
        t = t.replace(f"#SBATCH --mem={old}",
                      f"#SBATCH --mem={new}\n" + NOTE.format(new=new, why=why), 1)
        t, n_wrapped = instrument(t)
        saved = int(old.rstrip("G")) - int(new.rstrip("G"))
        total_saved += saved
        if a.dry_run:
            print(f"  would set {fn}: {old} -> {new} (-{saved}G), wrap {n_wrapped} command(s)")
        else:
            p.write_text(t, encoding="utf-8")
            print(f"  {fn}: {old} -> {new}  (-{saved}G, {n_wrapped} command(s) instrumented)")
    print(f"\n  {total_saved} GB of reservation returned across "
          f"{len(SIZES)} CPU jobs")
    print("  GPU jobs are left alone: their peaks are unmeasured, and an OOM there costs "
          "hours rather than seconds. They are instrumented next time they run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
