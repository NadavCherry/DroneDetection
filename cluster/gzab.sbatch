#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-06:00:00
#SBATCH --job-name sl_gzab
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/gzab-%j.out
#SBATCH --mem=5G
### 5G, measured rather than guessed: two sequential make_summary runs, 3.21 GB each.
### This cluster sets JobAcctGatherType=(null), so sacct MaxRSS and sstat are both
### empty and nothing ever contradicted an over-request. The python step below runs
### under /usr/bin/time -v and prints its own peak RSS into this job's log, so the
### next person sizing it has a measurement instead of a habit.

### A/B the SCORECARD INPUT PATH, holding everything else fixed.
###
### The question is whether reading gzipped scorecards produces the same numbers as
### reading plain ones. Comparing against the committed SUMMARY.md cannot answer it: that
### file records no --n-resamples and no commit, so a mismatch would be unattributable --
### it could be compression, a different resample count, or code that has changed since.
###
### So: the same tool, the same settings, the same seed, run twice over the same 42
### scorecards in the two storage forms. Equivalence is equivalence at any resample count,
### and the bootstrap is seeded, so 500 is as decisive as 10000 and finishes. The default
### 10000 did not complete in 4 h.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck; export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname)"
N=500
echo "### A: plain .json"
/usr/bin/time -v python -u tools/make_summary.py --scorecards work/_jsononly --n-resamples $N \
    --out work/_gzonly/SUMMARY_from_json.md || exit 1
echo "### B: gzipped .json.gz"
/usr/bin/time -v python -u tools/make_summary.py --scorecards work/_gzonly --n-resamples $N \
    --out work/_gzonly/SUMMARY_from_gz.md || exit 1
python - <<PY
a = open("work/_gzonly/SUMMARY_from_json.md", encoding="utf-8").read()
b = open("work/_gzonly/SUMMARY_from_gz.md",   encoding="utf-8").read()
print()
if a == b:
    print("IDENTICAL: the gzipped input path produces a byte-identical summary")
    print("           (%d chars compared)" % len(a))
else:
    print("DRIFT DETECTED between the .json and .json.gz input paths")
    import difflib
    for i, line in enumerate(difflib.unified_diff(a.splitlines(), b.splitlines(),
                                                  "from_json", "from_gz", lineterm="")):
        print(" ", line)
        if i > 40: print("  ..."); break
PY
echo "=== RESULT ==="
