#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-08:00:00
#SBATCH --job-name sl_regensum
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/regensum-%j.out
#SBATCH --mem=16G
### NO --gpus. Regenerate work/reports/SUMMARY.md so that it records the settings that
### produced it. The previous file recorded none, so its Monte-Carlo columns (CIs,
### p-values) could not be reproduced -- a regeneration gave identical effect sizes and
### identical verdicts beside CIs and p-values differing in the third decimal, with no way
### to tell a settings difference from a real change.
###
### 2000 resamples, not the argparse default of 10000. 10000 is a perfectly sound setting
### and would give slightly tighter Monte-Carlo columns; it is simply not achievable in
### this environment -- it finished in neither a 4 h nor a 10 h job on a 2-core cluster
### node, both timing out having written nothing. 2000 is what completes here, and it is
### what the rest of this project already uses: dt_compare, whose numbers are committed,
### runs at 2000, and size_curve at 1000-2000. The value is stamped into the output, so
### the file is exactly reproducible at whatever setting produced it.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck; export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname)"
python -u tools/make_summary.py --scorecards work/scorecards --n-resamples 2000 \
    --out work/reports/SUMMARY.md
echo "make_summary exit=$?"
head -8 work/reports/SUMMARY.md
echo "=== RESULT ==="
