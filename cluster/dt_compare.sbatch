#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-02:00:00
#SBATCH --job-name sl_dtcmp
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/dtcmp-%j.out
#SBATCH --mem=3G
### 3G, measured rather than guessed: 15 arms, half of make_summary's 30.
### This cluster sets JobAcctGatherType=(null), so sacct MaxRSS and sstat are both
### empty and nothing ever contradicted an over-request. The python step below runs
### under /usr/bin/time -v and prints its own peak RSS into this job's log, so the
### next person sizing it has a measurement instead of a habit.


### NO --gpus. Paired bootstrap + permutation over scorecards that already exist.
### Run as a job rather than over ssh because the login-node session kept dropping
### mid-computation and left no output to diagnose.
set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
source activate speck
export YOLO_AUTOINSTALL=false
export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname)"
/usr/bin/time -v python -u tools/dt_compare.py --out work/reports/dt_test_ap.md --json work/reports/dt_sweep.json
echo "=== RESULT ==="; echo "exit=$?"
