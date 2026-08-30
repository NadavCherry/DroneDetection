#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-02:00:00
#SBATCH --job-name sl_dtcmp
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/dtcmp-%j.out
#SBATCH --mem=16G

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
python -u tools/dt_compare.py --out work/reports/dt_test_ap.md
echo "=== RESULT ==="; echo "exit=$?"
