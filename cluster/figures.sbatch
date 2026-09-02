#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-01:00:00
#SBATCH --job-name sl_figures
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/figures-%j.out
#SBATCH --mem=3G
### 3G, measured rather than guessed: matplotlib plus one video decode for fig5.
### This cluster sets JobAcctGatherType=(null), so sacct MaxRSS and sstat are both
### empty and nothing ever contradicted an over-request. The python step below runs
### under /usr/bin/time -v and prints its own peak RSS into this job's log, so the
### next person sizing it has a measurement instead of a habit.

### NO --gpus. Five figures, each read from a saved result file. fig5 decodes video and
### rebuilds the temporal stack on the CPU; nothing here benefits from CUDA.
set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"; module load anaconda; source activate speck
export YOLO_AUTOINSTALL=false; export PYTHONPATH=$ROOT
echo "JOB $SLURM_JOB_ID on $(hostname)"
/usr/bin/time -v python -u tools/make_paper_figures.py --results work --out docs/media/paper
rc=$?
echo "=== RESULT ==="; ls -la docs/media/paper/ 2>/dev/null; echo "exit=$rc"; exit $rc
