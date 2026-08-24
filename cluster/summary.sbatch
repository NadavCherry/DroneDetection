#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-02:00:00
#SBATCH --job-name sl_summary
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/summary-%j.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

### The result tables. A job rather than an interactive command so the output survives the
### ssh drops that have been killing these calls, and so the bootstrap runs off the login
### node where it belongs.
set -uo pipefail
cd /home/cherryn/projects/SpeckLock
module load anaconda
source activate speck
### ultralytics may pip-install 'missing' packages MID-PROCESS and carry on with
### stale modules -- its TRT export did exactly that, upgraded triton for 729 s,
### segfaulted itself, and broke a sibling job that was importing torch from the
### same env at that moment (undefined symbol: ncclCommResume). One shared env,
### mutated under a running reader. Never again:
export YOLO_AUTOINSTALL=false
PYTHONPATH=. python -u tools/make_summary.py --out work/reports/SUMMARY.md
rc=$?
echo "=== RESULT ==="; echo "exit=$rc"
exit $rc
