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
PYTHONPATH=. python -u tools/make_summary.py --out work/reports/SUMMARY.md
rc=$?
echo "=== RESULT ==="; echo "exit=$rc"
exit $rc
