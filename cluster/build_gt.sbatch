#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-03:00:00
#SBATCH --job-name sl_gt
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/gt-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

### Per-video ground-truth JSONs for the ARD-MAV official test split. CPU: it is XML
### parsing and a frame count. Both arms are scored against these exact files.
set -uo pipefail
cd /home/cherryn/projects/SpeckLock
module load anaconda
source activate speck
PYTHONPATH=. python -u tools/make_dataset_external.py --task ardmav-gt
rc=$?
echo "=== RESULT ==="; echo "exit=$rc"
echo "ardmav gt files: $(ls work/ext_datasets/gt/ardmav 2>/dev/null | wc -l)"
echo "nps gt files:    $(ls work/ext_datasets/gt/nps 2>/dev/null | wc -l)"
exit $rc
