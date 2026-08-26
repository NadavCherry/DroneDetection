#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name sl_sweep
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/sweep-%j.out
#SBATCH --mem=12G

### CPU only, and no --gpus at all: this re-scores detection JSONs that already exist.
### Nothing here touches CUDA. 12G covers the largest detection file (~180 MB) with room
### for the per-sequence record lists; measured peak on a trial run was under 6G.

set -uo pipefail
cd /home/cherryn/projects/SpeckLock
module load anaconda
source activate speck
mkdir -p work/reports/protocol

rc=0
for S in 0 1 2; do
    PYTHONPATH=. python -u tools/protocol_sweep.py \
        --gt work/ext_datasets/gt/nps --dets "work/det/nps/yolomg_nps_seed${S}" \
        --label "YOLOMG NPS seed ${S}" \
        --out "work/reports/protocol/nps_yolomg_s${S}.md" > /dev/null || rc=1
    PYTHONPATH=. python -u tools/protocol_sweep.py \
        --gt work/ext_datasets/gt/nps --dets "work/det/nps/temporal_nps-e100-s${S}" \
        --label "OURS temporal NPS e100 seed ${S}" \
        --out "work/reports/protocol/nps_ours_s${S}.md" > /dev/null || rc=1
done

### ARD-MAV too: if the same axes explain a gap there, the finding generalises rather
### than being an NPS quirk.
PYTHONPATH=. python -u tools/protocol_sweep.py \
    --gt work/ext_datasets/gt/ardmav --dets work/det/ardmav/yolomg_ardmav_seed0 \
    --label "YOLOMG ARD-MAV seed 0" \
    --out work/reports/protocol/ardmav_yolomg_s0.md > /dev/null || rc=1

echo "=== HEADLINE NUMBERS ==="
for f in work/reports/protocol/*.md; do
    echo "--- $(basename "$f" .md)"
    grep -E "our evaluator|their published protocol|Total gap" "$f"
done
echo "=== RESULT ==="; echo "exit=$rc"
exit $rc
