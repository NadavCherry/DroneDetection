#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name sl_compare
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/compare-%j.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

### The statistics, on a CPU node. A paired bootstrap over 15 sequences carrying ~400k
### detections is real compute, and the Manager Node is not for that -- it is for launching
### and monitoring. Running it there once was a mistake; this is the fix.

set -uo pipefail
cd /home/cherryn/projects/SpeckLock
module load anaconda
source activate speck
export PYTHONPATH=.

echo "=== ARD-MAV: seed-matched singleframe vs temporal ==="
### Seed-matched pairs, not a pooled mix. The two arms share a seed, a split, a protocol
### and an evaluator, so the only thing left to explain a difference is the input
### representation -- which is the claim. Comparing s0 against s1 would fold seed variance
### into the effect and inflate it.
rc=0
for s in 0 1 2; do
    echo; echo "------------------------------------------------------ seed $s"
    python -u tools/compare.py \
        --scorecards work/scorecards/singleframe_ardmav-s$s.json \
                     work/scorecards/temporal_ardmav-s$s.json \
        --baseline singleframe_ardmav-s$s
    [ $? -ne 0 ] && rc=1
done

echo; echo "=== ARD-MAV vs the published bar (GLAD 0.80 @ IoU 0.5, 11-point AP) ==="
### No p-value is possible against a published scalar -- there is no second sample. The
### tool prints our interval, their point estimate, and every protocol difference, and
### marks the row NOT COMPARABLE if the protocols disagree.
for s in 0 1 2; do
    python -u tools/compare.py --scorecards work/scorecards/temporal_ardmav-s$s.json \
        --vs-published --dataset ardmav
    [ $? -ne 0 ] && rc=1
done

echo "=== RESULT ==="; echo "exit=$rc"
exit $rc
