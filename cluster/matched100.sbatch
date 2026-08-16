#!/bin/bash
#SBATCH --partition main
#SBATCH --time 1-12:00:00
#SBATCH --job-name sl_e100
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/e100-%A_%a.out
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --array=0-11
#SBATCH --requeue

### BUDGET-MATCHED runs: our arm at 100 epochs, the competitor's epoch count.
###
### The campaign so far gives each method its own published recipe -- ours 30 epochs at
### 640, YOLOMG 100 at 1280 -- which answers "which system is better as each is designed".
### It leaves one objection open, and it is a fair one: we take 103,590 gradient steps to
### their 219,400, so a loss could be blamed on budget rather than method, and a win is
### harder to interpret than it needs to be.
###
### These runs remove that. Same data, same splits, same everything, 100 epochs. Note the
### direction: matching the budget can only help us or leave us level, because we are the
### arm currently receiving LESS. That is why it is worth the GPU time and why the result
### is reportable either way -- if ours@100 does not beat ours@30, that is a real finding
### about our recipe and it goes in the table next to the rest.
###
### Both are reported. ours@30 is the shipped, deployable configuration; ours@100 is the
### budget-matched scientific comparison. Collapsing them into one number would hide which
### question was answered.
###
### --gpus=1 rather than rtx_3090:1: the 3090s are saturated by the 69 h benchmark runs
### while 4090 and rtx_6000 nodes idle. For a 640 px detector the card changes speed, not
### AP. Speed is measured separately by cluster/bench.sbatch, which does pin one card.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
source activate speck

I=$SLURM_ARRAY_TASK_ID
DS=ardmav;  [ "$I" -ge 6 ] && DS=nps
J=$(( I % 6 ))
if [ "$J" -lt 3 ]; then ARM=singleframe; SEED=$J; else ARM=temporal; SEED=$(( J - 3 )); fi
CFG="${ARM}_${DS}"
NAME="${CFG}-e100-s${SEED}"

echo "JOB $SLURM_JOB_ID task $I = $CFG seed $SEED at 100 epochs on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHONPATH=. python -u tools/train.py --config "$CFG" --seed "$SEED" \
    --epochs 100 --name "$NAME" --vram-gib 24
rc=$?

echo "=== RESULT ==="
echo "exit=$rc"
W="$ROOT/work/runs/$NAME/weights/best.pt"
[ -f "$W" ] && ls -la "$W" || echo "*** NO WEIGHTS ***"
[ $rc -eq 0 ] && echo "E100 OK ($NAME)" || echo "E100 FAILED ($NAME)"
exit $rc
