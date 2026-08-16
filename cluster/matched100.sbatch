#!/bin/bash
#SBATCH --partition main
#SBATCH --time 1-12:00:00
#SBATCH --job-name sl_e100
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/e100-%A_%a.out
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --array=0-11%3
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
### GPU BUDGET. This account is capped at a small number of concurrent GPUs (measured: 8
### pending with QOSMaxGRESPerUser at 8 held; treat 7 as the working ceiling). The cap is
### not the hazard -- SLURM enforces it. The hazard is ORDERING: the benchmark runs in
### cluster/yolomg_train.sbatch carry --requeue and take up to 69 h, so a preempted one
### re-enters the queue, and an unthrottled array of secondary jobs can take the freed slot
### ahead of it. A 20-hour run losing its place to a job I submitted myself is the kind of
### damage that is entirely avoidable. Hence the throttle below, and a dependency on the
### benchmark runs when this is submitted.
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

echo "JOB $SLURM_JOB_ID task $I = $CFG seed $SEED at 100 epochs -> work/runs_e100 on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

### --out-root, NOT --name. tools/train.py has no --name flag, and its run directory is
### out_root/<config>-s<seed> -- which for the same config and seed is the SAME directory
### the 30-epoch runs already occupy. Without a separate root these would have overwritten
### work/runs/temporal_ardmav-s0 and the finished ARD-MAV results with it. Separate roots
### also keep the two recipes distinguishable on disk, which is the point: both are
### reported, so both have to survive.
PYTHONPATH=. python -u tools/train.py --config "$CFG" --seed "$SEED" \
    --epochs 100 --out-root work/runs_e100 --vram-gib 24
rc=$?

echo "=== RESULT ==="
echo "exit=$rc"
W="$ROOT/work/runs_e100/${CFG}-s${SEED}/weights/best.pt"
[ -f "$W" ] && ls -la "$W" || echo "*** NO WEIGHTS ***"
[ $rc -eq 0 ] && echo "E100 OK (${CFG}-s${SEED})" || echo "E100 FAILED (${CFG}-s${SEED})"
exit $rc
