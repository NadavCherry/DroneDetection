#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-12:00:00
#SBATCH --job-name sl_npstrain
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/npstrain-%A_%a.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=24G
#SBATCH --array=0-5
#SBATCH --requeue

### 6 runs on 6 GPUs: 2 arms x 3 seeds on NPS-Drones, Dogfight split.
### Derived by arithmetic rather than indexed from a literal list -- a typo in a bash
### array is silent and would mislabel a whole arm.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
source activate speck
### ultralytics may pip-install 'missing' packages MID-PROCESS and carry on with
### stale modules -- its TRT export did exactly that, upgraded triton for 729 s,
### segfaulted itself, and broke a sibling job that was importing torch from the
### same env at that moment (undefined symbol: ncclCommResume). One shared env,
### mutated under a running reader. Never again:
export YOLO_AUTOINSTALL=false

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
    ARM=singleframe_nps; SEED=$SLURM_ARRAY_TASK_ID
else
    ARM=temporal_nps;    SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi
echo "JOB $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "ARM=$ARM SEED=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

### --vram-gib 24: the 3090 has 24 GB and the guard's default budget is 8, which would
### print a spurious LIKELY OOM for a batch measured at 7.08 GiB. The guard warns, it
### does not block, but a false warning in a log is a future person's wasted hour.
python -u tools/train.py --config "$ARM" --seed "$SEED" \
    --batch 16 --device 0 --vram-gib 24 --force
rc=$?

W=$ROOT/work/runs/${ARM}-s${SEED}/weights/best.pt
if [ -f "$W" ]; then
    echo "TRAIN OK  $ARM seed $SEED  -> $W"
else
    echo "NO WEIGHTS -- run failed (exit $rc)"; exit 1
fi
