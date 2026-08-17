#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-12:00:00
#SBATCH --job-name sl_evale100
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/evale100-%A_%a.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=24G
#SBATCH --array=0-11%3
#SBATCH --requeue

### Score the budget-matched runs. Their weights live in work/runs_e100/ under the SAME
### names as the 30-epoch runs in work/runs/, so every scorecard here is written with an
### -e100- tag: two runs of one config differing only in budget must never collide in
### work/scorecards, or the table silently reports one where it means the other.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
source activate speck

I=$SLURM_ARRAY_TASK_ID
DS=ardmav; [ "$I" -ge 6 ] && DS=nps
J=$(( I % 6 ))
if [ "$J" -lt 3 ]; then ARM=singleframe; MODE=rgb; SEED=$J
else ARM=temporal; MODE=temporal; SEED=$(( J - 3 )); fi
RUN="${ARM}_${DS}-s${SEED}"
TAG="${ARM}_${DS}-e100-s${SEED}"

if [ "$DS" = ardmav ]; then
    VIDROOT="$ROOT/data/external/ard_mav/ARD-MAV/videos"; EXTRA="--official-split"
else
    VIDROOT="$ROOT/data/external/nps/Videos_x/Videos"; EXTRA=""
fi
GT="$ROOT/work/ext_datasets/gt/$DS"
W="$ROOT/work/runs_e100/$RUN/weights/best.pt"
DET="$ROOT/work/det/$DS/$TAG"

echo "JOB $SLURM_JOB_ID task $I = $TAG ($MODE) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
[ -f "$W" ] || { echo "*** NO WEIGHTS at $W"; exit 1; }

PYTHONPATH=. python -u tools/infer_tiled.py --weights "$W" --mode "$MODE" \
    --gt-dir "$GT" --out-dir "$DET" --video-root "$VIDROOT" \
    --conf 0.001 --batch 1 --device 0
rc=$?
[ $rc -ne 0 ] && exit $rc

PYTHONPATH=. python -u tools/evaluate.py --dataset "$DS" --model "$TAG" $EXTRA \
    --gt "$GT" --dets "$DET" --seed "$SEED" --weights "$W" \
    --out "$ROOT/work/scorecards/${TAG}.json"
rc=$?
[ $rc -ne 0 ] && exit $rc

PYTHONPATH=. python tools/check_scorecard.py "$ROOT/work/scorecards/${TAG}.json"
rc=$?
echo "=== RESULT ==="; echo "exit=$rc"
[ $rc -eq 0 ] && echo "E100 EVAL OK ($TAG)" || echo "E100 EVAL FAILED ($TAG)"
exit $rc
