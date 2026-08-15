#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-12:00:00
#SBATCH --job-name sl_evalsota
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/evalsota-%A_%a.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=24G
#SBATCH --array=0-5
#SBATCH --requeue

### The competitor, scored the same way we are: same test videos, same GT files, same
### conf floor, same evaluator. Inference runs in the competitor's env (tph), scoring runs
### in ours (speck) -- the detection JSON is the only thing that crosses between them,
### which is exactly why tools/evaluate.py takes JSONs instead of weights.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
    DS=nps;    SEED=$SLURM_ARRAY_TASK_ID
    VIDROOT="$ROOT/data/external/nps/Videos_x/Videos"
    EXTRA=""
else
    DS=ardmav; SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
    VIDROOT="$ROOT/data/external/ard_mav/ARD-MAV/videos"
    EXTRA="--official-split"
fi
RUN="yolomg_${DS}_seed${SEED}"
W="$ROOT/work/sota_runs/$RUN/weights/best.pt"
GT="$ROOT/work/ext_datasets/gt/$DS"
DET="$ROOT/work/det/$DS/$RUN"

echo "JOB $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID = $RUN on $(hostname)"
[ -f "$W" ] || { echo "*** NO WEIGHTS at $W"; exit 1; }

### The test videos are named by the GT files, so the two arms cannot diverge on which
### sequences were scored -- a mismatch there is the classic way a comparison table lies.
VIDS=""
for g in "$GT"/*.json; do
    stem=$(basename "$g" .json)
    for ext in mp4 mov MOV MP4 avi; do
        [ -f "$VIDROOT/$stem.$ext" ] && { VIDS="$VIDS $VIDROOT/$stem.$ext"; break; }
    done
done
echo "videos to score: $(echo $VIDS | wc -w) (GT files: $(ls "$GT"/*.json | wc -l))"

source activate tph
PYTHONPATH="$ROOT" python -u tools/sota/infer_yolomg.py --weights "$W" \
    --videos $VIDS --out "$DET" --imgsz 1280 --conf 0.001 --method "$RUN"
rc=$?
echo "infer exit=$rc"
[ $rc -ne 0 ] && exit $rc

conda deactivate
source activate speck
PYTHONPATH=. python -u tools/evaluate.py --dataset "$DS" --model "$RUN" $EXTRA \
    --gt "$GT" --dets "$DET" --seed "$SEED" --weights "$W" \
    --out "$ROOT/work/scorecards/${RUN}.json"
rc=$?
echo "=== RESULT ==="; echo "evaluate exit=$rc"
exit $rc
