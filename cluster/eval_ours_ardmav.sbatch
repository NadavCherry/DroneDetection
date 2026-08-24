#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-12:00:00
#SBATCH --job-name sl_evalard
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/evalard-%A_%a.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=24G
#SBATCH --array=0-5
#SBATCH --requeue

### Our arm, ARD-MAV: full-frame tiled inference over the OFFICIAL 15-video test split,
### then a scorecard. Six tasks = 2 arms x 3 seeds. These weights already exist, so this
### array has no dependency and starts as soon as GPUs free up.
###
### Inference is full-frame even though training was tiled: the tiles are how this
### detector is trained, not how it is measured. YOLOMG is measured the same way, on the
### same clips, against the same GT.

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
    ARM=singleframe_ardmav; MODE=rgb;      SEED=$SLURM_ARRAY_TASK_ID
else
    ARM=temporal_ardmav;    MODE=temporal; SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi
RUN="${ARM}-s${SEED}"
W="$ROOT/work/runs/$RUN/weights/best.pt"
DET="$ROOT/work/det/ardmav/$RUN"
GT="$ROOT/work/ext_datasets/gt/ardmav"

echo "JOB $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID = $RUN ($MODE) on $(hostname)"
[ -f "$W" ] || { echo "*** NO WEIGHTS at $W"; exit 1; }
ls "$GT" | head -3

### --conf 0.001: the SAME floor the competitor is scored at. A floor difference between
### two arms truncates one precision-recall curve and not the other, which is exactly the
### mistake that once flattered our own control in this repo.
PYTHONPATH=. python -u tools/infer_tiled.py --weights "$W" --mode "$MODE" \
    --gt-dir "$GT" --out-dir "$DET" \
    --video-root "$ROOT/data/external/ard_mav/ARD-MAV/videos" \
    --conf 0.001 --batch 1 --device 0
rc=$?
echo "infer exit=$rc"
[ $rc -ne 0 ] && exit $rc

PYTHONPATH=. python -u tools/evaluate.py --dataset ardmav --official-split --model "$RUN" \
    --gt "$GT" --dets "$DET" --seed "$SEED" --weights "$W" \
    --out "$ROOT/work/scorecards/${RUN}.json"
rc=$?
### The scorecard has no "ap" field -- AP is computed on demand by pooled_ap, so the
### first version of this guard read d.get("ap"), got None every time, and would have
### aborted every job regardless of result. A guard that fires on success gets deleted.
### tools/check_scorecard.py computes AP the way tools/compare.py does, and also
### catches the partial failure the pooled number hides: a sequence scored over 0
### frames is being charged as a total miss while the others look fine.
PYTHONPATH=. python tools/check_scorecard.py "$ROOT/work/scorecards/${RUN}.json"
rc=$?

echo "=== RESULT ==="; echo "evaluate exit=$rc"
[ $rc -eq 0 ] && python -c "
import json;d=json.load(open('$ROOT/work/scorecards/${RUN}.json'))
print('AP:', d.get('ap'), ' sequences:', len(d.get('sequences',[])))"
exit $rc
