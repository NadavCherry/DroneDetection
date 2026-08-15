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

### No hand-rolled video lookup here. It used to try five exact filenames per GT stem in
### bash and, on a miss, drop the sequence and carry on -- while our arm used a resolver
### that also handles containers, case and NPS's unpadded Clip_41.mov, and aborted on a
### miss. evaluate.py scores an unmatched sequence as a TOTAL MISS with its full ground
### truth still charged, so that asymmetry deflated the COMPETITOR's AP only: the
### direction that ends in a retraction. Both arms now call tools/video_paths.resolve_all,
### and infer_yolomg names its output after the GT stem rather than the video stem --
### resolving Clip_41.mov and then writing Clip_41.json would look fixed and still score
### every sequence as a miss.

source activate tph
PYTHONPATH="$ROOT" python -u tools/sota/infer_yolomg.py --weights "$W" \
    --gt-dir "$GT" --video-root "$VIDROOT" \
    --out "$DET" --imgsz 1280 --conf 0.001 --method "$RUN"
rc=$?
echo "infer exit=$rc"
[ $rc -ne 0 ] && exit $rc

conda deactivate
source activate speck
PYTHONPATH=. python -u tools/evaluate.py --dataset "$DS" --model "$RUN" $EXTRA \
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
exit $rc
