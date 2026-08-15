#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-12:00:00
#SBATCH --job-name sl_evalours
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/evalours-%A_%a.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=24G
#SBATCH --array=0-5
#SBATCH --requeue

### Our arm, NPS: full-frame tiled inference over the 10 Dogfight test clips, then a
### scorecard. Six tasks = 2 arms x 3 seeds, one GPU each.
###
### Inference is full-frame even though training was tiled: the tiles are how this
### detector is trained, not how it is measured. YOLOMG is measured the same way, on the
### same clips, against the same GT.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
source activate speck

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
    ARM=singleframe_nps; MODE=rgb;      SEED=$SLURM_ARRAY_TASK_ID
else
    ARM=temporal_nps;    MODE=temporal; SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi
RUN="${ARM}-s${SEED}"
W="$ROOT/work/runs/$RUN/weights/best.pt"
DET="$ROOT/work/det/nps/$RUN"
GT="$ROOT/work/ext_datasets/gt/nps"

echo "JOB $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID = $RUN ($MODE) on $(hostname)"
[ -f "$W" ] || { echo "*** NO WEIGHTS at $W"; exit 1; }
ls "$GT" | head -3

### --conf 0.001: the SAME floor the competitor is scored at. A floor difference between
### two arms truncates one precision-recall curve and not the other, which is exactly the
### mistake that once flattered our own control in this repo.
PYTHONPATH=. python -u tools/infer_tiled.py --weights "$W" --mode "$MODE" \
    --gt-dir "$GT" --out-dir "$DET" \
    --video-root "$ROOT/data/external/nps/Videos_x/Videos" \
    --conf 0.001 --device 0
rc=$?
echo "infer exit=$rc"
[ $rc -ne 0 ] && exit $rc

PYTHONPATH=. python -u tools/evaluate.py --dataset nps --model "$RUN" \
    --gt "$GT" --dets "$DET" --seed "$SEED" --weights "$W" \
    --out "$ROOT/work/scorecards/${RUN}.json"
rc=$?
### An AP of exactly 0 with a full set of sequences is almost never a model result -- it is
### a plumbing result. It is what a wrong --video-root produces, because evaluate.py
### correctly scores unmatched sequences as total misses rather than skipping them. Six NPS
### scorecards already came back at 0.000 that way, complete and well-formed and wrong.
### Refuse to report success on it; a real zero can be re-run with this guard removed.
PYTHONPATH=. python - "$ROOT/work/scorecards/${RUN}.json" <<'PYCHK'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
ap, seqs = d.get("ap"), d.get("sequences", [])
frames = sum(s.get("n_frames", 0) for s in seqs) if seqs else 0
print(f"  scorecard: AP={ap}  sequences={len(seqs)}  frames={frames}")
if not seqs:
    sys.exit("ABORT: scorecard has no sequences")
if frames == 0:
    sys.exit("ABORT: scorecard scored 0 frames -- inference produced nothing")
if ap in (None, 0, 0.0):
    sys.exit(f"ABORT: AP={ap} over {len(seqs)} sequences -- this is a plumbing failure, "
             f"not a model result")
PYCHK
rc=$?
echo "=== RESULT ==="; echo "evaluate exit=$rc"
[ $rc -eq 0 ] && python -c "
import json;d=json.load(open('$ROOT/work/scorecards/${RUN}.json'))
print('AP:', d.get('ap'), ' sequences:', len(d.get('sequences',[])))"
exit $rc
