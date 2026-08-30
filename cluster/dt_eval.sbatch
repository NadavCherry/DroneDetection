#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-12:00:00
#SBATCH --job-name sl_dteval
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/dteval-%A_%a.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=24G
#SBATCH --array=0-11%2
#SBATCH --requeue

### FULL-FRAME HELD-OUT TEST AP FOR THE dt SWEEP.
###
### Item #1 established that this project's tile-level validation metric does not predict
### its full-frame test result: the same weights score val mAP50 0.941 and test AP 0.487.
### So a dt chosen on validation tiles is chosen on the weaker of the two available
### metrics, and the dt sweep cannot close on val alone. This scores every dt through the
### SAME path the headline numbers use -- tiled full-frame inference over the 10 held-out
### NPS test clips, then the unified evaluator.
###
### THE TRAP THIS JOB EXISTS TO AVOID: tools/infer_tiled.py takes --dt and DEFAULTS IT TO
### 6. The temporal stack at inference must use the same tap spacing the model was trained
### with; evaluating a dt=2 model through a dt=6 stack measures nothing. --dt is passed
### explicitly on every task and cross-checked against the run's own args.yaml below.
###
### dt=6 is NOT re-run. Its scorecards already exist from cluster/eval_e100.sbatch, which
### used this identical pipeline (same infer_tiled, same conf 0.001, same tile/overlap,
### same evaluator) on runs whose config parity with the dt runs was verified: same model,
### epochs, patience, batch, imgsz, seed. Re-running it would cost 3 GPU-hours to
### reproduce numbers we already hold.
###
### 12 tasks = 4 dt values x 3 seeds, throttled to 2 concurrent.

set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"
module load anaconda
source activate speck
export YOLO_AUTOINSTALL=false

I=$SLURM_ARRAY_TASK_ID
DTS=(2 4 8 12)
DT=${DTS[$(( I / 3 ))]}
SEED=$(( I % 3 ))
RUN="dt${DT}-s${SEED}"
TAG="dt${DT}_nps-s${SEED}"

W="$ROOT/work/runs_dt/$RUN/weights/best.pt"
GT="$ROOT/work/ext_datasets/gt/nps"
DET="$ROOT/work/det/nps_dt/$TAG"
VIDROOT="$ROOT/data/external/nps/Videos_x/Videos"

echo "JOB $SLURM_JOB_ID task $I = $TAG  (dt=$DT, seed=$SEED) on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
[ -f "$W" ] || { echo "*** NO WEIGHTS at $W"; exit 1; }

### Confirm the checkpoint really was trained at this dt, by reading the run's OWN
### args.yaml rather than trusting the directory name. A mislabelled arm would produce a
### plausible-looking curve that is simply wrong, and nothing downstream could detect it.
A="$ROOT/work/runs_dt/$RUN/args.yaml"
[ -f "$A" ] || { echo "*** no args.yaml for $RUN"; exit 1; }
DATA=$(grep -E "^data:" "$A" | awk '{print $2}')
case "$DATA" in
    *"_dt${DT}/"*) echo "  dt check OK: trained on $DATA" ;;
    *) echo "*** dt MISMATCH: task says dt=$DT but the run trained on $DATA"; exit 1 ;;
esac

PYTHONPATH=. python -u tools/infer_tiled.py --weights "$W" --mode temporal --dt "$DT" \
    --gt-dir "$GT" --out-dir "$DET" --video-root "$VIDROOT" \
    --conf 0.001 --batch 1 --device 0
rc=$?; [ $rc -ne 0 ] && exit $rc

### The detection files record the dt they were produced with. Verify it landed, so a
### future reader can tell these apart from the dt=6 set without trusting a filename.
PYTHONPATH=. python - "$DET" "$DT" <<'PY'
import json, sys, pathlib
det_dir, want = pathlib.Path(sys.argv[1]), int(sys.argv[2])
bad = []
for p in sorted(det_dir.glob("*.json")):
    got = (json.loads(p.read_text()).get("meta") or {}).get("dt")
    if got != want:
        bad.append((p.name, got))
if bad:
    print("*** dt not recorded as %d in: %s" % (want, bad[:5])); sys.exit(1)
print("  all %d detection files record dt=%d" % (len(list(det_dir.glob('*.json'))), want))
PY
rc=$?; [ $rc -ne 0 ] && exit $rc

PYTHONPATH=. python -u tools/evaluate.py --dataset nps --model "$TAG" \
    --gt "$GT" --dets "$DET" --seed "$SEED" --weights "$W" \
    --out "$ROOT/work/scorecards/${TAG}.json"
rc=$?; [ $rc -ne 0 ] && exit $rc

PYTHONPATH=. python tools/check_scorecard.py "$ROOT/work/scorecards/${TAG}.json"
rc=$?
echo "=== RESULT ==="; echo "exit=$rc"
[ $rc -eq 0 ] && echo "DT EVAL OK ($TAG)" || echo "DT EVAL FAILED ($TAG)"
exit $rc
