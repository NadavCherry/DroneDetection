#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-12:00:00
#SBATCH --job-name sl_dttrain
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/dttrain-%A_%a.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=24G
#SBATCH --array=0-11%4
#SBATCH --requeue

### The tap-spacing ablation: 4 spacings x 3 seeds on NPS-Drones.
###
### dt=6 is NOT retrained. Its three seeds already exist as work/runs_e100/temporal_nps-s*
### from the budget-matched campaign, produced by exactly the command below with exactly
### these hyperparameters. Retraining it would introduce a difference between the baseline
### and the arms that has nothing to do with dt.
###
### %4 throttle, not because 4 is optimal but because the account may hold at most 7 GPUs
### across ALL sessions and three are already committed to the leak-split experiment.
### SLURM queues the rest; waiting is fine, exceeding the limit is not. Check with
### ./cluster/gpu_budget.sh before raising this.
###
### 12 h: the 30-epoch NPS runs took ~1 h, so 100 epochs is ~3.5 h. The margin covers a
### slow node and a requeue.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
source activate speck
### ultralytics will pip-install "missing" packages MID-PROCESS and carry on with stale
### modules; it once upgraded triton for 729 s, segfaulted, and broke a sibling job that
### was importing torch from the same env at that moment.
export YOLO_AUTOINSTALL=false

DTS=(2 4 8 12)
DT=${DTS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))
DATA=$ROOT/work/ext_datasets/nps_yolo_temporal_dt${DT}/data.yaml
OUT=$ROOT/work/runs_dt
RUN=$OUT/dt${DT}-s${SEED}

echo "JOB $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "dt=$DT seed=$SEED -> $RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$DATA" ] || { echo "*** missing $DATA -- did dt_build run? ***"; exit 1; }

### Guard the comparison itself: an arm trained on a dataset whose BUILD.json says a
### different dt than this task believes would silently corrupt the whole curve, and the
### resulting figure would look perfectly smooth.
python - "$DATA" "$DT" <<'PY' || exit 1
import json, pathlib, sys
b = pathlib.Path(sys.argv[1]).parent / "BUILD.json"
got = json.loads(b.read_text()).get("dt")
want = int(sys.argv[2])
print(f"BUILD.json dt={got}, task expects {want}")
sys.exit(0 if got == want else 1)
PY

### Identical to the dt=6 baseline in cluster/matched100.sbatch except for --data:
### same config, same weights, same imgsz, same batch, same epochs, same seeds.
### Reusing the registered config rather than restating hyperparameters keeps them
### identical by construction instead of by my retyping them correctly.
PYTHONPATH=. python -u tools/train.py --config temporal_nps --seed "$SEED" \
    --data "$DATA" --epochs 100 --out-root "$OUT" --vram-gib 24 --force
rc=$?

### tools/train.py names its directory after the CONFIG, not the data, so every dt arm
### would land in the same 'temporal_nps-sN' directory and overwrite the previous one.
### Rename to the dt-tagged name immediately, and fail loudly rather than leave an
### ambiguous directory behind.
SRC=$OUT/temporal_nps-s${SEED}
if [ -d "$SRC" ]; then
    rm -rf "$RUN"
    mv "$SRC" "$RUN" || { echo "*** could not rename $SRC -> $RUN ***"; rc=1; }
fi

echo "=== RESULT ==="
W="$RUN/weights/best.pt"
if [ -f "$W" ]; then
    echo "DT TRAIN OK  dt=$DT seed=$SEED -> $W"
    python - "$RUN/results.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
h = {k.strip(): k for k in rows[0]}
best = max(rows, key=lambda r: float(r[h["metrics/mAP50(B)"]]))
print("  epochs=%d  best val mAP50=%.4f  mAP50-95=%.4f"
      % (len(rows), float(best[h["metrics/mAP50(B)"]]),
         float(best[h["metrics/mAP50-95(B)"]])))
PY
else
    echo "*** NO WEIGHTS (exit $rc) ***"; rc=1
fi
exit $rc
