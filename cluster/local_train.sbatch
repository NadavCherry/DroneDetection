#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-08:00:00
#SBATCH --job-name sl_loctrain
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/loctrain-%A_%a.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=24G
#SBATCH --array=0-8%2
#SBATCH --requeue

### The project's own task, 3 arms x 3 seeds: ours-temporal, ours-singleframe, YOLOMG.
### Train on 07_05 (465 frames), test on the held-out 10_06 flight.
###
### Throttled to 2 concurrent: the benchmark runs hold most of a small GPU quota and
### must never queue behind this. See cluster/matched100.sbatch for the reasoning.
###
### 0-2 ours-singleframe   3-5 ours-temporal   6-8 competitor
### Derived by arithmetic, never indexed from a literal list: a typo in a bash array is
### silent and would mislabel a whole arm.

set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"
module load anaconda

I=$SLURM_ARRAY_TASK_ID
if   [ "$I" -lt 3 ]; then ARM=singleframe_local_ab; SEED=$I;            KIND=ours
elif [ "$I" -lt 6 ]; then ARM=temporal_local_ab;    SEED=$((I-3));      KIND=ours
else                      ARM=yolomg_local;         SEED=$((I-6));      KIND=competitor
fi
echo "JOB $SLURM_JOB_ID task $I = $ARM seed $SEED ($KIND) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "$KIND" = ours ]; then
    source activate speck
### ultralytics may pip-install 'missing' packages MID-PROCESS and carry on with
### stale modules -- its TRT export did exactly that, upgraded triton for 729 s,
### segfaulted itself, and broke a sibling job that was importing torch from the
### same env at that moment (undefined symbol: ncclCommResume). One shared env,
### mutated under a running reader. Never again:
export YOLO_AUTOINSTALL=false
    ### No --name: tools/train.py has no such flag, and passing one aborts before the first
    ### batch with an argparse error -- which is how all six of these failed in one second.
    ### The run directory is cfg.run_name(seed) = "<config>-s<seed>", already exactly what
    ### cluster/local_eval.sbatch looks for.
    PYTHONPATH=. python -u tools/train.py --config "$ARM" --seed "$SEED" --vram-gib 24
    rc=$?
else
    ### The competitor's published recipe, unchanged: 100 epochs at 1280 from yolov5s.pt.
    ### 465 training frames makes this minutes rather than the 69 h the ARD-MAV runs take,
    ### so there is no reason to shorten it and every reason not to.
    source activate tph
    cd "$ROOT/third_party/YOLOMG"
    python "$ROOT/cluster/patch_yolomg_seed.py" || { echo "seed patch failed"; exit 1; }
    R="$ROOT/work/ext_datasets/yolomg_local"
    NAME="yolomg_local_seed${SEED}"
    LAST="$ROOT/work/sota_runs/$NAME/weights/last.pt"
    BEST="$ROOT/work/sota_runs/$NAME/weights/best.pt"
    RESULTS="$ROOT/work/sota_runs/$NAME/results.csv"
    [ -f yolov5s.pt ] || python -c "
import torch;torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt','yolov5s.pt')"
    ### --resume on a finished run is an AssertionError in YOLOv5 ("training to 100
    ### epochs is finished, nothing to resume"), not a no-op -- so a requeued task
    ### whose earlier attempt completed FAILS, and takes the afterok chain with it.
    DONE=0; [ -f "$RESULTS" ] && DONE=$(tail -n +2 "$RESULTS" 2>/dev/null | wc -l)
    if [ "$DONE" -ge 100 ] && [ -f "$BEST" ]; then
        echo "already complete: $DONE/100 epochs, best.pt present -- nothing to do"
        true
    elif [ -f "$LAST" ]; then
        echo "RESUMING from $LAST"; python -u train.py --resume "$LAST"
    else
        python -u train.py --data "$R/local.yaml" --cfg models/NPS_uav_s.yaml \
            --weights yolov5s.pt --batch-size 8 --epochs 100 --imgsz 1280 --seed "$SEED" \
            --name "$NAME" --project "$ROOT/work/sota_runs" --exist-ok --device 0
    fi
    rc=$?
fi

echo "=== RESULT ==="
echo "exit=$rc"
if [ $rc -eq 0 ]; then echo "LOCAL TRAIN OK ($ARM seed $SEED)"; else echo "LOCAL TRAIN FAILED ($ARM seed $SEED)"; fi
exit $rc
