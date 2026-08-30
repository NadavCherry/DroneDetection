#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-06:00:00
#SBATCH --job-name sl_locft
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/locft-%A_%a.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=24G
#SBATCH --array=0-8%3
#SBATCH --requeue

### THE FAIR LOCAL COMPARISON: every arm fine-tunes from ITS OWN NPS-trained checkpoint
### instead of from COCO weights.
###
### Why this run exists. Trained from scratch on 463 frames, YOLOMG scored 0.005 and its
### own validation mAP never passed 0.0025. The diagnosis is not architectural -- its
### NPS_uav_s.yaml already carries a P2 (stride-4) head, and AutoAnchor reported a perfect
### 1.000 BPR, and its box loss fell from 0.143 to 0.041, so it was learning to localise.
### What collapsed was objectness: one ~10 px target in a 1280 px frame with a P2 head is
### roughly four positives among ~400k candidate positions, and on 463 images the
### background wins. It learns to predict nothing, everywhere.
###
### That is a small-data failure, and the remedy any practitioner would reach for is a
### better initialisation. So BOTH arms start from their own NPS-trained weights -- drone
### footage, trained by us, 8,019 frames -- and fine-tune on 07_05. Symmetric: each method
### gets its own prior, neither gets the other's.
###
### tools/train.py takes --out-root and derives the run name from the config; it has no
### --name. Passing one killed all six of our arms in one second with an argparse error.
###
### What is NOT changed: their hyperparameters, their image size, their epoch count.
### Re-weighting one arm's objectness loss to rescue it would be tuning the competitor,
### and tuning only one side is how a comparison stops meaning anything.

set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"
module load anaconda

I=$SLURM_ARRAY_TASK_ID
if   [ "$I" -lt 3 ]; then ARM=singleframe_local_ab; SEED=$I;       KIND=ours;  SRC=singleframe_nps
elif [ "$I" -lt 6 ]; then ARM=temporal_local_ab;    SEED=$((I-3)); KIND=ours;  SRC=temporal_nps
else                      ARM=yolomg_local_ft;      SEED=$((I-6)); KIND=comp;  SRC=yolomg_nps
fi
echo "JOB $SLURM_JOB_ID task $I = $ARM seed $SEED, initialised from ${SRC}-s${SEED}"
nvidia-smi --query-gpu=name --format=csv,noheader


### A GPU was allocated but torch could not see it on two of three runs last time
### ("Invalid CUDA '--device 0'"), which wasted the allocation and failed the array. Check
### once, up front, in whichever env is about to train.
check_gpu() {
    python - <<'PY' || { echo "*** allocated GPU is not usable by this env; failing fast"; exit 1; }
import sys, torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("torch.cuda unavailable:", torch.cuda.is_available(),
          torch.cuda.device_count(), file=sys.stderr); sys.exit(1)
print("  gpu ok:", torch.cuda.get_device_name(0))
PY
}

if [ "$KIND" = ours ]; then
    source activate speck
### ultralytics may pip-install 'missing' packages MID-PROCESS and carry on with
### stale modules -- its TRT export did exactly that, upgraded triton for 729 s,
### segfaulted itself, and broke a sibling job that was importing torch from the
### same env at that moment (undefined symbol: ncclCommResume). One shared env,
### mutated under a running reader. Never again:
export YOLO_AUTOINSTALL=false
    check_gpu
    INIT="$ROOT/work/runs/${SRC}-s${SEED}/weights/best.pt"
    [ -f "$INIT" ] || { echo "*** no init weights at $INIT"; exit 1; }
    PYTHONPATH=. python -u tools/train.py --config "$ARM" --seed "$SEED" \
        --weights "$INIT" --out-root work/runs_ft --vram-gib 24
    rc=$?
else
    source activate tph
    check_gpu
    cd "$ROOT/third_party/YOLOMG"
    python "$ROOT/cluster/patch_yolomg_seed.py" || { echo "seed patch failed"; exit 1; }
    INIT="$ROOT/work/sota_runs/yolomg_nps_seed${SEED}/weights/best.pt"
    [ -f "$INIT" ] || { echo "*** no init weights at $INIT"; exit 1; }
    R="$ROOT/work/ext_datasets/yolomg_local"
    NAME="yolomg_local_ft_seed${SEED}"
    LAST="$ROOT/work/sota_runs/$NAME/weights/last.pt"
    BEST="$ROOT/work/sota_runs/$NAME/weights/best.pt"
    RESULTS="$ROOT/work/sota_runs/$NAME/results.csv"
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
            --weights "$INIT" --batch-size 8 --epochs 100 --imgsz 1280 --seed "$SEED" \
            --name "$NAME" --project "$ROOT/work/sota_runs" --exist-ok --device 0
    fi
    rc=$?
fi

echo "=== RESULT ==="; echo "exit=$rc"
[ $rc -eq 0 ] && echo "LOCAL FT OK ($ARM seed $SEED)" || echo "LOCAL FT FAILED ($ARM seed $SEED)"
exit $rc
