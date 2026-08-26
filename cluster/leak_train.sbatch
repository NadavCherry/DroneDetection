#!/bin/bash
#SBATCH --partition main
#SBATCH --time 2-00:00:00
#SBATCH --job-name sl_leaktrain
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/leaktrain-%A_%a.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --requeue

### THE DECISIVE EXPERIMENT FOR THE YOLOMG / NPS DISCREPANCY.
###
### Everything about these runs is identical to the runs already on disk -- same images,
### same labels, same masks, same architecture, same 100 epochs at 1280 px, same batch,
### same weights to start from. The ONLY difference is which frames are called "train"
### and which are called "val": their split_train_val.py shuffles frames flat instead of
### holding out whole videos. Whatever AP moves is what that one line is worth.
###
### Reference points already measured, same code, their own val.py, video-disjoint split:
###   yolomg NPS best val mAP@.5 = 0.790 / 0.788 / 0.809  (seeds 0/1/2)
###   yolomg NPS test AP (our evaluator, pooled)          = 0.497 / 0.535 / 0.549
###   paper claims                                        = 0.95
###
### 3 GPUs of the 7-GPU allowance. rtx_4090 pinned rather than a bare count: --gpus=1 once
### landed a job on a GTX 1080 Ti (sm_61) on a pheno node, and our torch needs sm_75+.
### 32G matches the existing yolomg_train.sbatch, which ran this exact workload; ours at
### 640 px needs less but shares the array.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
export YOLO_AUTOINSTALL=false

case "$SLURM_ARRAY_TASK_ID" in
  0) ARM=yolomg_leak_ctl; DATA=$ROOT/work/ext_datasets/yolomg_nps_leak_ctl/nps.yaml ;;
  1) ARM=yolomg_leak_all; DATA=$ROOT/work/ext_datasets/yolomg_nps_leak_all/nps.yaml ;;
  2) ARM=ours_leak_ctl;   DATA=$ROOT/work/ext_datasets/nps_temporal_leak_ctl/data.yaml ;;
  *) echo "bad task id"; exit 1 ;;
esac

echo "JOB $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "ARM=$ARM  DATA=$DATA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f "$DATA" ] || { echo "*** missing $DATA -- did leak_build run? ***"; exit 1; }
python -c "
import json,sys
a=json.load(open('$(dirname "$DATA")/SPLIT_AUDIT.json'))
print('split audit:', a['n_train'],'train /',a['n_val'],'val, clips_in_both =',a['clips_in_both'])
sys.exit(0 if a['clips_in_both']>0 else 1)" || { echo "*** not a leak split ***"; exit 1; }

if [ "$ARM" = "ours_leak_ctl" ]; then
    source activate speck
    OUT=$ROOT/work/runs_leak
    ### Same config the disjoint run used (temporal_nps), only --data redirected. Reusing
    ### the registered config rather than writing a new one keeps every hyperparameter
    ### identical by construction instead of by my retyping them correctly.
    python -u tools/train.py --config temporal_nps --seed 0 --data "$DATA" \
        --out-root "$OUT" --batch 16 --device 0 --vram-gib 24 --force
    rc=$?
    W=$OUT/temporal_nps-s0/weights/best.pt
else
    source activate tph
    MG=$ROOT/third_party/YOLOMG
    cd "$MG"
    python "$ROOT/cluster/patch_yolomg_seed.py" || { echo "seed patch failed"; exit 1; }
    [ -f yolov5s.pt ] || python -c "
import torch;torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt','yolov5s.pt')"

    ### Resume rather than restart -- these are ~24 h runs under --requeue. --resume on a
    ### FINISHED run is an AssertionError in YOLOv5, not a no-op, so check the epoch count
    ### before deciding. This exact trap already took down an afterok chain once.
    RUN=$ROOT/work/sota_runs/$ARM
    DONE=0; [ -f "$RUN/results.csv" ] && DONE=$(tail -n +2 "$RUN/results.csv" | wc -l)
    if [ "$DONE" -ge 100 ] && [ -f "$RUN/weights/best.pt" ]; then
        echo "already complete: $DONE/100 epochs"; rc=0
    elif [ -f "$RUN/weights/last.pt" ]; then
        echo "RESUMING from $RUN/weights/last.pt"
        python -u train.py --resume "$RUN/weights/last.pt"; rc=$?
    else
        python -u train.py --data "$DATA" --cfg models/NPS_uav_s.yaml \
            --weights yolov5s.pt --batch-size 8 --epochs 100 --imgsz 1280 \
            --seed 0 --name "$ARM" --project "$ROOT/work/sota_runs" \
            --exist-ok --device 0
        rc=$?
    fi
    W=$RUN/weights/best.pt
fi

### A completed run that dies in ultralytics' post-training fuse has a correct model on
### disk; treating that exit code as failure once cancelled a whole evaluation chain.
if [ $rc -ne 0 ] && [ -f "$W" ]; then
    echo "NOTE: exit $rc but weights exist -- failure is after training, not in it."
    rc=0
fi

echo "=== RESULT ==="
echo "arm=$ARM exit=$rc"
[ -f "$W" ] && ls -la "$W" || echo "*** NO WEIGHTS ***"
exit $rc
