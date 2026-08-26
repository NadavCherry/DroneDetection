#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-01:00:00
#SBATCH --job-name sl_theirval
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/theirval-%j.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=16G

### THE END-TO-END CONTROL for the YOLOMG / NPS discrepancy.
###
### tools/protocol_sweep.py already replicates their 101-point AP bit-exactly (checked
### against utils/metrics.compute_ap on random PR curves) and says the whole protocol is
### worth about +0.01. But that check re-scores detections produced by OUR inference
### wrapper. If tools/sota/infer_yolomg.py letterboxes, rescales or NMSes differently from
### their val.py, the sweep would be measuring our bug rather than their metric.
###
### So: THEIR val.py, THEIR dataloader, THEIR letterbox, THEIR NMS, THEIR AP -- on OUR
### trained checkpoints over OUR video-disjoint 757-image test list. If this lands near
### 0.50, our inference stack is faithful and the split is the only variable left.
###
### --workers 0 is not a style choice. The first attempt (job 20578844) died in 33 s with
### ConnectionResetError inside torch's pin_memory thread: their loader spawns 8 workers
### that each open both image streams, and the shared-memory segment on the GPU nodes is
### too small for it. Zero workers is slower and this job is minutes long either way.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
MG=$ROOT/third_party/YOLOMG
cd "$MG"
module load anaconda
source activate tph
export YOLO_AUTOINSTALL=false

echo "JOB $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
R=$ROOT/work/ext_datasets/yolomg_nps
echo "test images: $(wc -l < "$R/test.txt")"

rc=0
for S in 0 1 2; do
    W=$ROOT/work/sota_runs/yolomg_nps_seed$S/weights/best.pt
    echo "########## THEIR val.py, our checkpoint, seed $S ##########"
    [ -f "$W" ] || { echo "*** missing $W ***"; rc=1; continue; }
    ### --task test reads the 'test'/'test2' keys of the yaml, i.e. clips 41-50, the same
    ### 757 images our evaluator scores. conf 0.001 and iou 0.6 are the values our own
    ### infer_yolomg.py uses, so the two paths differ only in whose code runs.
    python -u val.py --data "$R/nps.yaml" --weights "$W" --task test \
        --conf-thres 0.001 --iou-thres 0.6 --imgsz 1280 --batch-size 8 \
        --workers 0 --device 0 --exist-ok \
        --project "$ROOT/work/theirval" --name "seed$S" || rc=1
done

echo "=== RESULT ==="
echo "exit=$rc"
