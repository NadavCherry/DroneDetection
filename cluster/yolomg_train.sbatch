#!/bin/bash
#SBATCH --partition main
#SBATCH --time 1-00:00:00
#SBATCH --job-name sl_mgtrain
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/mgtrain-%A_%a.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=32G
#SBATCH --array=0-5
#SBATCH --requeue

### The SOTA arm: YOLOMG on 2 datasets x 3 seeds, under OUR splits.
###
### The recipe below is THEIRS, not ours -- 100 epochs at 1280 px, batch 8, from
### yolov5s.pt, per the command in their README. Ours trains 30 epochs at 640. Giving the
### competitor a shorter schedule because it is convenient for us would buy a win that
### does not survive the first reviewer, and the GPU-hours are cheaper than the retraction.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
MG=$ROOT/third_party/YOLOMG
cd "$MG"
module load anaconda
source activate tph

### Derived, not indexed from a literal list: a typo in a bash array is silent.
if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
    DS=nps;    SEED=$SLURM_ARRAY_TASK_ID
else
    DS=ardmav; SEED=$(( SLURM_ARRAY_TASK_ID - 3 ))
fi
R=$ROOT/work/ext_datasets/yolomg_$DS
NAME="yolomg_${DS}_seed${SEED}"

echo "JOB $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "DS=$DS SEED=$SEED -> $NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "data: $(wc -l < "$R/train.txt") train / $(wc -l < "$R/val.txt") val / $(wc -l < "$R/test.txt") test"

### models/ARD100_drone_s.yaml is referenced by their README but is NOT in the repo; the
### only dual-branch config they ship is NPS_uav_s.yaml, so both datasets use it. Recorded
### here because it is a deviation from their published ARD100 setup that we cannot close.
CFG=models/NPS_uav_s.yaml

[ -f yolov5s.pt ] || python -c "
import torch;torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt','yolov5s.pt')"

python -u train.py --data "$R/$DS.yaml" --cfg "$CFG" --weights yolov5s.pt \
    --batch-size 8 --epochs 100 --imgsz 1280 --seed "$SEED" \
    --name "$NAME" --project "$ROOT/work/sota_runs" --exist-ok --device 0
rc=$?

echo "=== RESULT ==="
echo "exit=$rc"
W="$ROOT/work/sota_runs/$NAME/weights/best.pt"
[ -f "$W" ] && ls -la "$W" || echo "*** NO WEIGHTS ***"
[ $rc -eq 0 ] && echo "YOLOMG TRAIN OK ($NAME)" || echo "YOLOMG TRAIN FAILED ($NAME)"
exit $rc
