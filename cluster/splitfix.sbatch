#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-00:20:00
#SBATCH --job-name sl_splitfix
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/splitfix-%j.out
#SBATCH --mem=4G

### NO --gpus. This shuffles lines in text files and touches empty filenames in a temp
### directory. Every image, label and mask it references already exists and none are
### touched. Holding a 4090 for it would be exactly the infrastructure-on-a-GPU waste the
### cluster rules forbid.
###
### Gates the leak-split training array, which is the decisive experiment for the
### YOLOMG/NPS discrepancy. The first attempt at that array died 15 s in, on all three
### tasks, because this step wrote RELATIVE paths into the yamls and image lists: YOLOMG
### trains from third_party/YOLOMG and ultralytics joins its own `path:` key on top, so a
### relative path resolves against the wrong root in two different ways. make_leak_split
### now resolves both ends and stats sampled entries before writing anything.

set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"
module load anaconda
source activate speck
export YOLO_AUTOINSTALL=false
export PYTHONPATH=$ROOT

echo "JOB $SLURM_JOB_ID on $(hostname)"

echo "########## 1. rebuild leak splits, absolute paths ##########"
for spec in "yolomg_nps yolomg disjoint-clips yolomg_nps_leak_ctl" \
            "yolomg_nps yolomg all-clips yolomg_nps_leak_all" \
            "nps_yolo_temporal ours disjoint-clips nps_temporal_leak_ctl"; do
    set -- $spec
    python -u tools/sota/make_leak_split.py \
        --dataset "work/ext_datasets/$1" --style "$2" --pool "$3" --seed 0 \
        --out "work/ext_datasets/$4" || { echo "*** split $4 FAILED ***"; exit 1; }
done

echo "########## 2. run THEIR split script, unmodified ##########"
### Evidence rather than code-reading for the two structural claims: their script cannot
### emit a held-out test set (trainval_percent = 1.0 means the else-branch never fires),
### and its train/val partition is per-frame across all videos rather than per-video.
python -u tools/sota/demo_their_split.py \
    --names-from work/ext_datasets/yolomg_nps \
    --out work/reports/protocol/their_split_demo.json || exit 1

echo "########## 3. stale YOLOv5 label caches ##########"
### YOLOv5 writes <list>.cache beside each image list and trusts it when the hash matches.
### These lists are brand new, but the *.cache files from the previous build sit in the
### same directory and a stale one would silently feed the old partition to the trainer --
### which would look exactly like "the leak split changed nothing".
find work/ext_datasets/yolomg_nps_leak_ctl work/ext_datasets/yolomg_nps_leak_all \
     work/ext_datasets/nps_temporal_leak_ctl -name "*.cache" -print -delete 2>/dev/null

echo "=== RESULT ==="
for d in yolomg_nps_leak_ctl yolomg_nps_leak_all nps_temporal_leak_ctl; do
    n=$(python -c "import json;print(json.load(open('work/ext_datasets/$d/SPLIT_AUDIT.json'))['clips_in_both'])")
    t=$(head -1 work/ext_datasets/$d/train.txt)
    [ "$n" -gt 0 ] || { echo "*** $d: $n overlapping clips -- not a leak split ***"; exit 1; }
    case "$t" in /*) ;; *) echo "*** $d: train.txt is not absolute: $t ***"; exit 1;; esac
    echo "$d: $n clips in both sides, paths absolute"
done
echo "SPLITS OK"
