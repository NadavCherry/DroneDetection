#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-03:00:00
#SBATCH --job-name sl_sizecurve
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/sizecurve-%j.out
#SBATCH --mem=24G

### NO --gpus. Every stage here reads JSON detection files that already exist and does
### arithmetic on them. Nothing benefits from CUDA, so nothing here may hold a GPU.
### 24G covers the largest arm: NPS at conf 0.001 is a few hundred thousand detection
### records held across three seeds at once.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
source activate speck
export YOLO_AUTOINSTALL=false
export PYTHONPATH=$ROOT

echo "JOB $SLURM_JOB_ID on $(hostname)"

########################################################################
echo "########## 0. suite -- metrics.py gained gt_size/gt_sizes/ap_by_bins ##########"
### Gate everything behind the suite. dronedet/metrics.py is the module every number in
### this project comes out of, and the size curve below is the first consumer of fields
### just added to it. A regression there would not announce itself -- it would quietly
### produce a plausible-looking curve.
python -m pytest -q --no-header -x 2>&1 | tail -15
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "*** SUITE FAILED -- refusing to produce numbers from a broken evaluator ***"
    exit 1
fi

########################################################################
echo "########## 1. accuracy vs target size ##########"
### One evaluator, one confidence floor, one matching rule for every arm; the arms differ
### only in which detection files they read. Both bin sets are produced: AI-TOD's four
### standard bins for comparability with the small-object literature, and the finer
### mission bins because AI-TOD lumps 8-16 px into one cell and that interval is exactly
### where the interesting part of the curve lives.
run_curve () {   # $1 dataset  $2 gt  $3..  arm triples
    local ds="$1" gt="$2"; shift 2
    local args=()
    for a in "$@"; do
        set -- $(echo "$a" | tr ":" " ")
        args+=(--arm "$1" "$gt" "work/det/$2/$3")
    done
    for b in mission aitod; do
        echo "----- $ds / $b"
        python -u tools/size_curve.py --dataset "$ds" --bins "$b" \
            --rule iou --iou 0.5 --conf 0.001 --resamples 1000 \
            "${args[@]}" --out work/reports/size_curve \
            || echo "*** size_curve $ds/$b FAILED ***"
    done
}

run_curve nps work/ext_datasets/gt/nps \
    "ours:nps:temporal_nps-e100-s0"  "ours:nps:temporal_nps-e100-s1"  "ours:nps:temporal_nps-e100-s2" \
    "yolomg:nps:yolomg_nps_seed0"    "yolomg:nps:yolomg_nps_seed1"    "yolomg:nps:yolomg_nps_seed2" \
    "ours-single:nps:singleframe_nps-e100-s0" "ours-single:nps:singleframe_nps-e100-s1" \
    "ours-single:nps:singleframe_nps-e100-s2"

run_curve ardmav work/ext_datasets/gt/ardmav \
    "ours:ardmav:temporal_ardmav-e100-s0" "ours:ardmav:temporal_ardmav-e100-s1" \
    "ours:ardmav:temporal_ardmav-e100-s2" \
    "yolomg:ardmav:yolomg_ardmav_seed0"   "yolomg:ardmav:yolomg_ardmav_seed1" \
    "yolomg:ardmav:yolomg_ardmav_seed2" \
    "ours-single:ardmav:singleframe_ardmav-e100-s0" "ours-single:ardmav:singleframe_ardmav-e100-s1" \
    "ours-single:ardmav:singleframe_ardmav-e100-s2"

run_curve local_ft work/ext_datasets/gt/local \
    "ours:local_ft:temporal_local_ab-s0" "ours:local_ft:temporal_local_ab-s1" \
    "ours:local_ft:temporal_local_ab-s2" \
    "yolomg:local_ft:yolomg_local_ft_seed0" "yolomg:local_ft:yolomg_local_ft_seed1" \
    "yolomg:local_ft:yolomg_local_ft_seed2" \
    "ours-single:local_ft:singleframe_local_ab-s0" "ours-single:local_ft:singleframe_local_ab-s1" \
    "ours-single:local_ft:singleframe_local_ab-s2"

echo "=== RESULT ==="
ls -la work/reports/size_curve/ 2>/dev/null
echo "SIZECURVE DONE"
