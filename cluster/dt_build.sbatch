#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-12:00:00
#SBATCH --job-name sl_dtbuild
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/dtbuild-%A_%a.out
#SBATCH --mem=8G
#SBATCH --array=0-3

### NO --gpus. Decoding video, ego-stabilising with KLT+RANSAC, tiling and writing JPEG
### is entirely CPU work -- there is no CUDA kernel anywhere in this path, so holding a
### 4090 for the hours it takes would be exactly the waste the cluster rules forbid.
### 8G covers the largest ring buffer: dt=12 holds 2*12+1 = 25 frames of 1920x1080x3
### (~150 MB) plus the tiles in flight.
###
### THE TAP-SPACING SWEEP. dt=6 -- taps at t-12, t-6, t, a 13-frame aperture -- is the
### founding constant of this project and the headline of every description of it. Its
### entire published justification is one prose row in docs/reports/round3-deliverables.md
### (dt=9 rejected on val 0.710 vs 0.719) whose training runs were discarded and cannot be
### re-derived. For the thesis of the work, that is not enough.
###
### dt=6 already exists at work/ext_datasets/nps_yolo_temporal and is NOT rebuilt: reusing
### the exact tiles the shipped numbers came from is what makes this a controlled sweep
### rather than five new datasets that differ in more than dt.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
source activate speck
export YOLO_AUTOINSTALL=false
export PYTHONPATH=$ROOT

DTS=(2 4 8 12)
DT=${DTS[$SLURM_ARRAY_TASK_ID]}
OUT=work/ext_datasets/nps_yolo_temporal_dt${DT}

echo "JOB $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "dt=$DT  (taps t-$((2*DT)), t-$DT, t; aperture $((2*DT+1)) frames)  -> $OUT"

### make_dataset_external wrote every temporal build to the SAME directory regardless of
### --dt until _temporal_name was added, so this sweep would have overwritten the dt=6
### tiles in place. Refuse to run against a build that still has the old behaviour.
python - <<'PY' || exit 1
import importlib.util, sys, pathlib
spec = importlib.util.spec_from_file_location(
    "mde", pathlib.Path("tools/make_dataset_external.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
if not hasattr(m, "_temporal_name"):
    print("*** make_dataset_external has no _temporal_name: a non-default --dt would "
          "OVERWRITE nps_yolo_temporal. Refusing. ***"); sys.exit(1)
assert m._temporal_name("nps_yolo_temporal", True, 6) == "nps_yolo_temporal"
assert m._temporal_name("nps_yolo_temporal", True, 2) == "nps_yolo_temporal_dt2"
print("dt naming guard OK")
PY

### Same strides, same tile size, same min-side, same split as the dt=6 build recorded in
### work/ext_datasets/nps_yolo_temporal/BUILD.json. Only --dt changes.
python -u tools/make_dataset_external.py --task nps-temporal-tiled \
    --dt "$DT" --tile 640 --min-side 0 --stride-train 4 --stride-val 10
rc=$?

echo "=== RESULT ==="
if [ -d "$OUT/images/train" ]; then
    echo "dt=$DT  train tiles: $(ls "$OUT/images/train" | wc -l)"
    echo "dt=$DT  val   tiles: $(ls "$OUT/images/val" | wc -l)"
    cat "$OUT/BUILD.json" 2>/dev/null
else
    echo "*** no output at $OUT ***"; rc=1
fi
### The dt=6 dataset must still be intact afterwards. If a future edit reintroduces the
### shared-directory bug, this is the line that catches it before the training array runs.
n6=$(ls work/ext_datasets/nps_yolo_temporal/images/train 2>/dev/null | wc -l)
echo "dt=6 baseline still has $n6 train tiles"
[ "$n6" -gt 10000 ] || { echo "*** dt=6 dataset damaged ***"; rc=1; }
echo "exit=$rc"
exit $rc
