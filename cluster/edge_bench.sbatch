#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-03:00:00
#SBATCH --job-name sl_edgebench
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/edgebench-%j.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=24G

### ITEM #5: locate and benchmark the "ultra-fast" edge variant.
###
### The project remembers a ~100+ FPS model. It exists as EDGE-RT / RT-C -- one YOLOv8n-P2
### on the three-moment stack, full-frame at 1280, no tiling and no proposal stage -- but
### the 100+ figure is the SAME checkpoint exported at 640, not a smaller network. Verified
### by hashing every tensor blob in both .pt archives: identical digest, 441 tensors.
###
### Two things the published 74 / 84.8 / 104 fps numbers assumed and this job does not:
###   * a TensorRT engine, which the repo does not ship (.gitignore excludes *.engine
###     because they are architecture-specific) -- so a fresh clone silently runs the .pt;
###   * an RTX 5070 Laptop. This runs on a 4090 and rebuilds the engine for it, so NOTHING
###     here is comparable to the published figures. That is the point: the instruction was
###     not to quote an old number without reproducing it.
###
### Accuracy and speed come from the same pass over the same video, so the two axes of the
### table cannot drift apart.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
source activate speck
export YOLO_AUTOINSTALL=false
export PYTHONPATH=$ROOT

echo "JOB $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PT=$ROOT/final/edge_rt/edge_n1280.pt
[ -f "$PT" ] || { echo "*** missing $PT"; exit 1; }

### Build engines for THIS card. An engine built elsewhere would not load, and one built
### here is not transferable -- which is exactly why the repo ships none.
ENGDIR=$ROOT/work/edge_engines
mkdir -p "$ENGDIR"
for SZ in 1280 640; do
    E="$ENGDIR/edge_n${SZ}.engine"
    if [ -f "$E" ]; then echo "engine exists: $E"; continue; fi
    echo "### exporting TensorRT FP16 engine at imgsz $SZ"
    python - "$PT" "$SZ" "$E" <<'PY'
import shutil, sys, pathlib
from ultralytics import YOLO
pt, sz, dest = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
### Export writes beside the .pt and names the artifact after its stem, so both sizes
### would collide on one filename. Copy to a per-size stem first, then move the result.
stem = dest.parent / f"edge_src_{sz}.pt"
shutil.copy(pt, stem)
out = YOLO(str(stem)).export(format="engine", imgsz=sz, half=True, device=0, batch=1)
p = pathlib.Path(out)
p.replace(dest)
print("engine ->", dest)
PY
    [ -f "$E" ] || echo "*** engine export FAILED at $SZ (continuing with .pt only)"
done

ENGDIR_OK=$ENGDIR
ARGS=(--weights "$PT" --video "$ROOT/data/videos/10_06.mp4"
      --gt "$ROOT/realtime/work/gt_1006_v2.json"
      --imgsz 1280 640 --warmup 20 --out "$ROOT/work/reports/edge")
ARGS+=(--engine-dir "$ENGDIR")

python -u tools/bench_edge.py "${ARGS[@]}"
rc=$?
echo "=== RESULT ==="; echo "exit=$rc"
exit $rc
