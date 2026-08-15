#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name sl_bench
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/bench-%j.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=24G

### Speed for both arms on ONE GPU in one job, so the numbers are comparable to each
### other. A single job on purpose: two jobs could land on different nodes, and this
### cluster's GPUs are not all the same card.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
OUT="$ROOT/work/bench"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OURS="$ROOT/work/runs/temporal_ardmav-s0/weights/best.pt"
MG="$ROOT/work/sota_runs/yolomg_ardmav_seed0/weights/best.pt"

source activate speck
echo "=== ours, PyTorch fp32 ==="
PYTHONPATH=. python -u tools/sota/bench_both.py --arm ours --weights "$OURS" \
    --imgsz 640 --tiles 6 --out "$OUT/ours_pt_fp32.json" || echo "FAILED"
echo "=== ours, PyTorch fp16 ==="
PYTHONPATH=. python -u tools/sota/bench_both.py --arm ours --weights "$OURS" \
    --imgsz 640 --tiles 6 --half --out "$OUT/ours_pt_fp16.json" || echo "FAILED"

echo "=== ours, TensorRT FP16 ==="
### Export via ultralytics, then bench the .engine through the same code path, so the
### only thing that changed between the two rows is the backend.
PYTHONPATH=. python -u -c "
from ultralytics import YOLO
m = YOLO('$OURS')
p = m.export(format='engine', imgsz=640, half=True, batch=6, device=0)
print('ENGINE:', p)" 2>&1 | tail -5
ENG="${OURS%.pt}.engine"
if [ -f "$ENG" ]; then
    PYTHONPATH=. python -u tools/sota/bench_both.py --arm ours --weights "$ENG" \
        --imgsz 640 --tiles 6 --half --out "$OUT/ours_trt_fp16.json" || echo "FAILED"
else
    echo "*** no engine produced -- TensorRT row will be reported as unavailable, not guessed"
fi

conda deactivate
source activate tph
echo "=== yolomg, PyTorch fp32 (1280, mask32 included) ==="
if [ -f "$MG" ]; then
    PYTHONPATH="$ROOT" python -u tools/sota/bench_both.py --arm yolomg --weights "$MG" \
        --imgsz 1280 --frame-hw 1080 1920 --out "$OUT/yolomg_pt_fp32.json" || echo "FAILED"
    PYTHONPATH="$ROOT" python -u tools/sota/bench_both.py --arm yolomg --weights "$MG" \
        --imgsz 1280 --frame-hw 1080 1920 --half --out "$OUT/yolomg_pt_fp16.json" || echo "FAILED"
else
    echo "*** no YOLOMG weights yet"
fi

echo "=== RESULT ==="
ls -la "$OUT"
for f in "$OUT"/*.json; do echo "--- $(basename $f)"; python -c "
import json,sys;d=json.load(open('$f'))
print('  p50 %.2f ms  p95 %.2f  p99 %.2f  fps@p50 %s' % (d['p50_ms'],d['p95_ms'],d['p99_ms'],d['fps_at_p50']))"; done
n=$(ls "$OUT"/*.json 2>/dev/null | wc -l)
if [ "$n" -ge 3 ]; then echo "BENCH DONE ($n measurements)"; exit 0
else echo "BENCH INCOMPLETE: only $n measurement files"; exit 1; fi
### The last command in an sbatch IS the job's exit status, and SLURM's afterok
### reads it. Ending on an echo makes every job report success -- which is how a
### smoke test that printed 'FAILED' still let six GPU jobs launch behind it.

