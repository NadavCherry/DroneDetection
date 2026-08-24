#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name sl_trt
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/trt-%j.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=32G

### Only the TensorRT rows: the four PyTorch rows from job 20538504 are valid (produced
### BEFORE the env mutation) and are not re-measured. The export runs in a FRESH process
### with auto-install forbidden -- last time ultralytics upgraded triton inside the export
### process, kept running on stale modules, and segfaulted.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
source activate speck
export YOLO_AUTOINSTALL=false
OUT="$ROOT/work/bench"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

OURS="$ROOT/work/runs/temporal_ardmav-s0/weights/best.pt"
python -c "import tensorrt as t; print('tensorrt', t.__version__)" || { echo "no tensorrt"; exit 1; }

echo "=== export, fresh process ==="
PYTHONPATH=. python -u -c "
from ultralytics import YOLO
p = YOLO('$OURS').export(format='engine', imgsz=640, half=True, batch=8, device=0)
print('ENGINE:', p)" 2>&1 | tail -4
ENG="${OURS%.pt}.engine"
[ -f "$ENG" ] || { echo "*** still no engine -- TRT reported unavailable"; exit 1; }

for CONF in 0.001 0.25; do
  TAG=$(echo "$CONF" | tr -d .)
  echo "=== ours, TensorRT fp16, conf $CONF ==="
  PYTHONPATH=. python -u tools/sota/bench_both.py --arm ours --weights "$ENG" \
      --imgsz 640 --frame-hw 1080 1920 --conf "$CONF" --half \
      --out "$OUT/ours_trt_fp16_c${TAG}.json" || exit 1
done
echo "=== RESULT ==="
for f in "$OUT"/ours_trt_*.json; do
  python -c "
import json; d=json.load(open('$f'))
print('  %-24s p50 %7.1f ms  fps %5s' % ('$(basename $f .json)', d['p50_ms'], d['fps_at_p50']))"
done
echo "TRT DONE"
