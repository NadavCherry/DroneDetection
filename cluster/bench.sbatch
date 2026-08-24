#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-06:00:00
#SBATCH --job-name sl_bench
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/bench-%j.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=32G

### Speed for both arms on ONE GPU in one job, so the numbers are comparable to each other.
### rtx_3090 pinned on purpose: FPS numbers must all come from the same card.
###
### TWO configurations per arm, labelled, never mixed:
###   conf 0.001  "as evaluated"  -- the floor the AP tables were produced at. NMS cost is
###                dominated by candidate count, so this is the slowest honest number.
###   conf 0.25   "as deployed"   -- a realistic runtime floor, same for both arms.
### Reporting only one of these would either understate our deployable speed (0.001) or
### measure a configuration that produced no published AP (0.25 alone).

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
### ultralytics may pip-install 'missing' packages MID-PROCESS and carry on with
### stale modules -- its TRT export did exactly that, upgraded triton for 729 s,
### segfaulted itself, and broke a sibling job that was importing torch from the
### same env at that moment (undefined symbol: ncclCommResume). One shared env,
### mutated under a running reader. Never again:
export YOLO_AUTOINSTALL=false
for CONF in 0.001 0.25; do
  TAG=$(echo "$CONF" | tr -d .)
  echo "=== ours, PyTorch fp16, conf $CONF ==="
  PYTHONPATH=. python -u tools/sota/bench_both.py --arm ours --weights "$OURS" \
      --imgsz 640 --frame-hw 1080 1920 --conf "$CONF" --half \
      --out "$OUT/ours_pt_fp16_c${TAG}.json" || echo "FAILED"
done

echo "=== TensorRT: install if absent, then export ==="
python -c "import tensorrt" 2>/dev/null || pip install --no-cache-dir "tensorrt-cu12" 2>&1 | tail -2
PYTHONPATH=. python -u -c "
from ultralytics import YOLO
m = YOLO('$OURS')
p = m.export(format='engine', imgsz=640, half=True, batch=8, device=0)
print('ENGINE:', p)" 2>&1 | tail -6
ENG="${OURS%.pt}.engine"
if [ -f "$ENG" ]; then
    for CONF in 0.001 0.25; do
      TAG=$(echo "$CONF" | tr -d .)
      echo "=== ours, TensorRT fp16, conf $CONF ==="
      PYTHONPATH=. python -u tools/sota/bench_both.py --arm ours --weights "$ENG" \
          --imgsz 640 --frame-hw 1080 1920 --conf "$CONF" --half \
          --out "$OUT/ours_trt_fp16_c${TAG}.json" || echo "FAILED"
    done
else
    echo "*** no engine produced -- the TRT rows are reported as unavailable, not guessed"
fi

conda deactivate
source activate tph
for CONF in 0.001 0.25; do
  TAG=$(echo "$CONF" | tr -d .)
  echo "=== yolomg, PyTorch fp16, conf $CONF ==="
  PYTHONPATH="$ROOT" python -u tools/sota/bench_both.py --arm yolomg --weights "$MG" \
      --imgsz 1280 --frame-hw 1080 1920 --conf "$CONF" --half \
      --out "$OUT/yolomg_pt_fp16_c${TAG}.json" || echo "FAILED"
done

echo "=== RESULT ==="
n=$(ls "$OUT"/*_c*.json 2>/dev/null | wc -l)
for f in "$OUT"/*_c*.json; do
  python -c "
import json; d=json.load(open('$f'))
print('  %-26s p50 %7.1f ms  fps %5s  (pre %6.1f, fwd %6.1f)' % (
  '$(basename $f .json)', d['p50_ms'], d['fps_at_p50'],
  d['preprocess']['p50_ms'], d['forward_only']['p50_ms']))"
done
if [ "$n" -ge 4 ]; then echo "BENCH DONE ($n rows)"; exit 0
else echo "BENCH INCOMPLETE ($n rows)"; exit 1; fi
