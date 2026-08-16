#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-06:00:00
#SBATCH --job-name sl_loceval
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/loceval-%j.out
#SBATCH --gpus=1
#SBATCH --mem=24G

### Score all three arms on the held-out 10_06 flight and produce the within-sequence
### statistics. One job, not an array: 250 frames per arm is minutes, and keeping every
### arm on ONE GPU removes card-to-card variation from a comparison this small.
###
### --gpus=1 rather than rtx_3090:1 on purpose. The 3090s are saturated by the 69 h
### benchmark runs while 4090 and rtx_6000 nodes sit idle, and for a 640 px detector the
### card affects speed only, never the AP. Speed numbers come from cluster/bench.sbatch,
### which does pin one card type.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
GT="$ROOT/work/ext_datasets/gt/local"
VID="$ROOT/data/videos"
nvidia-smi --query-gpu=name --format=csv,noheader

rc=0
source activate speck
for S in 0 1 2; do
  for ARM in singleframe_local_ab temporal_local_ab; do
    MODE=rgb; [ "$ARM" = temporal_local_ab ] && MODE=temporal
    W="$ROOT/work/runs/${ARM}-s${S}/weights/best.pt"
    [ -f "$W" ] || { echo "*** missing $W"; rc=1; continue; }
    echo "=== ours: $ARM seed $S ($MODE) ==="
    PYTHONPATH=. python -u tools/infer_tiled.py --weights "$W" --mode "$MODE" \
        --gt-dir "$GT" --out-dir "$ROOT/work/det/local/${ARM}-s${S}" \
        --video-root "$VID" --conf 0.001 --batch 1 --device 0 || rc=1
  done
done

conda deactivate
source activate tph
for S in 0 1 2; do
  W="$ROOT/work/sota_runs/yolomg_local_seed${S}/weights/best.pt"
  [ -f "$W" ] || { echo "*** missing $W"; rc=1; continue; }
  echo "=== competitor: yolomg seed $S ==="
  PYTHONPATH="$ROOT" python -u tools/sota/infer_yolomg.py --weights "$W" \
      --gt-dir "$GT" --video-root "$VID" \
      --out "$ROOT/work/det/local/yolomg_local_seed${S}" \
      --imgsz 1280 --conf 0.001 --method "yolomg_local_seed${S}" || rc=1
done

conda deactivate
source activate speck
mkdir -p "$ROOT/work/reports"
for S in 0 1 2; do
  echo; echo "############ seed $S ############"
  PYTHONPATH=. python -u tools/compare_single_video.py \
      --gt "$GT/10_06.json" --baseline singleframe \
      --dets singleframe="$ROOT/work/det/local/singleframe_local_ab-s${S}" \
             temporal="$ROOT/work/det/local/temporal_local_ab-s${S}" \
             yolomg="$ROOT/work/det/local/yolomg_local_seed${S}" \
      --protocol specklock-centre --block-frames 30 \
      --out "$ROOT/work/reports/local_10_06_seed${S}.md" || rc=1
done

echo "=== RESULT ==="; echo "exit=$rc"
[ $rc -eq 0 ] && echo "LOCAL EVAL OK" || echo "LOCAL EVAL FAILED"
exit $rc
