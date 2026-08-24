#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-06:00:00
#SBATCH --job-name sl_locevalft
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/locevalft-%j.out
### GPU TYPE IS NOT OPTIONAL. An earlier version asked for --gpus=1 to exploit idle
### capacity, and SLURM duly handed it a GTX 1080 Ti: the pheno nodes advertise BOTH
### gpu:rtx_3090:1 and gpu:gtx_1080:1, so "any GPU" can mean a 2016 card. Our torch build
### supports sm_75 and up; the 1080 Ti is sm_61, and the run died on the first forward.
### The competitor's env (torch 2.4.1) tolerated the same node, so the failure looked like
### a bug in our arm rather than in the request.
###
### rtx_4090 rather than rtx_3090: sm_89, fully supported, ~25 idle nodes, and it leaves
### the 3090s to the 69 h benchmark runs instead of competing with them.
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=24G

### The FAIR local comparison, scored: every arm here was fine-tuned from its own
### NPS-trained checkpoint rather than from COCO, because from scratch on 463 frames the
### competitor never converged (its own val mAP peaked at 0.0025). Same videos, same
### split, same evaluator as the from-scratch run in cluster/local_eval.sbatch -- the two
### reports sit side by side and the difference between them IS the finding about how much
### initialisation matters at this data scale.
###
### Score all three arms on the held-out 10_06 flight and produce the within-sequence
### statistics. One job, not an array: 250 frames per arm is minutes, and keeping every
### arm on ONE GPU removes card-to-card variation from a comparison this small.
###

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
GT="$ROOT/work/ext_datasets/gt/local"
VID="$ROOT/data/videos"
nvidia-smi --query-gpu=name --format=csv,noheader

rc=0
source activate speck
### ultralytics may pip-install 'missing' packages MID-PROCESS and carry on with
### stale modules -- its TRT export did exactly that, upgraded triton for 729 s,
### segfaulted itself, and broke a sibling job that was importing torch from the
### same env at that moment (undefined symbol: ncclCommResume). One shared env,
### mutated under a running reader. Never again:
export YOLO_AUTOINSTALL=false
for S in 0 1 2; do
  for ARM in singleframe_local_ab temporal_local_ab; do
    MODE=rgb; [ "$ARM" = temporal_local_ab ] && MODE=temporal
    W="$ROOT/work/runs_ft/${ARM}-s${S}/weights/best.pt"
    [ -f "$W" ] || { echo "*** missing $W"; rc=1; continue; }
    echo "=== ours: $ARM seed $S ($MODE) ==="
    PYTHONPATH=. python -u tools/infer_tiled.py --weights "$W" --mode "$MODE" \
        --gt-dir "$GT" --out-dir "$ROOT/work/det/local_ft/${ARM}-s${S}" \
        --video-root "$VID" --conf 0.001 --batch 1 --device 0 || rc=1
  done
done

conda deactivate
source activate tph
for S in 0 1 2; do
  W="$ROOT/work/sota_runs/yolomg_local_ft_seed${S}/weights/best.pt"
  [ -f "$W" ] || { echo "*** missing $W"; rc=1; continue; }
  echo "=== competitor: yolomg seed $S ==="
  PYTHONPATH="$ROOT" python -u tools/sota/infer_yolomg.py --weights "$W" \
      --gt-dir "$GT" --video-root "$VID" \
      --out "$ROOT/work/det/local_ft/yolomg_local_ft_seed${S}" \
      --imgsz 1280 --conf 0.001 --method "yolomg_local_ft_seed${S}" || rc=1
done

conda deactivate
source activate speck
mkdir -p "$ROOT/work/reports"
for S in 0 1 2; do
  echo; echo "############ seed $S ############"
  PYTHONPATH=. python -u tools/compare_single_video.py \
      --gt "$GT/10_06.json" --baseline singleframe \
      --dets singleframe="$ROOT/work/det/local_ft/singleframe_local_ab-s${S}" \
             temporal="$ROOT/work/det/local_ft/temporal_local_ab-s${S}" \
             yolomg="$ROOT/work/det/local_ft/yolomg_local_ft_seed${S}" \
      --protocol specklock-centre --block-frames 30 \
      --out "$ROOT/work/reports/local_ft_10_06_seed${S}.md" || rc=1
done

echo "=== RESULT ==="; echo "exit=$rc"
[ $rc -eq 0 ] && echo "LOCAL EVAL OK" || echo "LOCAL EVAL FAILED"
exit $rc
