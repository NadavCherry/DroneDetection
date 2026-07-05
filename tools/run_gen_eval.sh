#!/usr/bin/env bash
# Generalization eval driver: run one detector config over the ARD-MAV test set
# (unseen clips, SEEN dataset) and the whole NPS corpus (UNSEEN dataset), then
# score both with the center-distance harness. Usage:
#   tools/run_gen_eval.sh <weights.pt> <tag> <method> [extra detect args...]
# e.g. tools/run_gen_eval.sh work/runs/ardmav-tiled-640/weights/best.pt tiled-sahi yolo-ft-sahi --tile 640
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
W="$1"; TAG="$2"; METHOD="$3"; shift 3
EXTRA=("$@")
DETS=work/ext_datasets/dets
mkdir -p work/logs

for SET in ardmav nps; do
  echo ">>> detect [$TAG] over $SET ..."
  $PY tools/detect_batch.py --gt-dir work/ext_datasets/gt/$SET \
      --out-dir "$DETS/${SET}_${TAG}" --method "$METHOD" --weights "$W" \
      --conf 0.02 --stab off "${EXTRA[@]}" > "work/logs/detect_${SET}_${TAG}.log" 2>&1
done

echo; echo "==================== RESULTS [$TAG] ===================="
$PY tools/eval_external.py --gt-dir work/ext_datasets/gt/ardmav \
    --det-dir "$DETS/ardmav_${TAG}" --tau 12 --label "ARD-MAV test — unseen clips, SEEN dataset [$TAG]"
$PY tools/eval_external.py --gt-dir work/ext_datasets/gt/nps \
    --det-dir "$DETS/nps_${TAG}" --tau 12 --label "NPS — UNSEEN dataset [$TAG]"
