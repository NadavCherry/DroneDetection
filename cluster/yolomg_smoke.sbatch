#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-01:00:00
#SBATCH --job-name sl_mgsmoke
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/mgsmoke-%j.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=24G

### One GPU, a few minutes, before six GPUs are booked for seven hours each.
### Proves: the env runs their YOLOv5-era code, the dual-stream loader finds our images2
### masks AND their labels, and a step actually backprops. Anything that fails here would
### otherwise fail six times in parallel at 03:00.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
MG=$ROOT/third_party/YOLOMG
cd "$MG"
module load anaconda
source activate tph
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

DS=nps
R=$ROOT/work/ext_datasets/yolomg_$DS

echo "=== data present? ==="
for s in train val test; do echo "  $s: $(wc -l < "$R/$s.txt") imgs / $(wc -l < "$R/${s}2.txt") masks"; done
echo "--- labels reachable from the mask path (img2label_paths2 rule) ---"
python - <<PY
from pathlib import Path
m = Path("$R/train2.txt").read_text().split()[0]
lbl = m.replace("/images2/", "/labels/").rsplit(".",1)[0] + ".txt"
print("  mask :", m)
print("  label:", lbl, "EXISTS" if Path(lbl).exists() else "*** MISSING ***")
assert Path(lbl).exists(), "the mask stream cannot reach its labels"
PY

echo; echo "=== data yaml ==="
cat "$R/$DS.yaml"

echo; echo "=== PROBE: do labels and masks actually reach the training loop? ==="
### The 3-epoch run below prints "Labels 0" in its val table, while the dataset scan in
### the same log says "N found, 0 missing, 0 empty, 0 corrupt". One of those is cosmetic
### and the other means the competitor trains on nothing. Guessing is not good enough
### when six GPU-days ride on it, so count the targets coming out of THEIR dataloader.
cd "$MG"
PYTHONPATH="$MG" python -u "$ROOT/cluster/probe_yolomg_loader.py" \n    --data "$ROOT/work/smoke/smoke.yaml" --yolomg "$MG" --imgsz 640 --batch 4
prc=$?
if [ $prc -ne 0 ]; then echo "YOLOMG SMOKE FAILED (loader probe)"; exit $prc; fi

echo "=== 3-epoch smoke train (tiny subset via --epochs 3 on a truncated list) ==="
mkdir -p "$ROOT/work/smoke"
head -200 "$R/train.txt" > "$ROOT/work/smoke/t.txt"
head -200 "$R/train2.txt" > "$ROOT/work/smoke/t2.txt"
head -60  "$R/val.txt"   > "$ROOT/work/smoke/v.txt"
head -60  "$R/val2.txt"  > "$ROOT/work/smoke/v2.txt"
cat > "$ROOT/work/smoke/smoke.yaml" <<YML
train: $ROOT/work/smoke/t.txt
train2: $ROOT/work/smoke/t2.txt
val: $ROOT/work/smoke/v.txt
val2: $ROOT/work/smoke/v2.txt
test: $ROOT/work/smoke/v.txt
test2: $ROOT/work/smoke/v2.txt

nc: 1
names: ['Drone']
YML

python -u train.py --data "$ROOT/work/smoke/smoke.yaml" --cfg models/NPS_uav_s.yaml \
    --weights yolov5s.pt --batch-size 4 --epochs 3 --imgsz 640 \
    --name smoke --project "$ROOT/work/smoke/runs" --exist-ok --device 0
rc=$?
echo "=== train exit=$rc ==="
if [ $rc -eq 0 ]; then echo "YOLOMG SMOKE OK"; else echo "YOLOMG SMOKE FAILED"; fi
### The last command in an sbatch IS the job's exit status, and SLURM's afterok
### reads it. Ending on an echo makes every job report success -- which is how a
### smoke test that printed 'FAILED' still let six GPU jobs launch behind it.
exit $rc
