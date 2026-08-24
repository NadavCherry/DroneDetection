#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-14:00:00
#SBATCH --job-name sl_mgbuild
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/mgbuild-%A_%a.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --requeue

### Build YOLOMG's two-stream data (RGB + mask32) on OUR splits. CPU only: this is
### homography estimation and JPEG writing, and a GPU would sit idle through all of it.
### Two tasks so ARD-MAV and NPS build side by side rather than end to end.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
source activate speck
### ultralytics may pip-install 'missing' packages MID-PROCESS and carry on with
### stale modules -- its TRT export did exactly that, upgraded triton for 729 s,
### segfaulted itself, and broke a sibling job that was importing torch from the
### same env at that moment (undefined symbol: ncclCommResume). One shared env,
### mutated under a running reader. Never again:
export YOLO_AUTOINSTALL=false

case "$SLURM_ARRAY_TASK_ID" in
  0) DS=ardmav ;;
  1) DS=nps ;;
  *) echo "bad task"; exit 1 ;;
esac
echo "JOB $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID = $DS on $(hostname)"
echo "HEAD: $(git log --oneline -1)"

### --stride 4 matches the stride our own builders use for the training arm, so neither
### method sees more supervision than the other. Eval splits use 3x, also matching ours.
### NOT `... | tail`: a pipeline's $? is the LAST stage's, so `rc` would be tail's 0 and
### a failed build would report success. Log in full, then read the builder's own status.
PYTHONPATH=. python -u tools/sota/build_yolomg.py --dataset "$DS" --stride 4
rc=$?

R="$ROOT/work/ext_datasets/yolomg_$DS"
echo "=== RESULT ==="
echo "exit=$rc"
for s in train val test; do
    ni=$(wc -l < "$R/$s.txt" 2>/dev/null || echo 0)
    nm=$(wc -l < "$R/${s}2.txt" 2>/dev/null || echo 0)
    echo "  $s: images=$ni masks=$nm"
done
du -sh "$R" 2>/dev/null
[ -f "$R/BUILD.json" ] && cat "$R/BUILD.json" | head -20

### Prove the pairing on disk rather than trusting the builder that just wrote it.
PYTHONPATH=. python - <<PY
### (assert below); its status is captured into vrc, not discarded
from pathlib import Path
r = Path("$R")
for s in ("train","val","test"):
    a=[Path(x).name for x in (r/f"{s}.txt").read_text().split()]
    b=[Path(x).name for x in (r/f"{s}2.txt").read_text().split()]
    assert a==b, f"{s}: MISALIGNED"
    print(f"  {s}: {len(a)} pairs, line-aligned OK")
PY
vrc=$?
if [ "$rc" -eq 0 ] && [ "$vrc" -eq 0 ]; then echo "YOLOMG BUILD DONE ($DS)"
else echo "YOLOMG BUILD FAILED ($DS): build rc=$rc verify rc=$vrc"; fi
### The last command in an sbatch IS the job's exit status, and SLURM's afterok
### reads it. Ending on an echo makes every job report success -- which is how a
### smoke test that printed 'FAILED' still let six GPU jobs launch behind it.
exit $(( rc || vrc ))
