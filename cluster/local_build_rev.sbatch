#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name sl_locbuildr
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/locbuildr-%A_%a.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=0-2

### REVERSE direction: 10_06 -> 07_05, so the eight labelled birds land in the TEST set. Three builds in parallel on CPU --
### our single-frame arm, our temporal arm, and the competitor's two-stream data.
### 548 annotated training frames, so these are minutes, not hours.

set -uo pipefail
ROOT=${SPECKLOCK_ROOT:-/home/cherryn/projects/SpeckLock}
cd "$ROOT"
module load anaconda
source activate speck
### ultralytics may pip-install 'missing' packages MID-PROCESS and carry on with
### stale modules -- its TRT export did exactly that, upgraded triton for 729 s,
### segfaulted itself, and broke a sibling job that was importing torch from the
### same env at that moment (undefined symbol: ncclCommResume). One shared env,
### mutated under a running reader. Never again:
export YOLO_AUTOINSTALL=false
echo "HEAD: $(git log --oneline -1)"

case "$SLURM_ARRAY_TASK_ID" in
  0) WHAT=ours-singleframe ;;
  1) WHAT=ours-temporal ;;
  2) WHAT=competitor ;;
  *) echo "bad task"; exit 1 ;;
esac
echo "JOB $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID = $WHAT on $(hostname)"

### The 10_06 ground truth, needed by every arm. Idempotent, so all three may write it.
PYTHONPATH=. python -u tools/make_dataset_external.py --task local-rev-gt
grc=$?

case "$WHAT" in
  ours-singleframe)
    PYTHONPATH=. python -u tools/make_dataset_external.py --task local-rev-tiled \
        --tile 640 --stride-train 1 --stride-val 4 --min-side 0
    rc=$? ;;
  ours-temporal)
    PYTHONPATH=. python -u tools/make_dataset_external.py --task local-rev-temporal \
        --tile 640 --stride-train 1 --stride-val 4 --min-side 0 --dt 6
    rc=$? ;;
  competitor)
    PYTHONPATH=. python -u tools/sota/build_yolomg.py --dataset local_rev --stride 1
    rc=$? ;;
esac

echo "=== RESULT ==="
echo "gt exit=$grc  build exit=$rc"
if [ $rc -eq 0 ] && [ $grc -eq 0 ]; then echo "LOCAL BUILD OK ($WHAT)"; else echo "LOCAL BUILD FAILED ($WHAT)"; fi
exit $(( rc || grc ))
