#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-02:00:00
#SBATCH --job-name sl_repair
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/repair-%j.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G

### Repair the `speck` environment. torch and pandas vanished from its site-packages --
### torch/ left holding an empty lib/ with no __init__.py and no dist-info, pandas gone
### entirely, while ultralytics, numpy, cv2 and scipy survived. The most likely cause is
### the home quota filling mid-write; it is not attributable to the cleanup, which touched
### only ~/.cache and ~/.conda/pkgs/*.tar.bz2.
###
### Versions come from requirements.txt, not from whatever pip resolves today: every
### ARD-MAV and NPS number in this campaign was produced under torch 2.11.0 / ultralytics
### 8.4.84, and a silently different stack would make the remaining runs incomparable to
### the ones already banked.
###
### CPU: this is a pip install and an import check.

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

echo "=== before ==="
python -c "import sys; print(' ', sys.executable)"
SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo "  site-packages: $SP"
ls "$SP/torch" 2>/dev/null | head -3; echo "  torch entries: $(ls $SP/torch 2>/dev/null | wc -l)"

### Remove the husk first: a directory with no __init__.py still shadows a real install as
### a namespace package, so pip would install underneath it and the import would stay broken.
rm -rf "$SP/torch" "$SP/torchvision" "$SP"/torch-*.dist-info "$SP"/torchvision-*.dist-info

echo; echo "=== reinstalling at the pinned versions ==="
pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -4
pip install --no-cache-dir pandas 2>&1 | tail -2

echo; echo "=== verify, from a neutral cwd so nothing can shadow the import ==="
cd /tmp
python - <<'PY'
import importlib, sys
ok = True
for m, want in (("torch", "2.11.0"), ("torchvision", "0.26.0"),
                ("ultralytics", "8.4.84"), ("numpy", None),
                ("cv2", None), ("scipy", None), ("pandas", None)):
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "?")
        bad = want and not str(v).startswith(want)
        print(f"  {m:12s} {v}" + ("   *** EXPECTED " + want if bad else ""))
        ok &= not bad
    except Exception as e:
        print(f"  {m:12s} FAILED: {e!r}"); ok = False
import torch
print(f"  torch.save present: {hasattr(torch, 'save')}")
ok &= hasattr(torch, "save")
sys.exit(0 if ok else 1)
PY
rc=$?

echo; echo "=== the repo's own import path ==="
cd "$ROOT" && PYTHONPATH=. python -c "
from tools.infer_tiled import tile_origins
from benchmarks.scorecard import pooled_ap
print('  repo imports OK')" 2>&1 | tail -3

echo "=== RESULT ==="; echo "exit=$rc"
[ $rc -eq 0 ] && echo "SPECK REPAIRED" || echo "SPECK STILL BROKEN"
exit $rc
