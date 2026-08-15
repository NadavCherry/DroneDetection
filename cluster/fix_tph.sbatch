#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-01:00:00
#SBATCH --job-name sl_fixtph
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/fixtph-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

### The competitor's env is missing pkg_resources: setuptools 81 removed it, and every
### YOLOv5-of-that-vintage imports it from utils/general.py. CPU, because it is a pip
### install and an import check.
###
### This job's REAL job is the import check at the end. Installing a package proves
### nothing; importing train.py's whole dependency chain is what tells us a GPU is worth
### booking. It exits non-zero if that chain is broken, so the smoke test chained behind
### it never starts against a known-bad env.

set -euo pipefail          # -e ON: any failing step must kill this job, not be narrated
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT/third_party/YOLOMG"
module load anaconda
source activate tph

echo "=== before ==="
python -c "import setuptools; print('setuptools', setuptools.__version__)" || true
python -c "import pkg_resources; print('pkg_resources OK')" || echo "pkg_resources MISSING"

echo "=== pin setuptools below the release that dropped pkg_resources ==="
pip install -q "setuptools<81" "packaging" 2>&1 | tail -3

echo "=== after ==="
python -c "import setuptools; print('setuptools', setuptools.__version__)"
python -c "import pkg_resources; print('pkg_resources OK')"

echo "=== the check that actually matters: import what train.py imports ==="
python - <<'PY'
import importlib, sys, traceback
mods = ["utils.general", "utils.datasets", "utils.augmentations", "utils.loss",
        "utils.metrics", "utils.torch_utils", "utils.autoanchor",
        "models.common", "models.yolo", "models.experimental", "val"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ok   {m}")
    except Exception as e:
        bad.append((m, repr(e)))
        print(f"  FAIL {m}: {e!r}")
if bad:
    print("\nIMPORT CHAIN BROKEN:")
    for m, e in bad:
        print(f"  {m}: {e}")
    sys.exit(1)
print("\nimport chain clean")
PY

echo "=== can it build the dual-branch model at all? ==="
python - <<'PY'
import torch
from models.yolo import Model
m = Model("models/NPS_uav_s.yaml", ch=3, nc=1)
n = sum(p.numel() for p in m.parameters())
print(f"  NPS_uav_s.yaml built: {n/1e6:.2f} M parameters")
PY

echo "TPH ENV FIXED"
