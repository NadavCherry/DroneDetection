#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-03:00:00
#SBATCH --job-name sl_repair2
#SBATCH --output /home/cherryn/projects/SpeckLock/cluster_logs/repair2-%j.out
#SBATCH --gpus=rtx_3090:1
#SBATCH --mem=24G

### Second repair of speck, and this one is on a GPU node because the first repair's
### verification was CPU-only -- imports passed, torch.save existed, and the env still
### could not run a convolution. "GET was unable to find an engine" on the first fp16 conv
### is cuDNN failing, and the mechanism fits the quota event exactly: reinstalling torch
### left nvidia-cudnn-cu12 and its siblings untouched because their dist-info survived,
### while their .so payloads were truncated by the same full disk that ate torch itself.
### pip trusts dist-info; the loader trusts nothing.
###
### So: force-reinstall the whole chain, then verify with actual GPU convs in both
### precisions -- the operation that failed, not a proxy for it.

set -uo pipefail
ROOT=/home/cherryn/projects/SpeckLock
cd "$ROOT"
module load anaconda
source activate speck
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "=== force-reinstall torch + full nvidia dependency chain ==="
pip install --force-reinstall --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -4

echo; echo "=== verify: the op that actually failed, both precisions ==="
python - <<'PY'
import sys, torch
print("  torch", torch.__version__, "| cudnn", torch.backends.cudnn.version(),
      "| gpu", torch.cuda.get_device_name(0))
for half in (False, True):
    x = torch.randn(8, 3, 640, 640, device="cuda")
    m = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
    if half:
        x, m = x.half(), m.half()
    y = m(x); torch.cuda.synchronize()
    print(f"  conv2d {'fp16' if half else 'fp32'} OK  ->", tuple(y.shape))
# and the real model end to end, once
from ultralytics import YOLO
mdl = YOLO("work/runs/temporal_ardmav-s0/weights/best.pt")
import numpy as np
img = np.random.randint(0, 255, (640, 640, 3), np.uint8)
mdl([img], imgsz=640, conf=0.25, verbose=False, device=0, half=True)
print("  ultralytics fp16 forward OK")
PY
rc=$?
echo "=== RESULT ==="; echo "exit=$rc"
[ $rc -eq 0 ] && echo "SPECK GPU-VERIFIED" || echo "SPECK STILL BROKEN"
exit $rc
