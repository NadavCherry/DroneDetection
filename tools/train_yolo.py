"""Fine-tune a P2-headed YOLO on the video dataset.

Recipe follows the tiny-object research notes:
  * P2 (stride-4) detection head  -> yolov8s-p2 config, COCO weights transferred
  * native-resolution training    -> imgsz 1280 (the video is 1280 wide)
  * reduced mosaic, mild scale jitter; mosaic disabled for the last epochs
  * this is the *scene-tuning demo* of the recipe -- the heavy fine-tune on
    real multi-scene data uses the same script with a bigger dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="work/dataset/data.yaml")
    ap.add_argument("--model", default="yolov8s-p2.yaml")
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--name", default="ft-p2-1280")
    ap.add_argument("--patience", type=int, default=25)
    # NWD (Normalized Wasserstein Distance) for tiny-object assignment+regression
    ap.add_argument("--nwd", action="store_true", help="enable NWD blend (dronedet.nwd)")
    ap.add_argument("--nwd-assign-ratio", type=float, default=0.5)
    ap.add_argument("--nwd-assign-c", type=float, default=16.0)
    ap.add_argument("--nwd-loss-ratio", type=float, default=0.5)
    ap.add_argument("--nwd-loss-c", type=float, default=2.0)
    ap.add_argument("--mc", action="store_true",
                    help="enable >3-channel .npy training (dronedet.mc_data)")
    # temporal-stack channels are stabilized grays at t-12/t-6/t; hue/sat
    # jitter would remix them semantically, so v3 runs pass 0 0
    ap.add_argument("--hsv", type=float, nargs=3, default=[0.015, 0.7, 0.4],
                    metavar=("H", "S", "V"))
    ap.add_argument("--scale", type=float, default=0.25,
                    help="scale-jitter fraction (raise for scale invariance)")
    ap.add_argument("--mosaic", type=float, default=0.3)
    a = ap.parse_args()

    if a.mc:
        from dronedet.mc_data import enable_multichannel
        enable_multichannel()

    if a.nwd:
        from dronedet.nwd import enable_nwd
        enable_nwd(a.nwd_assign_ratio, a.nwd_assign_c, a.nwd_loss_ratio, a.nwd_loss_c)

    model = YOLO(a.model).load(a.weights)
    model.train(
        data=a.data,
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=a.batch,
        name=a.name,
        # absolute path: the global ultralytics settings.json redirects
        # relative project dirs under another workspace's runs_dir
        project=str(Path("work/runs").resolve()),
        mosaic=a.mosaic,
        close_mosaic=12,
        scale=a.scale,
        translate=0.08,
        fliplr=0.5,
        degrees=0.0,
        shear=0.0,
        mixup=0.0,
        erasing=0.0,
        hsv_h=a.hsv[0],
        hsv_s=a.hsv[1],
        hsv_v=a.hsv[2],
        cos_lr=True,
        patience=a.patience,
        workers=4,
        plots=True,
    )


if __name__ == "__main__":
    main()
