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
from pathlib import Path

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
    a = ap.parse_args()

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
        mosaic=0.3,
        close_mosaic=12,
        scale=0.25,
        translate=0.08,
        fliplr=0.5,
        degrees=0.0,
        shear=0.0,
        mixup=0.0,
        erasing=0.0,
        cos_lr=True,
        patience=25,
        workers=8,
        plots=True,
    )


if __name__ == "__main__":
    main()
