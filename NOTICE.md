# Licence and third-party notices

## This repository

Copyright © 2026 Nadav Cherry.

The code in this repository is licensed under the **GNU Affero General Public
License, version 3** — see [`LICENSE`](LICENSE) for the full text.

AGPL-3.0 rather than a permissive licence because the detection and pursuit
pipelines import [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO,
which is itself AGPL-3.0. Releasing this work under a permissive licence would
have made a promise that the dependency does not allow me to keep. If you need
these pipelines under different terms, the Ultralytics licence is the constraint
to resolve first, not this one.

## Third-party components

| component | licence | how it is used |
|---|---|---|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | every trained detector here (`yolov8*-p2`, `yolo26n`) |
| [PyTorch](https://pytorch.org) / torchvision | BSD-3-Clause | training and inference |
| [OpenCV](https://opencv.org) | Apache-2.0 | stabilisation, motion detection, video I/O |
| [SAHI](https://github.com/obss/sahi) | MIT | sliced inference in the round-1 comparisons |
| [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) + [Pegasus Simulator](https://github.com/PegasusSimulator/PegasusSimulator) | NVIDIA Omniverse EULA · BSD-3-Clause | the pursuit renderer and airframe model. **Not redistributed here** — the container is yours to obtain and run. |
| [TensorRT](https://developer.nvidia.com/tensorrt) | NVIDIA SLA | the FP16 engines behind the edge profile. Engines are architecture-specific and are **not** committed. |

## Data

The two source videos in `data/videos/` and their hand labels
(`work/gt_user.json`) are my own, and are covered by this repository's licence.

The public datasets used for the generalisation work are **not redistributed
here** — only the code that downloads and converts them. Each remains under its
own terms, and you should read them before using either commercially:

| dataset | source |
|---|---|
| ARD-MAV | [WindyLab/Global-Local-MAV-Detection](https://github.com/WindyLab/Global-Local-MAV-Detection) |
| NPS-Drones | [Purdue UAV Dataset](https://engineering.purdue.edu/~bouman/UAV_Dataset/) |

## Scope of the results

Stated here as well as in the README, because it is the thing most easily
misread from a headline number:

* **Detection results are measured on real video** — hand-labelled, with the
  test video never trained on and never used for model selection.
* **Interception results are measured in simulation** — a closed-loop NVIDIA
  Isaac Sim renderer at 20 Hz, with a rendered town and rendered aircraft.
  **There is no flight test in this repository.**
