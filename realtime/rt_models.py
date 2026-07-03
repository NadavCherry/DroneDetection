"""Uniform detector wrappers over .pt / .engine (TensorRT) / .onnx weights."""

from __future__ import annotations

import numpy as np

from dronedet.detections import Detection


class Detector:
    """Returns, per input image, a list of (cls_name, conf, cx, cy).

    TensorRT engines are built with a fixed batch; ``fixed_batch`` pads
    every call to that size so latency is constant and engines stay static.
    """

    def __init__(self, weights: str, imgsz: int, conf: float = 0.05,
                 device: int | str = 0, fixed_batch: int | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights, task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.fixed_batch = fixed_batch

    def __call__(self, imgs: list[np.ndarray]):
        n = len(imgs)
        if n == 0:
            return []
        results = []
        step = self.fixed_batch or n
        for i in range(0, n, step):  # static-batch engines: chunk + pad
            chunk = list(imgs[i:i + step])
            if self.fixed_batch and len(chunk) < self.fixed_batch:
                chunk += [np.zeros_like(imgs[0])] * (self.fixed_batch - len(chunk))
            rs = self.model(chunk, imgsz=self.imgsz, conf=self.conf,
                            device=self.device, verbose=False)
            results.extend(rs[:min(step, n - i)])
        out = []
        for r in results[:n]:
            dets = []
            for b in r.boxes:
                cx = float(b.xyxy[0][0] + b.xyxy[0][2]) / 2
                cy = float(b.xyxy[0][1] + b.xyxy[0][3]) / 2
                dets.append((r.names[int(b.cls[0])], float(b.conf[0]), cx, cy))
            out.append(dets)
        return out


class Expert:
    """Low-rate near/big-object expert (full detections with boxes)."""

    def __init__(self, weights: str, imgsz: int = 1280, conf: float = 0.25,
                 device: int | str = 0):
        from ultralytics import YOLO

        self.model = YOLO(weights, task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def __call__(self, frame_bgr) -> list[Detection]:
        r = self.model(frame_bgr, imgsz=self.imgsz, conf=self.conf,
                       device=self.device, verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append(Detection(x1, y1, x2, y2, float(b.conf[0]),
                                 r.names[int(b.cls[0])]))
        return out
