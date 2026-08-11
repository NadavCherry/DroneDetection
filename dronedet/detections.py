"""Detection data model and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = "drone"

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def as_list(self) -> list:
        return [round(self.x1, 2), round(self.y1, 2), round(self.x2, 2),
                round(self.y2, 2), round(self.score, 4), self.label]

    @staticmethod
    def from_list(v: list) -> "Detection":
        return Detection(v[0], v[1], v[2], v[3], v[4], v[5] if len(v) > 5 else "drone")


@dataclass
class DetectionSet:
    """Per-frame detections for one method over one video."""
    video: str
    method: str
    frames: dict[int, list[Detection]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def add(self, frame: int, dets: list[Detection]) -> None:
        self.frames[frame] = dets

    def save(self, path: str | Path) -> None:
        payload = {
            "video": self.video,
            "method": self.method,
            "meta": self.meta,
            "frames": {str(k): [d.as_list() for d in v] for k, v in sorted(self.frames.items())},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload))

    @staticmethod
    def load(path: str | Path) -> "DetectionSet":
        raw = json.loads(Path(path).read_text())
        ds = DetectionSet(video=raw["video"], method=raw["method"], meta=raw.get("meta", {}))
        for k, v in raw["frames"].items():
            ds.frames[int(k)] = [Detection.from_list(x) for x in v]
        return ds


def nms(dets: list[Detection], dist_thresh: float = 8.0) -> list[Detection]:
    """Greedy suppression by center distance -- IoU is meaningless for
    few-pixel boxes, so near-duplicate centers are merged instead."""
    out: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all((d.cx - k.cx) ** 2 + (d.cy - k.cy) ** 2 > dist_thresh ** 2 for k in out):
            out.append(d)
    return out
