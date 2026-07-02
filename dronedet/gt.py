"""Ground-truth store: per-object, per-frame boxes with ignore flags."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GTObject:
    name: str
    ignore: bool = False  # ignore-region objects (e.g. birds): matching
    #                       detections are neither TP nor FP
    frames: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    # frame -> (cx, cy, w, h) in original image coordinates

    def box(self, frame: int):
        return self.frames.get(frame)


@dataclass
class GroundTruth:
    video: str
    objects: dict[str, GTObject] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        payload = {
            "video": self.video,
            "meta": self.meta,
            "objects": {
                name: {
                    "ignore": o.ignore,
                    "frames": {str(f): [round(v, 2) for v in box]
                               for f, box in sorted(o.frames.items())},
                }
                for name, o in self.objects.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload))

    @staticmethod
    def load(path: str | Path) -> "GroundTruth":
        raw = json.loads(Path(path).read_text())
        gt = GroundTruth(video=raw["video"], meta=raw.get("meta", {}))
        for name, o in raw["objects"].items():
            obj = GTObject(name=name, ignore=o.get("ignore", False))
            obj.frames = {int(f): tuple(v) for f, v in o["frames"].items()}
            gt.objects[name] = obj
        return gt
