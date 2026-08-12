"""ARD-MAV (Guo et al., 60 clips, 107,497 annotated frames, 1920x1080).

Layout::

    ARD-MAV/
      videos/<seq>.mp4
      Annotations/<seq>/<seq>_XXXX.xml        VOC, bndbox = xmin,ymin,xmax,ymax

Two facts about this dataset that cost time if they are rediscovered rather than read:

**The XML filename index is 1-based.** ``phantom05_0001.xml`` annotates decoded frame 0.
Off by one, every box lands one frame late; at 11.8 px median target and typical inter-frame
motion that is enough to break a centre-distance match without breaking anything visibly.
`tools/make_dataset_external.py` has had this right since round 5 and this file inherits it.

**An XML with no ``<object>`` is a labelled negative, not a gap.** ARD-MAV annotates frames
where the MAV has left the field of view, and those frames are the most valuable hard
negatives in the corpus — a clip's own background, at its own exposure, with its own
compression artefacts. They are kept as empty lists, never dropped.

The split is `benchmarks.catalog.ARD_TEST` (Guo et al.'s published 15) via the base class,
which is the whole reason this adapter exists rather than another positional re-split: the
previous builder defined the official list and then ignored it, so rounds 5–7 trained on
most of the official test set (`dronedet/tests/test_splits.py`).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .base import Adapter, Box, ImageSource


class ArdMavAdapter(Adapter):
    key = "ardmav"

    #: The corpus labels a single class and names it inconsistently across releases
    #: ('mav' in the paper, 'drone'/'UAV' in some XMLs), so all of them are the target.
    positive_classes = frozenset({"mav", "drone", "uav", "quadcopter"})

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.ann_root = self.root / "Annotations"
        self.video_root = self.root / "videos"

    def sequences(self) -> list[str]:
        """Annotation directories, not videos: a clip whose mp4 is missing can still be
        scored against someone else's detections, and a video with no labels cannot be
        used for anything."""
        if not self.ann_root.is_dir():
            raise FileNotFoundError(
                f"no Annotations/ under {self.root} — expected the ARD-MAV release layout "
                f"(videos/ + Annotations/<seq>/<seq>_XXXX.xml)")
        return sorted(p.name for p in self.ann_root.iterdir() if p.is_dir())

    def boxes(self, seq: str) -> dict[int, list[Box]]:
        ann_dir = self.ann_root / seq
        if not ann_dir.is_dir():
            raise FileNotFoundError(f"no annotations for sequence {seq!r} under {self.ann_root}")
        out: dict[int, list[Box]] = {}
        for xf in sorted(ann_dir.glob(f"{seq}_*.xml")):
            try:
                index_1based = int(xf.stem.split("_")[-1])
            except ValueError:
                continue
            try:
                root_el = ET.parse(xf).getroot()
            except ET.ParseError:
                # One truncated XML in a 107k-file corpus must not take the run down; the
                # count of parsed frames lands in the manifest, so a silent loss is visible.
                continue
            frame = index_1based - 1
            boxes: list[Box] = []
            for obj in root_el.findall("object"):
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                name_el = obj.find("name")
                cls = (name_el.text or "mav").strip().lower() if name_el is not None else "mav"
                x1, y1, x2, y2 = (float(bb.find(t).text)
                                  for t in ("xmin", "ymin", "xmax", "ymax"))
                boxes.append(Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), cls))
            out[frame] = boxes
        return out

    def image_source(self, seq: str) -> ImageSource:
        return ImageSource(kind="video", video=self.video_root / f"{seq}.mp4")
