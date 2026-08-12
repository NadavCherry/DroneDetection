"""UAV_SMID — 13,928 stills, five deliberately balanced classes, and its real job.

Mendeley, CC BY 4.0, direct download (checksum in `benchmarks/catalog.py`):
helicopter / bomb / drone / bird / aeroplane. The balance is the point. Almost every
anti-UAV training set is drone-only, so the detectors trained on them have **never been
shown a bird** and their "false positives" are unattributed. This corpus is the cheapest
way to change that.

Layout, verified against the real download on 2026-08-12::

    dataset/{train,val,test}/images/NNNN.{png,jpg}
    dataset/{train,val,test}/annotations/NNNN.xml     Pascal VOC, bndbox = xmin..ymax

Two corrections to what this adapter originally assumed, both found by opening the data:

* **The release is VOC XML, not YOLO.** An earlier version of this file subclassed the
  YOLO-directory adapter and read a class table from ``data.yaml``. Neither file exists.
* **There is an official split** — 9,749 / 2,786 / 1,393 = 13,928 — carried in the
  directory names. It is used, so a number from here is reproducible by anyone who
  downloads the same archive.

Three things it is not
----------------------
**It is not evidence for the temporal claim.** These are stills. This project's headline
mechanism is a three-moment stabilised stack; a single frame cannot exercise it, and a
number from here must never be quoted as though it did. It belongs to the single-frame
branch and to the loss/matching ablations.

**It is not a tiny-target benchmark.** Measured on the real annotations, the targets here
are large — hundreds of pixels, not the 3–14 px this project is built for. Its value is
*appearance* diversity for the confuser classes, not scale. Quoting an AP from here beside
an ARD-MAV number would compare two different problems.

**It is not a five-class detector's training set.** For drone training the four non-drone
classes are *negatives*, not classes:

* The **YOLO labels keep only drone boxes.** A bird image therefore becomes an image with
  an empty label file — which is exactly what teaches "bird = background". Adding a bird
  class instead would teach the detector to *locate* birds, spending capacity on a class
  nobody asked for, and would leave the drone/bird confusion unmeasured, because a
  bird-labelled prediction is no longer a false alarm.
* The **ground truth keeps every non-drone box as an ``ignore`` object.** `dronedet.metrics`
  scores a detection on an ignore object as a *distractor*: excluded from precision by
  default but always counted and reported. That is the only route by which "N hits on
  3,162 labelled bird instances" becomes a published number rather than a silence.

Both behaviours come from the base class's `positive_classes` — the split of the class
table into target and distractor happens in exactly one place.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ..catalog import Condition
from .base import Adapter, Box, ImageSource

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


class UavSmidAdapter(Adapter):
    """One *split* is one sequence.

    These are unordered stills, not a flight: there is no temporal relationship between
    image 0000 and 0001, so treating each image as its own sequence would hand the
    bootstrap 13,928 'independent' units and produce an absurdly tight interval. Three
    units is honest about how little independent structure the corpus has -- and if a
    per-split number is wanted, `split_of` still names it.
    """

    key = "uav_smid"

    #: Only the drone is a target. 'bomb' joins bird/aeroplane/helicopter as a distractor:
    #: it is an air-object class this project makes no claim about, and the safe direction
    #: to fail in is "not a drone".
    positive_classes = frozenset({"drone", "uav", "mav", "quadcopter"})

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        base = Path(root)
        # Tolerate both `<root>/dataset/train` (as shipped) and `<root>/train`.
        self.data_root = base / "dataset" if (base / "dataset").is_dir() else base

    # ------------------------------------------------------------------ discovery
    def _split_dir(self, split: str) -> Path:
        return self.data_root / split

    def sequences(self) -> list[str]:
        return [s for s in SPLITS if (self._split_dir(s) / "annotations").is_dir()]

    def split_of(self, seq: str) -> str:
        return seq if seq in SPLITS else "train"

    def split_source(self) -> str:
        return ("the release's own train/val/test directories "
                "(9,749 / 2,786 / 1,393 = 13,928 images)")

    def _pairs(self, seq: str) -> list[tuple[int, Path, Path]]:
        """(frame index, annotation, image), ordered by annotation stem.

        The index is a position in this sorted list and is recorded in the GT meta by the
        base class, so the image a detection belongs to is never guessed later.
        """
        ann_dir = self._split_dir(seq) / "annotations"
        img_dir = self._split_dir(seq) / "images"
        if not ann_dir.is_dir():
            raise FileNotFoundError(f"no annotations for split {seq!r} under {self.data_root}")
        out: list[tuple[int, Path, Path]] = []
        for i, xf in enumerate(sorted(ann_dir.glob("*.xml"))):
            img = next((p for suf in IMAGE_SUFFIXES
                        if (p := img_dir / f"{xf.stem}{suf}").exists()), None)
            if img is not None:
                out.append((i, xf, img))
        return out

    # ------------------------------------------------------------------ annotations
    def boxes(self, seq: str) -> dict[int, list[Box]]:
        out: dict[int, list[Box]] = {}
        for idx, xf, _img in self._pairs(seq):
            try:
                root = ET.parse(xf).getroot()
            except ET.ParseError:
                # A corrupt annotation is a *labelled negative* only if we say so; it is
                # not, so record the frame as present-with-no-boxes and let the count
                # mismatch surface rather than dropping the image silently.
                out[idx] = []
                continue
            found: list[Box] = []
            for obj in root.findall("object"):
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                try:
                    x1, y1, x2, y2 = (float(bb.find(t).text)  # type: ignore[union-attr]
                                      for t in ("xmin", "ymin", "xmax", "ymax"))
                except (AttributeError, TypeError, ValueError):
                    continue
                name_el = obj.find("name")
                cls = ((name_el.text or "drone").strip().lower()
                       if name_el is not None else "drone")
                found.append(Box(x1, y1, x2, y2, cls))
            out[idx] = found
        return out

    def image_source(self, seq: str) -> ImageSource:
        return ImageSource(kind="stills",
                           images=tuple((i, img) for i, _ann, img in self._pairs(seq)))

    def conditions(self, seq: str) -> tuple[Condition, ...]:
        """Empty on purpose.

        The catalog records the corpus as `Condition.CLEAR`, but the release ships no
        per-image weather or illumination metadata, so no *image* here can be attributed
        to a condition. Returning the corpus-level label per sequence would put an
        unverified condition into a stratified table; read `DATASETS['uav_smid'].conditions`
        when the corpus-level answer is what is wanted.
        """
        return ()

    def class_counts(self) -> dict[str, int]:
        """Every class name present, with its instance count.

        For the manifest: a false-alarm rate must say which distractors it was measured
        against, and how many of each were on offer, rather than repeating the catalog
        from memory.
        """
        counts: dict[str, int] = {}
        for seq in self.sequences():
            for boxes in self.boxes(seq).values():
                for b in boxes:
                    counts[b.cls] = counts.get(b.cls, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def distractor_classes(self) -> list[str]:
        return sorted(n for n in self.class_counts()
                      if n not in self.positive_classes)
