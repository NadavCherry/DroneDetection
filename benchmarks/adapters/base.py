"""What every dataset adapter must provide, and the conventions it may not break.

An adapter is the *only* place a dataset's native format is understood. Downstream —
`tools/prepare_data.py`, the evaluator, the scorecard — sees two things and nothing else:
a `GroundTruth` per sequence and an `ImageSource` per sequence. When a new dataset lands,
exactly one new file is written and nothing else changes.

Conventions, and why each one is here
-------------------------------------
**Boxes cross this boundary as ``xyxy`` and land in ground truth as ``(cx, cy, w, h)``.**
Every source format in this repo's reach is corner-based (VOC ``xmin/ymin/xmax/ymax``,
NPS ``(top,left,bottom,right)``, YOLO's normalised centre form is the one exception), and
`dronedet.gt` stores centres. Mixing the two costs ~w/2 px of phantom localisation error,
which at a 6 px target *is* the measurement. So the conversion happens once, in
`Adapter.ground_truth`, which is concrete here and which no adapter overrides; subclasses
implement `boxes()` and hand back corners.

**`ground_truth()` never inflates a box.** Label inflation (`--min-side`) is a *training*
device: it keeps IoU-based anchor assignment from starving a 4 px target of positives. In
ground truth it is not a device, it is a falsified annotation — and it silently caps the
achievable IoU, which is how this repo ended up with a structurally impossible COCO AP
(docs/research/verified-measurements-2026-08.md §6b). Inflation lives in the YOLO builder
and appears in the manifest; it never reaches `gt/`.

**Non-target classes become ``ignore`` objects, not deletions.** `dronedet.metrics` scores
a hit on an ignore object as a third outcome — a *distractor* — counted and reported
rather than discarded. That is the whole drone-vs-bird claim: "0 hits on 3,162 labelled
bird instances" only exists as a number if the birds survive into the ground truth. A
dataset that labels birds and an adapter that drops them is a wasted dataset.

**Splits come from the catalog, never from position.** `Dataset.official_test` /
`official_val` win whenever they exist. When they do not, `_fallback_split` hashes the
sequence id with SHA-1 — *not* `hash()`, which is salted per process, and not `i % 10`,
which is what produced the ARD-MAV leak (`dronedet/tests/test_splits.py`). Either way the
answer is recorded in the manifest with its provenance, so no number can later be quoted
against a published one without the split it actually used being visible.

Stdlib only (plus `dronedet.gt`, itself stdlib), so this package imports in the torch-free
CI job. Anything needing pixels takes a lazy `cv2` import inside the function that needs it.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from dronedet.gt import GroundTruth, GTObject

from ..catalog import DATASETS, Condition, Dataset

#: Class names that mean "the thing we are trying to detect", lowercased. Everything else
#: a dataset labels is a distractor: kept in ground truth with ``ignore=True``, dropped
#: from the YOLO positives so the image trains as a hard negative.
DRONE_ALIASES: frozenset[str] = frozenset(
    {"drone", "mav", "uav", "quadcopter", "quadrotor", "multirotor"})


@dataclass(frozen=True)
class Box:
    """One annotation in original-image pixels, corner format.

    ``cls`` is the dataset's own class name, lowercased and otherwise untouched — mapping
    it to positive/distractor is a policy decision and belongs to the adapter, not to the
    parse, so the raw name stays visible in ground truth and in the manifest.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    cls: str = "drone"

    @property
    def w(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def h(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def as_cxcywh(self) -> tuple[float, float, float, float]:
        return (self.cx, self.cy, self.w, self.h)


@dataclass(frozen=True)
class ImageSource:
    """Where the pixels for one sequence live.

    Two kinds, because the two halves of the catalog are genuinely different animals:
    ``video`` must be decoded strictly sequentially (never seek — both of this repo's own
    videos hide their opening seconds behind an MP4 edit list, and inter-frame codecs make
    positioned reads unreliable near stream ends), while ``stills`` is a list of files and
    the "frame index" is a position in the sorted list, recorded in the GT ``meta`` so the
    mapping is never guessed later.
    """

    kind: str                                     # "video" | "stills"
    video: Path | None = None
    images: tuple[tuple[int, Path], ...] = ()     # (frame index, file), sorted by index

    def __post_init__(self) -> None:
        if self.kind not in ("video", "stills"):
            raise ValueError(f"ImageSource.kind must be 'video' or 'stills', got {self.kind!r}")
        if self.kind == "video" and self.video is None:
            raise ValueError("ImageSource(kind='video') needs a video path")

    @property
    def exists(self) -> bool:
        if self.kind == "video":
            return self.video is not None and self.video.exists()
        return bool(self.images) and all(p.exists() for _, p in self.images)


class Adapter(ABC):
    """Base class. Subclasses implement `sequences`, `boxes` and `image_source`.

    `ground_truth` is deliberately *not* abstract: it is the one place corners become
    centres, and an adapter that overrode it could reintroduce the format bug this class
    exists to make impossible.
    """

    #: Catalog key, so `Dataset.official_test` / `official_val` / `conditions` are reachable.
    #: Empty for adapters with no catalog entry (the generic `yolo_dir`).
    key: str = ""

    #: Class names treated as the detection target. Everything else becomes a distractor.
    positive_classes: frozenset[str] = DRONE_ALIASES

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------ required surface
    @abstractmethod
    def sequences(self) -> list[str]:
        """Sequence ids, sorted and stable. For video sets, one per clip; for stills, the
        grouping the dataset itself uses (an on-disk split directory, usually)."""

    @abstractmethod
    def boxes(self, seq: str) -> dict[int, list[Box]]:
        """``{frame index: [Box, ...]}`` in original-image pixels, TRUE extent.

        A frame that is annotated and empty must appear with an empty list — that is the
        difference between "the drone is not here" (a usable hard negative) and "nobody
        looked at this frame", and only the parse can tell them apart.
        """

    def image_source(self, seq: str) -> ImageSource:
        raise NotImplementedError(
            f"{type(self).__name__} does not expose pixels yet; ground truth only")

    def conditions(self, seq: str) -> tuple[Condition, ...]:
        """Operating conditions of *this sequence*, from the dataset's own metadata.

        Returns ``()`` when the dataset ships no per-sequence metadata. Note what is
        deliberately not done here: the catalog's dataset-level conditions are *not*
        returned as a stand-in. "ARD-MAV contains clutter" is a fact about the corpus;
        asserting it of clip `phantom43` without having looked is the sort of unchecked
        claim this whole package exists to prevent. Read `Dataset.conditions` for the
        corpus-level answer; this method answers only when the dataset can back it.
        """
        return ()

    # ------------------------------------------------------------------ derived surface
    @property
    def dataset(self) -> Dataset | None:
        return DATASETS.get(self.key)

    def video_path(self, seq: str) -> str:
        """What goes in `GroundTruth.video`. Its only job is to identify the pixels the
        boxes were measured against, so the stills case names the directory."""
        src = self.image_source(seq)
        if src.kind == "video":
            return str(src.video)
        return str(self.root)

    def ground_truth(self, seq: str, boxes: dict[int, list[Box]] | None = None) -> GroundTruth:
        """Corners -> centres, positives -> objects, everything else -> ignore objects.

        Object ids are ``<class>_<i>``, where *i* is the index of the box among that
        class within its frame. This is not track identity and must not be read as such
        — nothing in centre-distance detection scoring needs identity, it needs every GT
        box present on its frame — but the ids stay stable frame to frame whenever the
        source lists its boxes in a stable order.

        `boxes` lets a caller that has already parsed hand the result back in. Purely an
        optimisation, and a real one: ARD-MAV is 107,497 XML files, so a builder that
        parses for the size report and again for the ground truth doubles the run for
        nothing.
        """
        by_frame = self.boxes(seq) if boxes is None else boxes
        objects: dict[str, GTObject] = {}
        n_pos = n_ign = 0
        for frame in sorted(by_frame):
            per_class: dict[str, int] = {}
            for box in by_frame[frame]:
                cls = box.cls.lower()
                i = per_class.get(cls, 0)
                per_class[cls] = i + 1
                name = f"{cls}_{i}"
                ignore = cls not in self.positive_classes
                obj = objects.get(name)
                if obj is None:
                    obj = objects[name] = GTObject(name=name, ignore=ignore)
                obj.frames[frame] = box.as_cxcywh()
                if ignore:
                    n_ign += 1
                else:
                    n_pos += 1

        try:
            src = self.image_source(seq)
            video = str(src.video) if src.kind == "video" else str(self.root)
        except (NotImplementedError, FileNotFoundError):
            # Ground truth is still useful without the pixels present -- someone else's
            # detections can be scored against it -- so a missing video is not fatal here.
            src, video = None, seq
        gt = GroundTruth(video=video, objects=objects)
        gt.meta = {
            "adapter": self.key or type(self).__name__,
            "sequence": seq,
            "split": self.split_of(seq),
            "split_source": self.split_source(),
            "conditions": [c.value for c in self.conditions(seq)],
            "annotated_frames": len(by_frame),
            "target_boxes": n_pos,
            "distractor_boxes": n_ign,
            # dronedet's own GT jsons carry these; keep the shape identical so the same
            # loaders work on both.
            "shifts": {},
            "exclude_frames": [],
        }
        if src is not None and src.kind == "stills":
            # The frame index of a stills set is a position in a sorted list and means
            # nothing on its own. Record the mapping or it cannot be reproduced.
            gt.meta["images"] = {str(i): p.name for i, p in src.images}
        return gt

    # ------------------------------------------------------------------ splits
    def split_of(self, seq: str) -> str:
        """'train' | 'val' | 'test'. Official lists win; otherwise `_fallback_split`."""
        d = self.dataset
        if d is not None and (d.official_test or d.official_val):
            if seq in d.official_test:
                return "test"
            if seq in d.official_val:
                return "val"
            return "train"
        return self._fallback_split(seq)

    def split_source(self) -> str:
        """Provenance of the split, verbatim into the manifest and every report built
        from it. 'official' may be compared with a published number; 'self-chosen' may
        not, and saying so is cheaper than being caught not saying it."""
        d = self.dataset
        if d is not None and (d.official_test or d.official_val):
            return "official"
        return "self-chosen (no official split published for this dataset)"

    @staticmethod
    def _fallback_split(seq: str) -> str:
        """Deterministic 70/20/10 whole-sequence split from SHA-1 of the sequence id.

        SHA-1 rather than `hash()` because the builtin is salted per process, so a rerun
        would silently produce a different split and two runs' numbers would not be
        comparable. Whole-sequence rather than per-frame because consecutive frames of one
        track are not independent samples — a frame-level split leaks the target's
        appearance into the test set and inflates everything.
        """
        bucket = int(hashlib.sha1(seq.encode()).hexdigest()[:8], 16) % 10
        return "test" if bucket == 0 else "val" if bucket in (1, 2) else "train"

    def split_map(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        for seq in self.sequences():
            out[self.split_of(seq)].append(seq)
        return {k: sorted(v) for k, v in out.items()}


# ------------------------------------------------------------------------- shared helpers
def image_size(path: str | Path) -> tuple[int, int]:
    """``(width, height)`` from the file header, falling back to a full decode.

    Header-first because a stills corpus is ~14k files and the YOLO parse needs nothing
    but the dimensions to denormalise its labels; decoding all of them to learn two
    integers each costs minutes for no information. The fallback keeps correctness when
    the format is anything other than PNG/JPEG.
    """
    p = Path(path)
    with p.open("rb") as fh:
        head = fh.read(26)
        if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            return (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big"))
        if head[:2] == b"\xff\xd8":
            fh.seek(2)
            while True:
                b = fh.read(1)
                if not b:
                    break
                if b != b"\xff":
                    continue
                marker = fh.read(1)
                while marker == b"\xff":            # fill bytes are legal between markers
                    marker = fh.read(1)
                if not marker:
                    break
                m = marker[0]
                if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:   # standalone, no length field
                    continue
                length = int.from_bytes(fh.read(2), "big")
                # SOF0..SOF15, excluding DHT (C4), JPG (C8) and DAC (CC), carry the size.
                if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                    payload = fh.read(5)
                    h = int.from_bytes(payload[1:3], "big")
                    w = int.from_bytes(payload[3:5], "big")
                    return (w, h)
                fh.seek(length - 2, 1)

    import cv2                                       # lazy: keeps the package CI-light

    img = cv2.imread(str(p))
    if img is None:
        raise OSError(f"cannot read image dimensions: {p}")
    return (int(img.shape[1]), int(img.shape[0]))


def parse_yolo_label(text: str, width: int, height: int,
                     names: dict[int, str]) -> list[Box]:
    """One YOLO ``.txt`` -> boxes in pixels.

    YOLO stores ``class cx cy w h`` normalised to [0, 1]; a trailing confidence column
    (written by prediction dumps, not by annotation) is tolerated and ignored. Unknown
    class indices raise rather than defaulting to class 0: silently relabelling a bird as
    a drone is the exact failure this dataset was acquired to measure.
    """
    out: list[Box] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"malformed YOLO label at line {line_no}: {line!r}")
        idx = int(float(parts[0]))
        if idx not in names:
            raise ValueError(
                f"class index {idx} at line {line_no} is not in the dataset's class list "
                f"{sorted(names)}; refusing to guess what it is")
        cx, cy, w, h = (float(v) for v in parts[1:5])
        out.append(Box(x1=(cx - w / 2) * width, y1=(cy - h / 2) * height,
                       x2=(cx + w / 2) * width, y2=(cy + h / 2) * height,
                       cls=names[idx].lower()))
    return out


def read_class_names(path: str | Path) -> dict[int, str]:
    """``index -> name`` from a YOLO ``data.yaml`` or a ``classes.txt``.

    A deliberately minimal reader rather than PyYAML, which is not installed in the CI
    job and would be a new dependency for one ``names:`` block. It understands the two
    shapes anything in this ecosystem actually writes — a mapping (``0: drone``) and a
    flow/block list (``names: [drone, bird]``) — and raises on anything else instead of
    guessing, because an index->name table guessed wrong turns every class-conditional
    number downstream into fiction.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".txt":
        names = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return {i: n for i, n in enumerate(names)}

    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped.lstrip().startswith("names"):
            continue
        after = stripped.split(":", 1)[1].strip()
        if after.startswith("["):                     # names: [drone, bird, ...]
            items = [s.strip().strip("'\"") for s in after.strip("[]").split(",")]
            return {j: s.lower() for j, s in enumerate(items) if s}
        out: dict[int, str] = {}
        seq_idx = 0
        for follow in lines[i + 1:]:
            body = follow.split("#", 1)[0].rstrip()
            if not body.strip():
                continue
            if not body[:1].isspace():                # dedent ends the block
                break
            item = body.strip()
            if item.startswith("- "):                 # names:\n  - drone
                out[seq_idx] = item[2:].strip().strip("'\"").lower()
                seq_idx += 1
            elif ":" in item:                         # names:\n  0: drone
                k, v = item.split(":", 1)
                out[int(k.strip())] = v.strip().strip("'\"").lower()
            else:
                break
        if out:
            return out
    raise ValueError(
        f"no parseable 'names:' block in {p}; write one as '0: drone' lines or supply a "
        f"classes.txt — this reader will not guess the class order")
