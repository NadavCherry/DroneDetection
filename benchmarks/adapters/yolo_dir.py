"""Generic adapter for anything already in YOLO layout — and the base of the stills half.

Most released detection datasets now ship as YOLO directories, in one of four layouts that
differ only in where the split name is spliced into the path. Rather than a new adapter per
release, this one discovers the layout and normalises it:

    <root>/{train,valid,test}/images + /labels      (Roboflow's export, the commonest)
    <root>/images/{train,val,test}  + labels/{...}  (what this repo's own builders write)
    <root>/images + <root>/labels                   (flat, no split)
    <root>/*.jpg  + <root>/*.txt                    (flat, side by side)

Conventions this adapter enforces, each because the alternative is a silent wrong number:

* **A "sequence" is a split, not a clip.** A stills corpus has no clips; the unit that
  matters is the group of images that must not leak into another group. So `sequences()`
  returns split names, and `split_of()` returns the sequence itself.
* **Frame index is a position in the sorted file list**, and the mapping is written into
  the ground truth's ``meta["images"]`` by the base class. Without that, an index in a GT
  json is an unreproducible integer.
* **A missing label file means background**, exactly as YOLO reads it — and it is kept, not
  skipped. Negative images are the scarcest thing in an anti-UAV training set.
* **The class table is read from the dataset, never assumed.** `read_class_names` parses
  ``data.yaml``/``classes.txt`` and raises if it cannot; guessing that index 3 is "bird"
  when it is "drone" would not crash anything, it would just make every claim false.
* **Splits that are not on disk are hashed, per image, with SHA-1** — deterministic across
  runs and machines (`hash()` is salted per process), and stamped ``self-chosen`` on the
  manifest so nobody quotes the resulting number against a published one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .base import Adapter, Box, ImageSource, image_size, parse_yolo_label, read_class_names

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

#: Directory name -> canonical split. 'valid' is Roboflow's spelling of 'val'.
SPLIT_ALIASES: dict[str, str] = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}


def _sorted_images(d: Path) -> list[Path]:
    return sorted((p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
                  key=lambda p: p.name)


class YoloDirAdapter(Adapter):
    """A YOLO-layout directory. `key` stays empty: there is no catalog entry for
    "whatever the user pointed at", so the split can only ever be on-disk or self-chosen,
    never official — and `split_source()` says so."""

    key = ""

    def __init__(self, root: str | Path, *, names: dict[int, str] | None = None) -> None:
        super().__init__(root)
        self._names = names
        self._groups: dict[str, list[tuple[Path, Path | None]]] | None = None
        self._on_disk_split = False

    # ------------------------------------------------------------------ layout discovery
    def class_names(self) -> dict[int, str]:
        if self._names is None:
            for candidate in ("data.yaml", "data.yml", "classes.txt", "obj.names"):
                p = self.root / candidate
                if p.exists():
                    self._names = read_class_names(p)
                    break
            else:
                raise FileNotFoundError(
                    f"no class table in {self.root} (looked for data.yaml, data.yml, "
                    f"classes.txt, obj.names). Supply one, or pass names= explicitly: this "
                    f"adapter will not assume an index->name order, because a wrong one "
                    f"relabels birds as drones without failing.")
        return self._names

    def _pair(self, images: list[Path], label_dir: Path | None
              ) -> list[tuple[Path, Path | None]]:
        out = []
        for img in images:
            lbl = None
            if label_dir is not None:
                cand = label_dir / f"{img.stem}.txt"
                lbl = cand if cand.exists() else None
            out.append((img, lbl))
        return out

    def groups(self) -> dict[str, list[tuple[Path, Path | None]]]:
        """``split -> [(image, label|None)]``, discovered once and cached.

        Cached because `boxes()` and `image_source()` must agree on the frame indices, and
        the cheapest way to guarantee that is for both to read the same list object rather
        than to re-sort the directory and hope the filesystem agreed with itself.
        """
        if self._groups is not None:
            return self._groups
        if not self.root.is_dir():
            raise FileNotFoundError(f"no dataset root at {self.root}")

        groups: dict[str, list[tuple[Path, Path | None]]] = {}

        # Layout 1: <root>/<split>/images + labels
        for child in sorted(p for p in self.root.iterdir() if p.is_dir()):
            split = SPLIT_ALIASES.get(child.name.lower())
            if split and (child / "images").is_dir():
                lbl = child / "labels"
                groups.setdefault(split, []).extend(
                    self._pair(_sorted_images(child / "images"), lbl if lbl.is_dir() else None))

        # Layout 2: <root>/images/<split> + <root>/labels/<split>
        if not groups and (self.root / "images").is_dir():
            for child in sorted(p for p in (self.root / "images").iterdir() if p.is_dir()):
                split = SPLIT_ALIASES.get(child.name.lower())
                if split:
                    lbl = self.root / "labels" / child.name
                    groups.setdefault(split, []).extend(
                        self._pair(_sorted_images(child), lbl if lbl.is_dir() else None))

        if groups:
            self._on_disk_split = True
            self._groups = {k: sorted(v, key=lambda t: t[0].name) for k, v in groups.items()}
            return self._groups

        # Layout 3/4: flat. Split it ourselves, per image, deterministically.
        if (self.root / "images").is_dir():
            images = _sorted_images(self.root / "images")
            lbl_dir = self.root / "labels" if (self.root / "labels").is_dir() else None
        else:
            images = _sorted_images(self.root)
            lbl_dir = self.root
        if not images:
            raise FileNotFoundError(f"no images found under {self.root}")
        flat = self._pair(images, lbl_dir)
        self._on_disk_split = False
        split_groups: dict[str, list[tuple[Path, Path | None]]] = {}
        for img, lbl in flat:
            split_groups.setdefault(self._hash_split(img.stem), []).append((img, lbl))
        self._groups = {k: v for k, v in sorted(split_groups.items())}
        return self._groups

    @staticmethod
    def _hash_split(stem: str) -> str:
        bucket = int(hashlib.sha1(stem.encode()).hexdigest()[:8], 16) % 10
        return "test" if bucket == 0 else "val" if bucket in (1, 2) else "train"

    # ------------------------------------------------------------------ required surface
    def sequences(self) -> list[str]:
        return sorted(self.groups())

    def boxes(self, seq: str) -> dict[int, list[Box]]:
        names = self.class_names()
        pairs = self.groups().get(seq)
        if pairs is None:
            raise KeyError(f"no such split {seq!r}; have {sorted(self.groups())}")
        out: dict[int, list[Box]] = {}
        for i, (img, lbl) in enumerate(pairs):
            if lbl is None:
                out[i] = []                       # YOLO's own convention: no label = background
                continue
            w, h = image_size(img)
            out[i] = parse_yolo_label(lbl.read_text(encoding="utf-8"), w, h, names)
        return out

    def image_source(self, seq: str) -> ImageSource:
        pairs = self.groups().get(seq)
        if pairs is None:
            raise KeyError(f"no such split {seq!r}; have {sorted(self.groups())}")
        return ImageSource(kind="stills",
                           images=tuple((i, img) for i, (img, _) in enumerate(pairs)))

    # ------------------------------------------------------------------ splits
    def split_of(self, seq: str) -> str:
        """The sequence *is* the split here — see the module docstring."""
        self.groups()
        return SPLIT_ALIASES.get(seq.lower(), seq)

    def split_source(self) -> str:
        d = self.dataset
        if d is not None and (d.official_test or d.official_val):
            return "official"
        self.groups()
        if self._on_disk_split:
            return ("on-disk (the release's own train/val/test directories; official only "
                    "if the release says so — the catalog records no official split)")
        return "self-chosen (SHA-1 per-image 70/20/10; publish this split with any number)"
