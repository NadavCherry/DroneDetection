"""Halmstad Drone Detection Dataset — the bird/plane/helicopter corpus, and its friction.

Why this dataset is worth the trouble: it is the only set in the catalog that is
simultaneously **video** (so a temporal method can be run on it), **bird-labelled** (so a
drone-vs-bird false-alarm rate is computable rather than asserted), **night-inclusive**,
and **CC0 with no form** (so anyone can reproduce the number). Nothing else scores on all
four.

Why it is not finished here: **the labels are MATLAB objects, and this repo has no MATLAB.**

The MATLAB friction, precisely
------------------------------
The annotations ship as ``.mat`` files written by MATLAB's Ground Truth Labeler, i.e. a
serialised MATLAB *class instance* (`groundTruth` / `labelType`), not a numeric array.
That distinction is the whole problem:

* `scipy.io.loadmat` reads MAT v5 numeric arrays. Faced with a class instance it returns
  `MatlabOpaque` records plus an unresolved ``__function_workspace__`` blob — the object
  graph is there, but the field names, the class table and the subsystem offsets that
  reconstruct it are MATLAB-internal (the "MCOS" format) and are not public.
* `h5py` opens the v7.3 variants but shows the same thing one layer down: an ``/#refs#``
  group of anonymous datasets joined by object references.

So a parser here would be a reverse-engineered MCOS decoder, and a wrong one would not
crash — it would produce *plausible boxes in the wrong frames*, which is the worst
possible failure for a dataset whose entire purpose is a false-alarm claim. Hence
`NotImplementedError` rather than a best effort.

The unblock, in the order of decreasing cost to trust
-----------------------------------------------------
1. **Export from MATLAB or GNU Octave once**, to the JSON sidecar this adapter already
   reads (format below). One conversion, checked by eye against a handful of rendered
   frames, and this file starts working. This is the recommended route; the export
   snippet has NOT been run here, so treat it as a sketch, not a recipe.
2. **A published third-party conversion**, if one exists — inherit someone else's
   verification rather than none.
3. An MCOS decoder. Only with a frame-by-frame visual check against the videos.

Sidecar format, one file per sequence, at ``<root>/labels_json/<seq>.json``::

    {"frames": {"0": [[x1, y1, x2, y2], ...], "1": [], ...}}

Corner pixels in the original frame, **0-based decoded frame index**, and an empty list
where the annotator marked the object absent (that is a usable hard negative and must not
be confused with an unlabelled frame, which is simply not a key). The class is taken from
the filename, not the sidecar, because each Halmstad clip contains exactly one object
class and the filename is the dataset's own statement of it.

Two further gaps this adapter cannot close on its own, recorded so they are not
rediscovered at publication time:

* **No official split.** `Dataset.official_test` is empty, so the base class falls back
  to a SHA-1 whole-sequence split and stamps ``self-chosen`` on the manifest. A number on
  a self-chosen split is not comparable with anyone's — publish the split file alongside
  the number or the number is not reproducible.
* **No official metric.** Nothing here says IoU 0.5, IoU 0.25 or centre distance, so the
  protocol must be chosen and stated by us.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..catalog import Condition
from .base import Adapter, Box, ImageSource

#: Filename tokens -> class. Halmstad names its clips by the object they contain, e.g.
#: ``V_DRONE_001.mp4`` / ``IR_BIRD_014.mp4``, and each clip holds exactly one class.
_CLASS_TOKENS: dict[str, str] = {
    "DRONE": "drone",
    "BIRD": "bird",
    "AIRPLANE": "airplane",
    "AEROPLANE": "airplane",
    "PLANE": "airplane",
    "HELICOPTER": "helicopter",
}

_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v")


class HalmstadAdapter(Adapter):
    key = "halmstad"

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.sidecar_root = self.root / "labels_json"

    # ------------------------------------------------------------------ sequences
    def sequences(self) -> list[str]:
        """Video stems, found recursively — the release splits IR and visible into
        sibling directories and the exact names have changed between mirrors, so the
        layout is discovered rather than assumed."""
        if not self.root.is_dir():
            raise FileNotFoundError(f"no Halmstad root at {self.root}")
        stems = {p.stem for p in self.root.rglob("*")
                 if p.suffix.lower() in _VIDEO_SUFFIXES}
        return sorted(stems)

    def _video_file(self, seq: str) -> Path | None:
        for p in self.root.rglob(f"{seq}.*"):
            if p.suffix.lower() in _VIDEO_SUFFIXES:
                return p
        return None

    def image_source(self, seq: str) -> ImageSource:
        path = self._video_file(seq)
        if path is None:
            raise FileNotFoundError(f"no video file for Halmstad sequence {seq!r} under {self.root}")
        return ImageSource(kind="video", video=path)

    # ------------------------------------------------------------------ labels
    def class_of(self, seq: str) -> str:
        """Class from the filename token. Unknown -> 'unknown', which the base class
        treats as a distractor: an unrecognised clip can never become a drone positive
        by accident, which is the safe direction to fail in for this corpus."""
        upper = seq.upper()
        for token, cls in _CLASS_TOKENS.items():
            if token in upper:
                return cls
        return "unknown"

    def boxes(self, seq: str) -> dict[int, list[Box]]:
        sidecar = self.sidecar_root / f"{seq}.json"
        if not sidecar.exists():
            raise NotImplementedError(
                "Halmstad labels are MATLAB Ground Truth Labeler objects (MCOS class "
                "instances inside .mat), which scipy.io.loadmat cannot reconstruct and "
                "which this repo will not decode by guesswork — a wrong decode yields "
                "plausible boxes on wrong frames, and this dataset exists to support a "
                "false-alarm claim that such an error would silently invalidate.\n"
                f"MISSING: {sidecar}\n"
                "Produce it once from MATLAB/Octave (sketch, untested here: load the .mat, "
                "iterate gTruth.LabelData, and write {\"frames\": {\"<0-based frame>\": "
                "[[x1,y1,x2,y2], ...]}} as JSON), spot-check a few frames against the "
                "video, then rerun. Also still missing and NOT substitutable: the .xlsx "
                "manifest (per-clip conditions, incl. which clips are night) and an "
                "official split — Halmstad publishes neither a split nor a metric, so "
                "both must be chosen by us and published with any number.")
        raw = json.loads(sidecar.read_text())
        cls = self.class_of(seq)
        out: dict[int, list[Box]] = {}
        for frame_str, boxes in raw["frames"].items():
            out[int(frame_str)] = [Box(float(b[0]), float(b[1]), float(b[2]), float(b[3]), cls)
                                   for b in boxes]
        return out

    # ------------------------------------------------------------------ conditions
    def conditions(self, seq: str) -> tuple[Condition, ...]:
        """Only what the filename actually states.

        The IR clips are a different *modality*, not a condition, and `Condition` has no
        member for it — so nothing is returned for them rather than mislabelling infrared
        as NIGHT. The clips that really are night-time visible footage are identified in
        the release's .xlsx manifest, which is not parsed yet; until it is, this dataset
        cannot support a night-stratified row, and pretending otherwise here would put a
        fabricated condition label into a published table.
        """
        return ()
