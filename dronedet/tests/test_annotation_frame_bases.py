"""Every annotation format's frame origin, pinned against the shipped files.

Three annotation sets are parsed in this repo and they do not agree with each other:

    ARD-MAV XML        <vid>_0001.xml ...   1-based, so frame0 = k - 1
    Purdue NPS v1      "time_layer: N"      1-based, so frame0 = N - 1
    Dogfight NPS       "0,1,x1,y1,x2,y2"    0-BASED, used as-is

Getting one wrong does not raise. It shifts every box by a frame, and on drone-to-drone
footage a frame is several pixels against a ten-pixel box, so IoU against the true position
collapses. That is exactly what happened: `parse_nps_dogfight` subtracted 1 as the other
two formats require, our NPS AP came out at 0.15 against a published 0.89-0.95, and the
only visible trace was a box filed under frame -1.

These tests read the real annotation files when they are present and skip otherwise, so
they are meaningful on the cluster and harmless on a laptop without the corpora.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools import make_dataset_external as M


def _first_existing(paths):
    return next((p for p in paths if p.exists()), None)


def test_nps_dogfight_frames_are_zero_based():
    """0..N-1 against a video of N frames. If this file were 1-based its minimum would
    be 1 and its maximum would equal the frame count."""
    d = M.NPS_ANN_DOGFIGHT
    if not d.exists():
        pytest.skip("Dogfight NPS annotations not present")
    f = _first_existing([d / "Clip_041.txt", *sorted(d.glob("Clip_*.txt"))[:1]])
    if f is None:
        pytest.skip("no Dogfight annotation files")

    idx = [int(ln.split(",")[0]) for ln in f.read_text(encoding="utf-8").splitlines()
           if ln.strip() and ln.split(",")[0].lstrip("-").isdigit()]
    assert idx, f"no parseable frame indices in {f.name}"
    assert min(idx) == 0, (
        f"{f.name} starts at frame {min(idx)}, not 0 -- if this corpus has changed to "
        f"1-based, parse_nps_dogfight must subtract 1 again")


def test_the_dogfight_parser_does_not_shift_frames():
    d = M.NPS_ANN_DOGFIGHT
    if not d.exists():
        pytest.skip("Dogfight NPS annotations not present")
    clip = sorted(p.stem for p in d.glob("Clip_*.txt"))
    if not clip:
        pytest.skip("no Dogfight annotation files")

    parsed = M.parse_nps_dogfight(clip[0])
    raw = [int(ln.split(",")[0]) for ln
           in (d / f"{clip[0]}.txt").read_text(encoding="utf-8").splitlines()
           if ln.strip() and ln.split(",")[0].lstrip("-").isdigit()]

    assert min(parsed) >= 0, f"parser produced negative frame index {min(parsed)}"
    assert min(parsed) == min(raw), "parser shifted the first frame"
    assert max(parsed) == max(raw), "parser shifted the last frame"


def test_ardmav_xml_is_one_based_and_the_parser_converts_it():
    """The opposite convention, and it must keep its -1. ARD-MAV scores ~0.77-0.80 against
    GLAD's published 0.80, which is the corroborating evidence that this one is right."""
    ann = M.ARD_ROOT / "Annotations"
    if not ann.exists():
        pytest.skip("ARD-MAV annotations not present")
    vid_dir = next((p for p in sorted(ann.iterdir()) if p.is_dir()), None)
    if vid_dir is None:
        pytest.skip("no ARD-MAV annotation directories")

    ks = sorted(int(p.stem.split("_")[-1]) for p in vid_dir.glob(f"{vid_dir.name}_*.xml"))
    if not ks:
        pytest.skip("no ARD-MAV xml files")
    assert min(ks) == 1, f"ARD-MAV xml numbering starts at {min(ks)}, not 1"

    parsed = M.parse_ardmav(vid_dir.name)
    assert min(parsed) == 0, "parse_ardmav must map the 1-based xml onto 0-based frames"
    assert max(parsed) == max(ks) - 1


def test_the_three_parsers_document_their_frame_origin():
    """A future reader must not have to infer the convention from a subtraction."""
    src = Path(M.__file__).read_text(encoding="utf-8")
    dogfight = src[src.index("def parse_nps_dogfight"):src.index("def _nps_video")]
    assert re.search(r"0-BASED|0-based", dogfight), \
        "parse_nps_dogfight must state its frame origin"
