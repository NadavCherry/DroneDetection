"""Pin the competitor's detection JSON to the schema our evaluator actually reads.

`tools/sota/infer_yolomg.py` runs in the competitor's environment (YOLOv5-era, numpy < 2)
and cannot import `dronedet`, so it writes the detection schema as a literal. That literal
is a copy, and copies drift. If `Detection.as_list` ever grows a field or reorders one,
every YOLOMG scorecard silently becomes garbage -- or, worse, plausible.

So this file asserts the two agree, without importing the competitor's stack: it reads the
literal out of the source and checks a round trip through the real `DetectionSet`.
"""

from __future__ import annotations

import json
from pathlib import Path

from dronedet.detections import Detection, DetectionSet

SRC = Path(__file__).resolve().parents[2] / "tools" / "sota" / "infer_yolomg.py"


def test_the_hand_written_detection_row_matches_Detection_as_list():
    """Six fields, in this order: x1 y1 x2 y2 score label."""
    row = Detection(1.234, 2.345, 3.456, 4.567, 0.891011, "drone").as_list()

    assert len(row) == 6
    assert row[:4] == [1.23, 2.35, 3.46, 4.57], "x1 y1 x2 y2, rounded to 2dp"
    assert row[4] == 0.891, "score, rounded to 4dp"
    assert row[5] == "drone"

    src = SRC.read_text(encoding="utf-8")
    # The literal in the competitor script must round to the same precision.
    assert 'round(float(xyxy[0]), 2)' in src
    assert 'round(float(sc), 4)' in src
    assert '"drone"' in src


def test_a_json_written_in_that_shape_loads_as_a_DetectionSet(tmp_path):
    """The actual contract: whatever infer_yolomg writes, DetectionSet.load must read."""
    p = tmp_path / "Clip_041.json"
    p.write_text(json.dumps({
        "video": "Clip_041.mov", "method": "yolomg",
        "meta": {"n_frames": 3, "mask_dt": 2, "scored_every_frame": True},
        "frames": {"0": [], "1": [[10.0, 20.0, 14.0, 24.0, 0.9312, "drone"]], "2": []},
    }), encoding="utf-8")

    ds = DetectionSet.load(p)

    assert ds.video == "Clip_041.mov" and ds.method == "yolomg"
    assert set(ds.frames) == {0, 1, 2}
    d = ds.frames[1][0]
    assert (d.x1, d.y1, d.x2, d.y2) == (10.0, 20.0, 14.0, 24.0)
    assert d.score == 0.9312 and d.label == "drone"
    assert ds.frames[0] == [] and ds.frames[2] == []


def test_boundary_frames_are_emitted_empty_rather_than_omitted():
    """A two-sided mask cannot exist at a video boundary. Emitting those frames as empty
    keeps their ground truth in the denominator; omitting them would delete it, which
    would flatter the competitor rather than cost it."""
    src = SRC.read_text(encoding="utf-8")
    assert "frames.setdefault(str(f), [])" in src, \
        "boundary frames must be filled in as empty detections"


def test_the_conf_floor_is_not_raised_above_the_competitors_own():
    """YOLOMG's val.py uses --conf-thres 0.001. Scoring it at a higher floor truncates its
    precision-recall curve and lowers its AP for a reason that has nothing to do with the
    method -- the same mismatch that flattered our own control once already."""
    src = SRC.read_text(encoding="utf-8")
    assert '"--conf", type=float, default=0.001' in src
