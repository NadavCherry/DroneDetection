"""Tests for the dataset adapters and `tools/prepare_data.py`.

Three things are tested here and everything else is incidental, because these three are
the ones that fail *silently* — they produce a full directory of plausible files and a
number that means something other than what its table says:

1. **Centre-format conversion.** `dronedet.gt` stores ``(cx, cy, w, h)``; every source
   format here is corner-based. Get it wrong and the boxes are off by w/2 — about one
   target width at this scale — which shifts every centre-distance match without ever
   raising.
2. **Official-split honouring.** The precedent is a shipped bug: `combined_splits()`
   defined the published ARD-MAV test list and then re-split by position, so rounds 5–7
   trained on most of the official test set (see `test_splits.py`). A split is a claim
   about what a number means, so it gets a test.
3. **`--min-side`.** Default 0 = true extent. Inflation must reach the YOLO labels and
   *never* the ground truth, and it must be reported, because its cost (a cap on
   achievable IoU) only becomes visible months later at evaluation time.

Everything runs on synthetic fixtures in `tmp_path`. No real dataset, no GPU, no torch —
CI installs numpy/scipy/opencv-headless/pytest and nothing else, and this file must pass
there. Stills rather than video wherever pixels are needed: writing an mp4 would drag a
codec into the test matrix for no extra coverage.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

cv2 = pytest.importorskip("cv2", reason="the image fixtures need OpenCV")

from benchmarks.adapters import Box, build  # noqa: E402
from benchmarks.adapters.base import image_size, parse_yolo_label, read_class_names  # noqa: E402
from benchmarks.catalog import DATASETS  # noqa: E402
from dronedet.gt import GroundTruth  # noqa: E402

prepare_data = pytest.importorskip(
    "prepare_data", reason="tools/ not importable (needs numpy + opencv)")


# ============================================================================== fixtures
def _write_voc(path: Path, boxes: list[tuple[int, int, int, int]], name: str = "mav") -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = path.stem + ".jpg"
    for (x1, y1, x2, y2) in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = name
        bb = ET.SubElement(obj, "bndbox")
        for tag, v in zip(("xmin", "ymin", "xmax", "ymax"), (x1, y1, x2, y2)):
            ET.SubElement(bb, tag).text = str(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ET.tostring(root))


@pytest.fixture
def ardmav_root(tmp_path: Path) -> Path:
    """A three-clip ARD-MAV, chosen so the official split has one of each kind:
    phantom05 is an official test video, phantom06 an official val one, phantom01 neither.
    """
    root = tmp_path / "ARD-MAV"
    (root / "videos").mkdir(parents=True)
    for seq in ("phantom01", "phantom05", "phantom06"):
        (root / "videos" / f"{seq}.mp4").write_bytes(b"")     # presence only; never decoded
        ann = root / "Annotations" / seq
        # 1-based filename index: _0001 annotates decoded frame 0.
        _write_voc(ann / f"{seq}_0001.xml", [(100, 200, 110, 206)])
        _write_voc(ann / f"{seq}_0002.xml", [])               # labelled negative, not a gap
        _write_voc(ann / f"{seq}_0003.xml", [(300, 300, 306, 304), (10, 10, 20, 20)])
    return root


def _img(path: Path, w: int = 100, h: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cv2.imwrite(str(path), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


@pytest.fixture
def yolo_root(tmp_path: Path) -> Path:
    """Roboflow-shaped stills with an on-disk split and two classes, one of them a
    distractor. 100x80 px so the default 640 tile never triggers."""
    root = tmp_path / "smid"
    (root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: valid/images\nnames:\n  0: drone\n  1: bird\n")
    spec = {
        "train": [("a", "0 0.5 0.5 0.1 0.05"), ("b", "1 0.25 0.25 0.2 0.2")],
        "valid": [("c", "0 0.4 0.6 0.1 0.05")],
        "test": [("d", "0 0.5 0.5 0.2 0.1")],
    }
    for split, items in spec.items():
        for stem, label in items:
            _img(root / split / "images" / f"{stem}.jpg")
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
            (root / split / "labels" / f"{stem}.txt").write_text(label + "\n")
    return root


@pytest.fixture
def smid_root(tmp_path: Path) -> Path:
    """The real UAV_SMID shape, verified against the 2026-08-12 download: Pascal VOC XML
    under `dataset/<split>/annotations`, images beside it, split carried in the directory
    name. An earlier fixture here used a Roboflow YOLO layout, which the release does not
    ship -- the adapter was written against a guess and the guess was wrong."""
    root = tmp_path / "smid"
    spec = {
        "train": [("0000", [("drone", 40, 40, 60, 50)]), ("0001", [("bird", 10, 10, 30, 30)])],
        "val": [("0002", [("drone", 20, 30, 40, 45)])],
        "test": [("0003", [("helicopter", 5, 5, 55, 45)])],
    }
    for split, items in spec.items():
        ann = root / "dataset" / split / "annotations"
        img = root / "dataset" / split / "images"
        ann.mkdir(parents=True, exist_ok=True)
        for stem, objects in items:
            _img(img / f"{stem}.png")
            body = "".join(
                f"<object><name>{c}</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
                f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
                for c, x1, y1, x2, y2 in objects)
            (ann / f"{stem}.xml").write_text(
                f"<annotation><filename>{stem}.png</filename>"
                f"<size><width>100</width><height>80</height><depth>3</depth></size>"
                f"{body}</annotation>")
    return root


# ====================================================================== centre conversion
def test_ground_truth_is_centre_format_not_corners(ardmav_root):
    """The single easiest bug in this package: xyxy straight into a cx,cy,w,h store."""
    gt = build("ardmav", ardmav_root).ground_truth("phantom01")
    cx, cy, w, h = gt.objects["mav_0"].frames[0]
    assert (cx, cy, w, h) == pytest.approx((105.0, 203.0, 10.0, 6.0))
    # The corner values must NOT appear as if they were a centre box.
    assert (cx, cy) != (100.0, 200.0)


def test_voc_filename_index_is_one_based(ardmav_root):
    """`phantom01_0001.xml` annotates decoded frame 0. One frame of slip is invisible on
    screen and fatal to a centre-distance match at 6 px."""
    boxes = build("ardmav", ardmav_root).boxes("phantom01")
    assert sorted(boxes) == [0, 1, 2]
    assert len(boxes[0]) == 1 and boxes[0][0].x1 == 100
    assert boxes[1] == []                    # annotated and empty
    assert len(boxes[2]) == 2


def test_simultaneous_boxes_get_distinct_object_ids(ardmav_root):
    gt = build("ardmav", ardmav_root).ground_truth("phantom01")
    assert set(gt.objects) == {"mav_0", "mav_1"}
    assert gt.objects["mav_1"].frames[2] == pytest.approx((15.0, 15.0, 10.0, 10.0))


def test_gt_json_round_trips_through_the_dronedet_schema(ardmav_root, tmp_path):
    gt = build("ardmav", ardmav_root).ground_truth("phantom01")
    gt.save(tmp_path / "g.json")
    back = GroundTruth.load(tmp_path / "g.json")
    assert back.objects["mav_0"].frames[0] == pytest.approx((105.0, 203.0, 10.0, 6.0))


def test_yolo_normalised_labels_denormalise_to_pixels(yolo_root):
    """YOLO's centre form is normalised; the store's is in pixels. 0.1 of a 100 px image
    is a 10 px box, not a 0.1 px one."""
    gt = build("yolo_dir", yolo_root).ground_truth("train")
    cx, cy, w, h = gt.objects["drone_0"].frames[0]
    assert (cx, cy, w, h) == pytest.approx((50.0, 40.0, 10.0, 4.0))


def test_parse_yolo_label_refuses_an_unknown_class_index():
    """Defaulting an unseen index to class 0 would relabel a bird as a drone and invert
    the one claim these bird datasets were acquired to support."""
    with pytest.raises(ValueError, match="not in the dataset's class list"):
        parse_yolo_label("7 0.5 0.5 0.1 0.1", 100, 100, {0: "drone"})


def test_image_size_header_parse_agrees_with_a_full_decode(tmp_path):
    for name in ("x.jpg", "x.png"):
        p = tmp_path / name
        _img(p, w=37, h=23)
        assert image_size(p) == (37, 23)
        assert image_size(p) == (cv2.imread(str(p)).shape[1], cv2.imread(str(p)).shape[0])


# ============================================================================== distractors
def test_non_target_classes_survive_as_ignore_objects(smid_root):
    """`dronedet.metrics` scores a hit on an ignore object as a counted *distractor*.
    Dropping birds at parse time is what makes a bird false-alarm rate unpublishable."""
    gt = build("uav_smid", smid_root).ground_truth("train")
    assert gt.objects["drone_0"].ignore is False
    assert gt.objects["bird_0"].ignore is True
    assert gt.meta["target_boxes"] == 1 and gt.meta["distractor_boxes"] == 1


def test_uav_smid_keeps_bird_images_as_negatives_not_as_a_bird_class(smid_root, tmp_path):
    """The bird image must be exported with an EMPTY label: that is what teaches
    'bird = background'. A bird class would teach the detector to locate birds instead."""
    out = tmp_path / "prep"
    prepare_data.main(["uav_smid", "--root", str(smid_root), "--out", str(out)])
    labels = sorted((out / "yolo/labels/train").glob("*.txt"))
    bodies = {p.name: p.read_text().strip() for p in labels}
    assert len(bodies) == 2
    assert sum(1 for v in bodies.values() if v == "") == 1        # the bird image
    kept = [v for v in bodies.values() if v]
    assert len(kept) == 1 and kept[0].startswith("0 ")           # the drone, as class 0
    assert "1:" not in (out / "yolo/data.yaml").read_text()      # one class, always


def test_class_table_is_read_not_assumed(tmp_path):
    root = tmp_path / "d"
    _img(root / "images" / "a.jpg")
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    with pytest.raises(FileNotFoundError, match="no class table"):
        build("yolo_dir", root).ground_truth("train")


@pytest.mark.parametrize("body,expect", [
    ("names:\n  0: drone\n  1: bird\n", {0: "drone", 1: "bird"}),
    ("names: [drone, bird]\n", {0: "drone", 1: "bird"}),
    ("names:\n  - drone\n  - bird\n", {0: "drone", 1: "bird"}),
])
def test_read_class_names_handles_the_shapes_people_actually_write(tmp_path, body, expect):
    p = tmp_path / "data.yaml"
    p.write_text("path: .\n" + body + "nc: 2\n")
    assert read_class_names(p) == expect


# ============================================================================== splits
def test_official_ard_split_is_honoured(ardmav_root):
    ad = build("ardmav", ardmav_root)
    assert ad.split_source() == "official"
    assert ad.split_of("phantom05") == "test"      # in catalog ARD_TEST
    assert ad.split_of("phantom06") == "val"       # in catalog ARD_VAL
    assert ad.split_of("phantom01") == "train"
    assert ad.split_map() == {"train": ["phantom01"], "val": ["phantom06"],
                              "test": ["phantom05"]}


def test_every_official_test_video_would_be_held_out(ardmav_root):
    """Not just the three in the fixture: the whole published list, so a future edit to
    the catalog cannot quietly move one into training."""
    ad = build("ardmav", ardmav_root)
    ard = DATASETS["ardmav"]
    assert len(ard.official_test) == 15
    for seq in ard.official_test:
        assert ad.split_of(seq) == "test", f"LEAK: official test video {seq} is not test"
    for seq in ard.official_val:
        assert ad.split_of(seq) == "val"


def test_on_disk_split_is_used_and_labelled_as_such(yolo_root):
    ad = build("yolo_dir", yolo_root)
    assert ad.sequences() == ["test", "train", "val"]     # 'valid' normalised to 'val'
    assert ad.split_of("val") == "val"
    assert "on-disk" in ad.split_source()


def test_fallback_split_is_deterministic_and_declared(tmp_path):
    """A flat layout has no split, so one is invented — but it must be the same one on
    every machine and every rerun (`hash()` is salted per process) and it must be stamped
    'self-chosen', because a number on it is not comparable with a published one."""
    root = tmp_path / "flat"
    (root / "classes.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("drone\n")
    for i in range(40):
        _img(root / "images" / f"img{i:03d}.jpg")
        (root / "labels").mkdir(parents=True, exist_ok=True)
        (root / "labels" / f"img{i:03d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    first = build("yolo_dir", root).split_map()
    second = build("yolo_dir", root).split_map()
    assert first == second
    assert set(first) == {"train", "val", "test"} and all(first.values())
    assert "self-chosen" in build("yolo_dir", root).split_source()


def test_test_images_are_never_written_into_the_training_set(yolo_root, tmp_path):
    """The leak that actually shipped was not malice, it was a builder that could reach
    the test sequences at all."""
    out = tmp_path / "prep"
    prepare_data.main(["yolo_dir", "--root", str(yolo_root), "--out", str(out)])
    assert not (out / "yolo/images/test").exists()
    train_stems = {p.stem for p in (out / "yolo/images/train").glob("*.jpg")}
    assert not any("__d" in s or s.endswith("test") for s in train_stems)
    data_yaml = (out / "yolo/data.yaml").read_text()
    assert "test:" not in data_yaml
    # Ground truth, by contrast, is written for every sequence including test -- that is
    # what the test split is FOR.
    assert (out / "gt" / "test.json").exists()


def test_export_test_writes_to_a_directory_data_yaml_does_not_reference(yolo_root, tmp_path):
    out = tmp_path / "prep"
    prepare_data.main(["yolo_dir", "--root", str(yolo_root), "--out", str(out), "--export-test"])
    assert list((out / "yolo/images/test").glob("*.jpg"))
    assert "test:" not in (out / "yolo/data.yaml").read_text()


# ============================================================================== min-side
def test_min_side_defaults_to_true_extent():
    box = Box(45.0, 38.0, 55.0, 42.0)                       # 10 x 4, true extent
    lines = prepare_data._label_lines([box], 0.0, 100, 80)
    _, cx, cy, w, h = lines[0].split()
    assert (float(w) * 100, float(h) * 80) == pytest.approx((10.0, 4.0))
    assert (float(cx) * 100, float(cy) * 80) == pytest.approx((50.0, 40.0))


def test_min_side_inflates_only_the_short_side_and_keeps_the_centre():
    box = Box(45.0, 38.0, 55.0, 42.0)                       # 10 x 4
    lines = prepare_data._label_lines([box], 24.0, 100, 80)
    _, cx, cy, w, h = lines[0].split()
    assert (float(w) * 100, float(h) * 80) == pytest.approx((24.0, 24.0))
    assert (float(cx) * 100, float(cy) * 80) == pytest.approx((50.0, 40.0))


def test_inflation_never_reaches_the_ground_truth(yolo_root, tmp_path):
    """`--min-side` is a training device. In ground truth it is a falsified annotation,
    and it is what made this repo's COCO AP arithmetically impossible (§6b)."""
    a, b = tmp_path / "true", tmp_path / "inflated"
    prepare_data.main(["yolo_dir", "--root", str(yolo_root), "--out", str(a)])
    prepare_data.main(["yolo_dir", "--root", str(yolo_root), "--out", str(b),
                       "--min-side", "24"])
    assert (a / "gt/train.json").read_text() == (b / "gt/train.json").read_text()
    assert (a / "yolo/labels/train").exists()
    true_label = [p for p in (a / "yolo/labels/train").glob("*.txt") if p.read_text().strip()][0]
    infl_label = (b / "yolo/labels/train" / true_label.name)
    assert true_label.read_text() != infl_label.read_text()


def test_manifest_records_the_inflation_and_its_cost(yolo_root, tmp_path, capsys):
    out = tmp_path / "prep"
    prepare_data.main(["yolo_dir", "--root", str(yolo_root), "--out", str(out),
                       "--min-side", "24"])
    printed = capsys.readouterr().out
    assert "Label inflation" in printed and "can reach IoU 0.5" in printed

    man = json.loads((out / "manifest.json").read_text())
    assert man["min_side"] == 24.0
    assert man["label_extent"] == "inflated to a minimum side of 24 px"
    assert man["gt_extent"].startswith("true extent always")
    assert man["split_source"] and man["splits"]["test"] == ["test"]

    table = {row["min_side"]: row for row in man["size_distribution"]["achievable_iou"]}
    assert 24.0 in table and 0.0 in table          # the chosen row and the honest baseline
    assert table[0.0]["median_best_iou"] == 1.0
    assert table[24.0]["median_best_iou"] < 1.0
    assert any("IoU" in w for w in man["warnings"])


def test_no_inflation_warning_when_the_default_is_used(yolo_root, tmp_path):
    out = tmp_path / "prep"
    prepare_data.main(["yolo_dir", "--root", str(yolo_root), "--out", str(out)])
    man = json.loads((out / "manifest.json").read_text())
    assert man["min_side"] == 0.0
    assert not any("min-side" in w for w in man["warnings"])
    assert man["size_distribution"]["achievable_iou"][0]["min_side"] == 0.0


# ============================================================================== tiling
def test_large_images_are_tiled_at_native_scale_not_downscaled(tmp_path):
    """A whole-frame export resizes a 1920-wide frame to 640, which turns an 11.8 px
    ARD-MAV target into 3.9 px — below the size at which any of this works. So images
    larger than the tile are cropped at native scale instead, and the target's pixel size
    in the training crop must equal its pixel size in the source."""
    root = tmp_path / "big"
    (root / "classes.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("drone\n")
    _img(root / "images" / "big000.jpg", w=1600, h=1200)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    # a 16 x 12 px target at (800, 600) in a 1600 x 1200 image
    (root / "labels" / "big000.txt").write_text("0 0.5 0.5 0.01 0.01\n")

    out = tmp_path / "prep"
    prepare_data.main(["yolo_dir", "--root", str(root), "--out", str(out),
                       "--tile", "640", "--export-test", "--neg-per-image", "0"])
    crops = sorted((out / "yolo/images").rglob("*.jpg"))
    assert crops, "nothing exported"
    assert cv2.imread(str(crops[0])).shape[:2] == (640, 640)
    label = crops[0].parent.parent.parent / "labels" / crops[0].parent.name / (crops[0].stem + ".txt")
    _, _, _, w, h = label.read_text().split()
    assert (float(w) * 640, float(h) * 640) == pytest.approx((16.0, 12.0))


def test_a_source_shorter_than_the_tile_is_padded_and_still_labelled_correctly(tmp_path):
    """A crop that must be padded up to the tile is the one case where the window and the
    written image can disagree, and the disagreement is silent: the label is normalised by
    one and read against the other. On a 1600x400 source the target at y=200 was labelled
    at y=320 with its height 60 % too large, and every fixture here happened to be larger
    than the tile in BOTH dimensions, so nothing caught it. Assert against the pixels.
    """
    root = tmp_path / "wide"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "classes.txt").write_text("drone\n")
    W, H = 1600, 400                       # wider than the 640 tile, SHORTER than it
    img = np.zeros((H, W, 3), np.uint8)
    img[194:206, 792:808] = 255            # a 16 x 12 target centred at (800, 200)
    cv2.imwrite(str(root / "images" / "wide000.jpg"), img)
    (root / "labels" / "wide000.txt").write_text(
        f"0 {800 / W:.6f} {200 / H:.6f} {16 / W:.6f} {12 / H:.6f}\n")

    out = tmp_path / "prep"
    prepare_data.main(["yolo_dir", "--root", str(root), "--out", str(out), "--tile", "640",
                       "--export-test", "--neg-per-image", "0", "--jitter", "0"])
    crops = sorted((out / "yolo/images").rglob("*.jpg"))
    assert crops, "nothing exported"
    written = cv2.imread(str(crops[0]))
    assert written.shape[:2] == (640, 640)          # padded, not stretched

    label = (crops[0].parent.parent.parent / "labels" / crops[0].parent.name
             / (crops[0].stem + ".txt"))
    _, cx, cy, w, h = label.read_text().split()
    ih, iw = written.shape[:2]
    ys, xs = np.nonzero(written.max(axis=2) > 128)
    # The label must describe the target where it actually landed in the file on disk.
    assert (float(cx) * iw, float(cy) * ih) == pytest.approx(
        (xs.mean() + 0.5, ys.mean() + 0.5), abs=1.0)
    assert (float(w) * iw, float(h) * ih) == pytest.approx((16.0, 12.0))


def test_tile_zero_writes_whole_frames(tmp_path):
    root = tmp_path / "big"
    (root / "classes.txt").parent.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("drone\n")
    _img(root / "images" / "big000.jpg", w=1600, h=1200)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "big000.txt").write_text("0 0.5 0.5 0.01 0.01\n")

    out = tmp_path / "prep"
    prepare_data.main(["yolo_dir", "--root", str(root), "--out", str(out),
                       "--tile", "0", "--export-test"])
    crops = sorted((out / "yolo/images").rglob("*.jpg"))
    assert cv2.imread(str(crops[0])).shape[:2] == (1200, 1600)


def test_dry_run_writes_nothing(ardmav_root, tmp_path):
    out = tmp_path / "prep"
    prepare_data.main(["ardmav", "--root", str(ardmav_root), "--out", str(out), "--dry-run"])
    assert not out.exists()


def test_gt_only_writes_ground_truth_for_every_sequence_and_no_images(ardmav_root, tmp_path):
    out = tmp_path / "prep"
    prepare_data.main(["ardmav", "--root", str(ardmav_root), "--out", str(out), "--gt-only"])
    assert {p.stem for p in (out / "gt").glob("*.json")} == {
        "phantom01", "phantom05", "phantom06"}
    assert not (out / "yolo").exists()
    splits = json.loads((out / "splits.json").read_text())
    assert splits["splits"]["test"] == ["phantom05"]
    assert splits["split_source"] == "official"


# ============================================================================== halmstad
def test_halmstad_refuses_to_fake_the_matlab_labels(tmp_path):
    root = tmp_path / "halmstad"
    (root / "Video_Visible").mkdir(parents=True)
    (root / "Video_Visible" / "V_DRONE_001.mp4").write_bytes(b"")
    ad = build("halmstad", root)
    assert ad.sequences() == ["V_DRONE_001"]
    assert ad.class_of("V_DRONE_001") == "drone"
    assert ad.class_of("IR_BIRD_014") == "bird"
    with pytest.raises(NotImplementedError) as exc:
        ad.boxes("V_DRONE_001")
    msg = str(exc.value)
    assert "labels_json" in msg and "MCOS" in msg
    assert "official split" in msg and ".xlsx" in msg


def test_halmstad_reads_a_converted_sidecar_when_one_exists(tmp_path):
    """The unblock path: one MATLAB/Octave export into a documented JSON shape, and the
    adapter works — without this repo ever guessing at MCOS."""
    root = tmp_path / "halmstad"
    (root / "Video_Visible").mkdir(parents=True)
    (root / "Video_Visible" / "V_BIRD_002.mp4").write_bytes(b"")
    (root / "labels_json").mkdir(parents=True)
    (root / "labels_json" / "V_BIRD_002.json").write_text(
        json.dumps({"frames": {"0": [[10, 20, 30, 40]], "1": []}}))
    gt = build("halmstad", root).ground_truth("V_BIRD_002")
    assert gt.objects["bird_0"].frames[0] == pytest.approx((20.0, 30.0, 20.0, 20.0))
    assert gt.objects["bird_0"].ignore is True          # a bird is never a drone positive
    assert "self-chosen" in gt.meta["split_source"]


def test_conditions_are_empty_rather_than_inherited_from_the_catalog(ardmav_root, smid_root):
    """A corpus-level condition asserted of one clip is an unverified claim, and a
    condition-stratified table built from unverified claims is worse than no table."""
    assert build("ardmav", ardmav_root).conditions("phantom01") == ()
    assert build("uav_smid", smid_root).conditions("train") == ()
    assert DATASETS["ardmav"].conditions                # the corpus-level answer still exists


def test_uav_smid_uses_the_releases_own_split_as_its_sequences(smid_root):
    """One split is one sequence, not one image. 13,928 images treated as independent
    units would hand the bootstrap an absurdly tight interval for a corpus of unordered
    stills that carry no temporal structure at all."""
    ad = build("uav_smid", smid_root)
    assert ad.sequences() == ["train", "val", "test"]
    assert ad.split_of("val") == "val"
    assert "9,749" in ad.split_source()


def test_uav_smid_parses_voc_xml_not_yolo(smid_root):
    """Regression: this adapter was originally written against a Roboflow YOLO layout the
    release does not ship. Corner XML in, centre GT out."""
    gt = build("uav_smid", smid_root).ground_truth("train")
    cx, cy, w, h = gt.objects["drone_0"].frames[0]
    assert (cx, cy, w, h) == pytest.approx((50.0, 45.0, 20.0, 10.0))


def test_uav_smid_class_counts_name_every_distractor_actually_present(smid_root):
    """A false-alarm rate must say which distractors it was measured against and how many
    were on offer, rather than repeating the catalog from memory."""
    ad = build("uav_smid", smid_root)
    assert ad.class_counts() == {"drone": 2, "bird": 1, "helicopter": 1}
    assert ad.distractor_classes() == ["bird", "helicopter"]
