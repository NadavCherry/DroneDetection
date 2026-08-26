"""The temporal ARD-MAV builder: taps, channel order, and stabilised label geometry.

The expensive bug this file exists to catch does not raise. If the pixels are warped
into the stabilised frame and the boxes are not, every tile still looks correct, every
label sits a few pixels off its target, and training converges to a model that is
confidently wrong. On ARD-MAV's 11.8 px median box, "a few pixels off" is the whole
object. So the end-to-end test below drives a synthetic sequence with a *known* camera
pan and asserts the written label lands on the target's measured brightest pixel.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_builder():
    """Import tools/make_dataset_external.py by path -- `tools` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "make_dataset_external", REPO / "tools" / "make_dataset_external.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_builder()


# --------------------------------------------------------------------------- taps

def test_taps_are_oldest_middle_newest():
    """Channel order is the shipped one: dstack([t-2dt, t-dt, t]), oldest first.

    Reversing it does not crash and does not look wrong; it hands the model the arrow
    of time backwards, against weights trained the other way round.
    """
    assert M._tap_indices(13, 6) == [0, 6, 12]


@pytest.mark.parametrize("held,expected", [
    (1,  [0, 0, 0]),      # first frame: every tap clamps to it
    (2,  [0, 0, 1]),
    (7,  [0, 0, 6]),      # t-dt only just becomes available
    (8,  [0, 1, 7]),
    (13, [0, 6, 12]),     # first fully-populated stack
])
def test_warmup_clamps_to_the_oldest_frame_held(held, expected):
    """Mirrors make_datasets_v3.clamp_chans' ``grays[max(0, t - 2*DT)]``.

    Without this the first 12 frames of every sequence are a regime the model never
    saw in training -- which is exactly when an interceptor most needs a detection.
    """
    assert M._tap_indices(held, 6) == expected


def test_taps_span_the_declared_aperture():
    """dt is the aperture, so a changed dt must move the taps and nothing else."""
    for dt in (1, 3, 6, 10):
        assert M._tap_indices(2 * dt + 1, dt) == [0, dt, 2 * dt]


# ------------------------------------------------- alignment is to NOW, not to frame 0

def _buf(shifts):
    """A ring buffer of (gray, dx, dy), each frame carrying a bright marker at x 30..33."""
    out = []
    for dx, dy in shifts:
        g = np.zeros((64, 64), np.uint8)
        g[30:34, 30:34] = 255
        out.append((g, float(dx), float(dy)))
    return out


def test_the_current_frame_is_never_resampled():
    """The newest tap comes through untouched, bit for bit.

    It is the frame the labels belong to. Resampling it would blur a 6 px target for no
    reason, and would degrade exactly the channel the detector relies on most.
    """
    buf = _buf([(100, -50), (60, -30), (0, 0)])
    assert np.array_equal(M._stack_aligned_to_now(buf, dt=1)[-1], buf[-1][0])


def test_a_huge_accumulated_drift_does_not_move_the_current_frame():
    """THE regression this refactor exists for.

    Registering every frame to frame 0 accumulates without bound on a moving camera,
    walks content off a canvas the size of the frame, and takes the ground-truth boxes
    with it -- 78 % of the labels on phantom23, 35 % overall, silently, and only on the
    arm whose claim was under test. Aligning to *now* makes the current frame's warp
    identically zero however far the camera has travelled.
    """
    for drift in (0, 500, 5000):
        buf = _buf([(drift + 8, drift - 4), (drift + 4, drift - 2), (drift, drift)])
        taps = M._stack_aligned_to_now(buf, dt=1)
        assert np.array_equal(taps[-1], buf[-1][0]), f"drift {drift} disturbed frame t"
        assert taps[-1].max() == 255, "the target vanished from the current frame"


def test_older_taps_are_warped_by_the_relative_shift():
    """A tap k frames behind moves by exactly the camera's motion since then."""
    buf = _buf([(10, 0), (5, 0), (0, 0)])       # t at 0, t-1 at +5, t-2 at +10
    taps = M._stack_aligned_to_now(buf, dt=1)
    newest_x = int(np.argmax(taps[-1].max(axis=0)))
    oldest_x = int(np.argmax(taps[0].max(axis=0)))
    assert newest_x == 30
    assert oldest_x == 40, f"expected +10 px of relative warp, got {oldest_x - newest_x}"


def test_a_static_camera_leaves_no_trail():
    """No camera motion, no warp: all three taps land identically. Otherwise a static
    scene would show a trail and the stack would report movement that never happened."""
    taps = M._stack_aligned_to_now(_buf([(0, 0)] * 3), dt=1)
    assert np.array_equal(taps[0], taps[1]) and np.array_equal(taps[1], taps[2])


# --------------------------------------------------------------------------- fixtures

def _synthetic_pan(path: Path, n_frames=30, W=320, H=240,
                   pan_per_frame=2, dot_world_per_frame=3, dot_x0=90, dot_y=120):
    """A textured static scene panned by a moving camera, with one bright target.

    Phase correlation needs texture to lock onto, so the background is noise rather
    than flat. The camera pans `pan_per_frame` px/frame; the target moves
    `dot_world_per_frame` px/frame in WORLD coordinates, so in raw frame coordinates
    it drifts at the difference of the two. Returns boxes in RAW frame coordinates,
    which is what the real annotations are.
    """
    rng = np.random.default_rng(7)
    pad = pan_per_frame * n_frames + 8
    bg = rng.integers(60, 140, size=(H, W + pad), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    assert writer.isOpened(), "mp4v writer unavailable"
    boxes, half = {}, 3
    for t in range(n_frames):
        ox = pan_per_frame * t
        frame = cv2.cvtColor(bg[:, ox:ox + W], cv2.COLOR_GRAY2BGR).copy()
        dot_frame_x = dot_x0 + (dot_world_per_frame - pan_per_frame) * t
        cv2.rectangle(frame, (dot_frame_x - half, dot_y - half),
                      (dot_frame_x + half, dot_y + half), (255, 255, 255), -1)
        writer.write(frame)
        boxes[t] = [(dot_frame_x - half, dot_y - half, dot_frame_x + half, dot_y + half)]
    writer.release()
    return boxes


def _read_tile(img_dir: Path, lbl_dir: Path, stem: str, tile: int):
    img = cv2.imread(str(img_dir / f"{stem}.jpg"), cv2.IMREAD_UNCHANGED)
    raw = (lbl_dir / f"{stem}.txt").read_text(encoding="utf-8").strip()
    labels = []
    for ln in raw.splitlines():
        f = ln.split()
        if len(f) == 5:
            labels.append((float(f[1]) * tile, float(f[2]) * tile))
    return img, labels


# --------------------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("temporal")
    vid = root / "pan.mp4"
    boxes = _synthetic_pan(vid)
    img_dir, lbl_dir = root / "images", root / "labels"
    frame_ids = [20]                       # well past 2*dt, so the stack is full
    n_img, n_box = M.extract_yolo_tiled_temporal(
        vid, boxes, frame_ids, img_dir, lbl_dir, "pan",
        min_side=0, tile=128, jitter=0.0, neg_per_frame=0, dt=6)
    return dict(root=root, img_dir=img_dir, lbl_dir=lbl_dir,
                n_img=n_img, n_box=n_box, boxes=boxes, tile=128)


def test_it_emits_a_tile_and_a_box(built):
    assert built["n_img"] >= 1 and built["n_box"] >= 1


def _channel_peaks(img, margin=25):
    """(x, y) centroid of the bright region in each channel.

    Centroid rather than `argmax`: the target is a saturated square, so argmax picks an
    arbitrary pixel off a flat plateau and jitters by the width of the target -- which
    on a 7 px square is enough to fail a 2 px assertion for no reason at all.
    """
    out = []
    for c in range(3):
        ch = img[:, :, c].astype(float)
        ys, xs = np.nonzero(ch >= ch.max() - margin)
        w = ch[ys, xs]
        out.append((float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())))
    return out


def test_the_stack_is_three_channels(built):
    img, _ = _read_tile(built["img_dir"], built["lbl_dir"], "pan_00020_0", built["tile"])
    assert img.ndim == 3 and img.shape[2] == 3


def test_the_target_leaves_a_trail_ordered_oldest_to_newest(built):
    """The target sits at three separated places, advancing with the channel index.

    Peak *position* rather than channel difference: an mp4v -> JPEG round-trip leaves a
    uniform colour cast of about one grey level, so a mean-difference test would pass on
    codec noise alone. Where the peaks are is a claim about motion; how far apart the
    channel values are is not.
    """
    img, _ = _read_tile(built["img_dir"], built["lbl_dir"], "pan_00020_0", built["tile"])
    xs = [p[0] for p in _channel_peaks(img)]
    assert len(set(xs)) == 3, f"channel peaks coincide at {xs}; no motion recorded"
    assert xs[0] < xs[1] < xs[2], f"peaks not ordered oldest->newest: {xs}"
    # dt=6 frames at 3 px/frame in world coordinates -> ~18 px between adjacent taps.
    assert 10 <= xs[1] - xs[0] <= 26, f"tap spacing {xs[1]-xs[0]} px does not match dt"
    assert 10 <= xs[2] - xs[1] <= 26, f"tap spacing {xs[2]-xs[1]} px does not match dt"


def test_the_label_lands_on_the_target_in_stabilised_coordinates(built):
    """THE test. Pixels are warped into the stabilised frame; boxes must follow.

    The camera pans 2 px/frame, so at frame 20 an unshifted box is ~40 px from the
    target -- far outside this tolerance and far outside a 12 px drone. This asserts
    the label sits on the brightest pixel of the NEWEST channel, which is where the
    target actually is at time t.
    """
    img, labels = _read_tile(built["img_dir"], built["lbl_dir"], "pan_00020_0", built["tile"])
    assert labels, "no label written"
    px, py = _channel_peaks(img)[2]          # newest channel = the target at time t
    lx, ly = labels[0]
    assert abs(lx - px) <= 3 and abs(ly - py) <= 3, (
        f"label at ({lx:.1f}, {ly:.1f}) but target is at ({px:.1f}, {py:.1f}) -- "
        "boxes are not being carried into stabilised coordinates")


def test_a_single_frame_tile_shows_one_target_not_three(built, tmp_path):
    """Control: the same video through the single-frame builder has coincident peaks.

    This is what keeps the trail test honest. It proves the three peaks come from the
    stacking and not from something in the synthetic fixture -- and it is the whole
    difference between the two ARD-MAV builders, stated as an assertion.
    """
    img_dir, lbl_dir = tmp_path / "sf_img", tmp_path / "sf_lbl"
    M.extract_yolo_tiled(built["root"] / "pan.mp4", built["boxes"], [20],
                         img_dir, lbl_dir, "pan", min_side=0, tile=128,
                         jitter=0.0, neg_per_frame=0)
    img, _ = _read_tile(img_dir, lbl_dir, "pan_00020_0", 128)
    xs = [p[0] for p in _channel_peaks(img)]
    assert max(xs) - min(xs) <= 2, f"single-frame tile should show one target, got peaks {xs}"


def test_empty_frame_selection_is_a_no_op(built, tmp_path):
    n_img, n_box = M.extract_yolo_tiled_temporal(
        built["root"] / "pan.mp4", built["boxes"], [], tmp_path / "i", tmp_path / "l",
        "pan", min_side=0, tile=128)
    assert (n_img, n_box) == (0, 0)


# --------------------------------------------------------------------------- jpeg chroma

def test_chroma_444_preserves_the_inter_channel_difference_better(tmp_path):
    """4:2:0 stores the channel differences at half resolution; 4:4:4 does not.

    The temporal signal lives in the differences BETWEEN channels, which JPEG carries
    as chroma. This measures the cost rather than asserting it, so the ablation in
    MISSION item 7 has a number to start from.
    """
    rng = np.random.default_rng(3)
    stack = np.dstack([rng.integers(0, 255, (64, 64), dtype=np.uint8) for _ in range(3)])
    out = {}
    for name, flag in (("420", False), ("444", True)):
        p = tmp_path / f"{name}.jpg"
        cv2.imwrite(str(p), stack, M._jpeg_params(95, flag))
        back = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(int)
        out[name] = np.abs(back - stack.astype(int)).mean()
    assert out["444"] < out["420"], (
        f"4:4:4 should round-trip a temporal stack more faithfully "
        f"(444 err {out['444']:.2f} vs 420 err {out['420']:.2f})")


# --------------------------------------------------------------- dt sweep safety

def test_a_non_default_dt_gets_its_own_directory():
    """A dt sweep must not overwrite the dataset the shipped weights were trained on.

    Every temporal builder wrote to a fixed name -- `nps_yolo_temporal` -- regardless of
    the `--dt` it was handed, while the CLI has advertised `--dt` all along. Running the
    sweep MISSION item 7 asks for would therefore have destroyed the dt=6 tiles in place,
    silently, after hours of CPU, leaving a table that compares a dt=2 model against
    numbers whose dataset no longer exists.
    """
    assert M._temporal_name("nps_yolo_temporal", True, M.TEMPORAL_DT) == \
        "nps_yolo_temporal", "the default must keep its bare name: paths, configs and " \
        "manifests already reference it"
    for dt in (2, 3, 4, 8, 12):
        got = M._temporal_name("nps_yolo_temporal", True, dt)
        assert got == f"nps_yolo_temporal_dt{dt}"
        assert got != "nps_yolo_temporal"


def test_single_frame_builds_never_get_a_dt_suffix():
    """dt is meaningless without a temporal stack, so it must not reach the name."""
    for dt in (2, 6, 12):
        assert M._temporal_name("nps_yolo_tiled", False, dt) == "nps_yolo_tiled"
