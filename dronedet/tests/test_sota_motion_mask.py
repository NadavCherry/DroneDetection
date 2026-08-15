"""Tests for the vendored YOLOMG motion mask.

These exist because the comparator's numbers are only worth publishing if its input is
actually the thing its authors describe. A mask that is silently all-zeros would hand us a
flattering win over a crippled baseline, and nothing downstream would notice: YOLOMG would
simply train into its appearance branch and score like a plain YOLOv5s.

So the properties tested here are the two that would make the comparison dishonest if they
failed -- the mask must respond to object motion, and it must NOT respond to camera motion.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from tools.sota.motion_mask import fd5_mask, motion_compensate


def _textured_background(h=540, w=960, seed=0):
    """Camera compensation needs something to track. Uniform noise is the honest choice:
    a gradient or a checkerboard gives KLT an unrealistically easy time."""
    rng = np.random.default_rng(seed)
    bg = rng.integers(40, 200, size=(h, w), dtype=np.uint8)
    return cv2.GaussianBlur(bg, (5, 5), 0)


def _shift(img, dx, dy):
    m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def _with_dot(img, cx, cy, r=3, value=255):
    out = img.copy()
    cv2.circle(out, (int(cx), int(cy)), r, int(value), -1)
    return out


def test_camera_pan_alone_leaves_almost_no_residual():
    """The whole point of compensation. If a pure pan survived into the mask, every frame
    of ego-motion video would be a solid wall of false motion and the mask would be noise."""
    bg = _textured_background()
    frames = [_shift(bg, -6 * k, -3 * k) for k in (0, 1, 2)]

    mask = fd5_mask(*frames)

    # Trim the border: the warp genuinely cannot know what lies outside the source frame,
    # and upstream's own `mask` return value exists to mark exactly that region.
    interior = mask[40:-40, 40:-40]
    assert interior.mean() < 12.0, f"pan leaked into the mask: mean {interior.mean():.1f}"


def test_an_object_moving_against_a_panning_camera_survives():
    """The signal the comparator is built on: after compensation, what is left is what
    moved differently from the scene."""
    bg = _textured_background()
    frames = [_with_dot(_shift(bg, -6 * k, -3 * k), 300 + 9 * k, 270 + 5 * k)
              for k in (0, 1, 2)]

    mask = fd5_mask(*frames)

    # The dot sits at its frame-1 position, because fd5 centres the mask on the middle frame.
    at_object = mask[270 + 5 - 8:270 + 5 + 8, 300 + 9 - 8:300 + 9 + 8].max()
    elsewhere = np.median(mask[40:-40, 40:-40])
    assert at_object > 60.0, f"moving object did not survive compensation: {at_object:.1f}"
    assert at_object > 8 * max(elsewhere, 1.0), (
        f"object peak {at_object:.1f} is not distinguishable from background {elsewhere:.1f}")


def test_the_mask_is_centred_on_the_middle_frame_not_the_last():
    """Guards the indexing convention documented in `fd5_mask`. Off-by-one here would
    misalign every mask against its label by one frame -- a ~9 px error at these speeds,
    which is larger than the targets themselves."""
    bg = _textured_background()
    # Static camera, dot far apart in each frame so the three candidate centres are distinct.
    frames = [_with_dot(bg, cx, 270) for cx in (200, 400, 600)]

    mask = fd5_mask(*frames)
    peaks = {cx: mask[262:278, cx - 10:cx + 10].max() for cx in (200, 400, 600)}

    assert peaks[400] == max(peaks.values()), (
        f"mask is not centred on the middle frame; peaks were {peaks}")


def test_compensate_refuses_colour_input():
    """Upstream only ever calls this on grayscale, and cv2.resize on a 3-channel image
    would silently produce a differently-shaped result rather than fail."""
    colour = np.zeros((540, 960, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="single-channel"):
        motion_compensate(colour, colour)


def test_untrackable_frames_do_not_produce_nan():
    """Deviation 2 from upstream, pinned. A black frame tracks nothing; upstream means an
    empty array and returns nan, which then propagates into the saved JPEG as garbage."""
    black = np.zeros((540, 960), dtype=np.uint8)

    compensated, _border, avg_dist, mx, my, _h = motion_compensate(black, black)

    assert np.isfinite([avg_dist, mx, my]).all(), "empty-track path produced nan"
    assert np.isfinite(compensated).all()
