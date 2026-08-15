"""YOLOMG's motion mask, vendored so the comparator is reproducible from this repo alone.

WHAT THIS IS
------------
YOLOMG (Guo et al., "Vision-based Drone-to-Drone Detection with Appearance and Pixel-Level
Motion Fusion", arXiv:2503.07115, March 2025) is the current state of the art on
drone-to-drone detection, from the same group that published ARD-MAV. Its second input
stream is a *motion mask*: the current frame differenced against an ego-motion-compensated
neighbour. This module reproduces that mask.

The two functions below are transcribed from the authors' own code at
https://github.com/Irisky123/YOLOMG (commit 090a74c), `test_code/MOD_Functions.py`
(`motion_compensate`) and `test_code/FD5_mask.py` (the averaging). They are GPL-3.0, as is
that repository. The transcription is deliberate and the deviations are enumerated below;
everything else is theirs, down to the magic numbers.

WHY VENDOR IT RATHER THAN IMPORT IT
-----------------------------------
`MOD_Functions` imports torch, torchvision, onnxruntime and PIL at module scope for
functions this pipeline never calls, so importing it drags a second deep-learning stack
into a job that needs OpenCV and nothing else. `motion_compensate` itself depends on
cv2 and numpy alone. Vendoring the 40 lines that matter keeps the mask build a CPU job
with no GPU stack, and -- more importantly -- puts the comparator's definition under our
tests, where a silent upstream change cannot alter published numbers.

THE DEVIATIONS, ALL OF THEM
---------------------------
1. Paths. Theirs are hard-coded absolute paths to the first author's home directory
   (`/home/user-guo/data/drone-dataset/...`). Ours are parameters.
2. `np.mean` of an empty list. When KLT tracks nothing -- a black frame, a hard cut --
   theirs emits a RuntimeWarning and returns nan for `motion_x/motion_y`. We return 0.0.
   The compensated image is unaffected: that path is already guarded upstream by their own
   `len(good_old) < 15` fallback to a near-identity homography.
3. Grayscale input. `motion_compensate` is called on already-grayscaled, already-blurred
   frames in every one of their FD*_mask callers, so the `cv2.resize` inside it operates
   on a single channel. We keep that contract and assert it.
4. THE HOMOGRAPHY IS MAPPED BACK INTO NATIVE COORDINATES. This one changes numbers, so
   the reasoning is set out in full below.

DEVIATION 4, IN FULL
--------------------
Upstream estimates the homography between points in a fixed 1920x1080 tracking space -- it
resizes both frames to `960*SCALE x 540*SCALE` regardless of the source -- and then hands
that matrix straight to `cv2.warpPerspective` on the frame at its NATIVE resolution. The
two coordinate systems only agree when the source is exactly 1920x1080. Anywhere else,
every translation term is off by the resize factor, the warp under-compensates the camera,
and the "motion mask" fills with residual ego-motion across the whole frame -- which is the
precise failure the mask exists to avoid.

Measured, same physical pan, mean interior residual before -> after this fix:

    960x540      6.35 -> 0.00
    1920x1080    0.00 -> 0.00      (S is the identity; bit-identical no-op)
    1920x1280    1.91 -> 0.18
    3840x2160   12.49 -> 0.00      (H carried exactly half the true pan)

WHY THIS IS NOT "IMPROVING THE COMPETITOR", WHICH THIS PACKAGE OTHERWISE FORBIDS
-------------------------------------------------------------------------------
Every video in ARD-MAV, and in upstream's own ARD100, is 1920x1080. There the resize is the
identity and this code path is unreachable, so no published YOLOMG number was ever produced
under the broken condition and correcting it cannot flatter them relative to their paper.

NPS is where it bites: 20 of its 50 clips are 1280x960 -- including ALL FOUR validation
clips, so YOLOMG would have selected `best.pt` on a validation set whose motion stream was
100% corrupt, plus 12 training and 4 test clips. NPS is also the dataset where YOLOMG
publish AP@0.5 = 0.95. Beating a 0.95 method on its own benchmark, having silently broken
the input its headline contribution rests on, is a retraction, not a result.

And the asymmetry is what settles it: OUR arm already gets this right. `dronedet/stabilize`
rescales its translation back to full-resolution pixels after estimating on a downscaled
frame. Applying the correction to ourselves and not to the competitor is exactly the thumb
on the scale this package exists to prevent.

Reproducing the method faithfully means reproducing what the method IS -- ego-compensate,
then difference -- not carrying a coordinate-space defect onto data where it fires and the
authors' data never did.

WHAT WE DO **NOT** CHANGE, ON PURPOSE
-------------------------------------
The mask is written as JPEG. JPEG quantisation damages exactly the faint few-pixel
residual this mask exists to carry, and we know it does -- it is the same effect that
made us build the temporal stack with 4:4:4 chroma. Writing PNG here would produce a
*better* motion mask than YOLOMG's, and then the thing we beat would not be YOLOMG. The
comparator is reproduced faithfully or it is not a comparator.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Upstream's LK parameters, verbatim (MOD_Functions.motion_compensate).
_LK = dict(winSize=(15, 15), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))

#: Upstream resizes both frames to 960x540 * SCALE before tracking, regardless of the
#: source resolution, and lays a GRID_W x GRID_H pixel grid of track points over the result.
_SCALE = 2
_GRID_W, _GRID_H = 32 * 2, 24 * 2

#: Tracks longer than this are treated as mismatches and dropped.
_MAX_TRACK_PX = 50

#: Below this many surviving tracks, upstream abandons the homography estimate and uses a
#: near-identity matrix -- note 0.999, not 1.0; that is theirs and we keep it.
_MIN_TRACKS = 15
_NEAR_IDENTITY = np.array([[0.999, 0, 0], [0, 0.999, 0], [0, 0, 1]])

#: Gaussian blur applied to every frame before differencing (FD5_mask).
_BLUR = (11, 11)


def motion_compensate(frame1: np.ndarray, frame2: np.ndarray):
    """Warp `frame1` into `frame2`'s frame of reference. Transcribed from YOLOMG.

    Grid-seeded KLT tracking -> RANSAC homography -> inverse-warp. Returns the compensated
    image, a mask marking pixels the warp pulled in from outside the source frame, the mean
    track length, the mean x/y translation, and the homography.

    Both inputs must be single-channel: upstream only ever calls this on grayscale.
    """
    if frame1.ndim != 2 or frame2.ndim != 2:
        raise ValueError(f"expected single-channel frames, got {frame1.shape} and "
                         f"{frame2.shape}; upstream calls this after cvtColor to GRAY")
    height, width = frame2.shape[:2]

    g1 = cv2.resize(frame1, (960 * _SCALE, 540 * _SCALE), interpolation=cv2.INTER_CUBIC)
    g2 = cv2.resize(frame2, (960 * _SCALE, 540 * _SCALE), interpolation=cv2.INTER_CUBIC)

    n_w = int(g2.shape[1] / _GRID_W - 1)
    n_h = int(g2.shape[0] / _GRID_H - 1)
    pts_prev = np.array(
        [(np.float32(i * _GRID_W + _GRID_W / 2.0), np.float32(j * _GRID_H + _GRID_H / 2.0))
         for i in range(n_w) for j in range(n_h)],
        dtype=np.float32).reshape(n_w * n_h, 1, 2)

    pts_cur, st, _err = cv2.calcOpticalFlowPyrLK(g1, g2, pts_prev, None, **_LK)
    good_new, good_old = pts_cur[st == 1], pts_prev[st == 1]

    dx = good_new[:, 0] - good_old[:, 0]
    dy = good_new[:, 1] - good_old[:, 1]
    keep = np.sqrt(dx * dx + dy * dy) <= _MAX_TRACK_PX
    # Deviation 2: upstream means an empty list here and gets nan with a warning.
    motion_x = float(dx[keep].mean()) if keep.any() else 0.0
    motion_y = float(dy[keep].mean()) if keep.any() else 0.0
    avg_dist = float(np.sqrt(dx[keep] ** 2 + dy[keep] ** 2).mean()) if keep.any() else 0.0

    if len(good_old) < _MIN_TRACKS:
        homography = _NEAR_IDENTITY.copy()
    else:
        homography, _status = cv2.findHomography(good_new, good_old, cv2.RANSAC, 3.0)
        if homography is None:            # RANSAC can fail outright on degenerate input
            homography = _NEAR_IDENTITY.copy()

    # Deviation 4, and the one that actually changes numbers -- see the module docstring.
    # `homography` was estimated between points living in the fixed 1920x1080 tracking
    # space above, but `warpPerspective` below is handed frame1 at its NATIVE resolution
    # with dsize from frame2's native shape. Feeding native coordinates to a matrix
    # expressed in tracking coordinates scales every translation term by the resize
    # factor. Map it across: for S mapping native -> tracking, H_native = S^-1 H S.
    #
    # At exactly 1920x1080 S is the identity and this is a bit-identical no-op, which is
    # why the defect is invisible on ARD-MAV (all 60 videos are 1920x1080) and on
    # upstream's own ARD100. It fires on 20 of NPS's 50 clips, which are 1280x960 --
    # anisotropically, 1.5x in x against 1.125x in y.
    sx, sy = (960 * _SCALE) / width, (540 * _SCALE) / height
    if (sx, sy) != (1.0, 1.0):
        S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
        homography = np.linalg.inv(S) @ homography @ S

    compensated = cv2.warpPerspective(
        frame1, homography, (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # The region the inverse warp had to invent, so callers can discount border artefacts.
    vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                      dtype=np.float32).reshape(-1, 1, 2)
    vertex_trans = cv2.perspectiveTransform(vertex, np.linalg.inv(homography))
    im = np.zeros(frame1.shape[:2], dtype="uint8")
    cv2.fillPoly(im, np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2), 255)

    return compensated, 255 - im, avg_dist, motion_x, motion_y, homography


#: The tap spacing in YOLOMG's `mask32`, which is the configuration their ARD100 data yaml
#: uses and therefore the one behind their headline numbers. It is NOT adjacent frames, and
#: the code does not say so anywhere: it falls out of the four-deep frame buffer in
#: `test_code/generate_mask5.py`, which calls `FD5_mask(lastFrame1, lastFrame3, current)`
#: and saves the result under `count - 2`. Tracing the shift gives, for the mask belonging
#: to frame t, the taps f[t-2], f[t], f[t+2] -- five frames of span at dt = 2. (Their
#: `mask31`/FD3 variant is the adjacent-frame version, dt = 1, and is not what mask32 is.)
#:
#: This is the number the whole comparison turns on. SpeckLock's stack spans 13 frames at
#: dt = 6; YOLOMG's mask spans 5 at dt = 2. Same family -- ego-compensate, then difference --
#: at very different apertures, which is precisely the variable worth reporting.
YOLOMG_MASK32_DT = 2

#: And it is not causal: the mask for frame t reads f[t+2]. SpeckLock's taps are t-12, t-6, t.
#: A like-for-like latency claim has to say this out loud, so it is recorded rather than
#: mentioned in prose that a table will not carry.
YOLOMG_MASK32_IS_CAUSAL = False


def fd5_mask(prev2: np.ndarray, prev1: np.ndarray, current: np.ndarray) -> np.ndarray:
    """YOLOMG's `mask32` motion mask for the frame at `prev1`. Transcribed from FD5_mask.

    Call it as ``fd5_mask(f[t - 2], f[t], f[t + 2])`` to get the mask belonging to frame t;
    see `YOLOMG_MASK32_DT` for why the spacing is 2 and not 1.

    Note the indexing, because it is the thing most likely to be got wrong: the mask belongs
    to the MIDDLE argument, not the last one. Upstream differences `prev1` against both
    neighbours compensated into `prev1`'s frame, then averages, so the mask is two-sided and
    centred. Getting this off by one misaligns every mask against its label by a couple of
    frames, which at these speeds is a larger error than the targets are wide -- it would
    quietly cripple the baseline and hand us a win we did not earn.

    Returns float64 in [0, 255], matching upstream, which writes the un-rounded average
    straight to `cv2.imwrite`.
    """
    g = [cv2.cvtColor(cv2.GaussianBlur(f, _BLUR, 0), cv2.COLOR_BGR2GRAY)
         if f.ndim == 3 else cv2.GaussianBlur(f, _BLUR, 0)
         for f in (prev2, prev1, current)]

    back, _m1, *_ = motion_compensate(g[0], g[1])
    fore, _m2, *_ = motion_compensate(g[2], g[1])
    return (cv2.absdiff(g[1], back).astype(np.float64)
            + cv2.absdiff(g[1], fore).astype(np.float64)) / 2


__all__ = ["fd5_mask", "motion_compensate"]
