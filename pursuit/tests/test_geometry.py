"""Unit tests for :mod:`pursuit.geometry`.

The sensing model is the one place where a silent sign error is fatal and
invisible: a chaser that steers the wrong way still produces plausible-looking
telemetry, it just never intercepts. So the tests here are deliberately literal
about the conventions the module docstring promises --

* image frame is OpenCV (``+u`` right, ``+v`` down),
* body frame is REP-103 (``+x`` forward, ``+y`` **left**, ``+z`` up),
* therefore a target right of centre has **negative** azimuth and a target
  above centre has **positive** elevation,

-- and about the two round trips the rest of the package leans on
(pixel <-> bearing, range <-> pixel span).

Every camera used here is a real one or a deliberately awkward one. In
particular ``SIM`` is the rig's actual calibration copied from
``pursuit/sandbox.py``: its principal point sits at row 257 of an 840-row frame,
so any test that would still pass with ``cx, cy`` replaced by
``width / 2, height / 2`` is not testing anything.
"""
from __future__ import annotations

import math

import pytest

from pursuit.geometry import (
    Intrinsics,
    bearing_from_pixel,
    body_to_world,
    los_unit,
    normalized_offset,
    pixel_from_bearing,
    range_from_span,
    span_from_range,
    world_to_body,
    wrap_pi,
)

# The rig's real camera (see ``SIM_INTRINSICS`` in pursuit/sandbox.py). Strongly
# off-centre principal point -- cy is at 31% of the frame height, not 50%.
SIM = Intrinsics(width=1440, height=840, fx=921.8145952785566,
                 fy=923.9695163260498, cx=691.6137045337061,
                 cy=257.22911647658873)

# A textbook camera with the principal point exactly at the frame centre and
# fx == fy, so failures there are failures of the maths, not of the calibration.
CENTRED = Intrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)

ALL_CAMS = [CENTRED, SIM]
CAM_IDS = ["centred", "sim-offcentre"]

TARGET_SPAN_M = 0.47  # IRIS_SPAN_M, the drone the pursuit stack actually chases


def corner_and_grid_pixels(intr: Intrinsics):
    """Frame corners, principal point, frame centre and an interior grid."""
    w, h = float(intr.width), float(intr.height)
    pts = [
        (0.0, 0.0), (w - 1.0, 0.0), (0.0, h - 1.0), (w - 1.0, h - 1.0),  # corners
        (intr.cx, intr.cy),                                              # boresight
        (w * 0.5, h * 0.5),                                              # frame centre
    ]
    for i in range(5):
        for j in range(5):
            pts.append((w * i / 4.0, h * j / 4.0))
    return pts


# --------------------------------------------------------------------------
# bearing_from_pixel / pixel_from_bearing round trips
# --------------------------------------------------------------------------

@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_pixel_bearing_pixel_roundtrip_including_corners(intr):
    """u,v -> az,el -> u,v is exact to well under a thousandth of a pixel."""
    for u, v in corner_and_grid_pixels(intr):
        az, el = bearing_from_pixel(intr, u, v)
        u2, v2 = pixel_from_bearing(intr, az, el)
        assert u2 == pytest.approx(u, abs=1e-9), f"u round trip at ({u}, {v})"
        assert v2 == pytest.approx(v, abs=1e-9), f"v round trip at ({u}, {v})"


@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_bearing_pixel_bearing_roundtrip(intr):
    """The other direction: az,el -> u,v -> az,el, over +/-45 degrees."""
    angles = [math.radians(d) for d in (-45, -30, -12.5, -1e-3, 0.0, 1e-3, 12.5, 30, 45)]
    for az in angles:
        for el in angles:
            u, v = pixel_from_bearing(intr, az, el)
            az2, el2 = bearing_from_pixel(intr, u, v)
            assert az2 == pytest.approx(az, abs=1e-12)
            assert el2 == pytest.approx(el, abs=1e-12)


@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_principal_point_is_exactly_zero_bearing(intr):
    az, el = bearing_from_pixel(intr, intr.cx, intr.cy)
    assert az == 0.0
    assert el == 0.0
    u, v = pixel_from_bearing(intr, 0.0, 0.0)
    assert u == pytest.approx(intr.cx, abs=1e-12)
    assert v == pytest.approx(intr.cy, abs=1e-12)


def test_offcentre_principal_point_is_not_the_frame_centre():
    """The geometric centre of a frame with an off-centre cx,cy is NOT boresight.

    Hard-coded numbers so that swapping ``cx`` for ``width / 2`` anywhere in
    ``bearing_from_pixel`` fails loudly instead of quietly biasing every shot.
    """
    az, el = bearing_from_pixel(SIM, SIM.width * 0.5, SIM.height * 0.5)
    assert az == pytest.approx(-0.030784203379842258, abs=1e-12)
    assert el == pytest.approx(-0.17437560788017684, abs=1e-12)
    # 10 degrees of elevation bias is the whole ball game -- it must be big.
    assert math.degrees(el) < -9.9


@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_bearing_matches_closed_form_arctangent(intr):
    """Not the small-angle approximation: the true arctangent, as documented."""
    for du, dv in [(-500.0, -300.0), (-1.0, 7.0), (0.0, 0.0), (123.5, -456.25),
                   (700.0, 500.0)]:
        az, el = bearing_from_pixel(intr, intr.cx + du, intr.cy + dv)
        assert az == pytest.approx(-math.atan(du / intr.fx), abs=1e-15)
        assert el == pytest.approx(-math.atan(dv / intr.fy), abs=1e-15)


def test_edge_of_frame_differs_materially_from_small_angle_approximation():
    """The module claims the linear approximation is off by degrees at the edge."""
    u_edge = SIM.width - 1.0
    az, _ = bearing_from_pixel(SIM, u_edge, SIM.cy)
    linear = -(u_edge - SIM.cx) / SIM.fx
    assert math.degrees(abs(linear - az)) > 5.0
    # ...and the true value stays inside a physically sane half-FOV.
    assert math.degrees(abs(az)) == pytest.approx(39.05, abs=0.05)


@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_bearing_is_bounded_by_ninety_degrees_everywhere(intr):
    for u, v in corner_and_grid_pixels(intr):
        az, el = bearing_from_pixel(intr, u, v)
        assert abs(az) < math.pi / 2
        assert abs(el) < math.pi / 2


# --------------------------------------------------------------------------
# sign conventions
# --------------------------------------------------------------------------

def test_target_right_of_centre_gives_negative_azimuth():
    """+u is right, +y (and so +azimuth) is LEFT: the two must disagree."""
    az, _ = bearing_from_pixel(SIM, SIM.cx + 200.0, SIM.cy)
    assert az < 0.0
    assert az == pytest.approx(-math.atan(200.0 / SIM.fx), abs=1e-15)


def test_target_left_of_centre_gives_positive_azimuth():
    az, _ = bearing_from_pixel(SIM, SIM.cx - 200.0, SIM.cy)
    assert az > 0.0
    assert az == pytest.approx(math.atan(200.0 / SIM.fx), abs=1e-15)


def test_target_above_centre_gives_positive_elevation():
    """+v is DOWN in the image, elevation is positive UP."""
    _, el = bearing_from_pixel(SIM, SIM.cx, SIM.cy - 150.0)
    assert el > 0.0
    assert el == pytest.approx(math.atan(150.0 / SIM.fy), abs=1e-15)


def test_target_below_centre_gives_negative_elevation():
    _, el = bearing_from_pixel(SIM, SIM.cx, SIM.cy + 150.0)
    assert el < 0.0
    assert el == pytest.approx(-math.atan(150.0 / SIM.fy), abs=1e-15)


@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_azimuth_strictly_decreases_left_to_right_and_elevation_top_to_bottom(intr):
    us = [i * (intr.width - 1) / 12.0 for i in range(13)]
    azs = [bearing_from_pixel(intr, u, intr.cy)[0] for u in us]
    assert all(b < a for a, b in zip(azs, azs[1:])), azs
    vs = [j * (intr.height - 1) / 12.0 for j in range(13)]
    els = [bearing_from_pixel(intr, intr.cx, v)[1] for v in vs]
    assert all(b < a for a, b in zip(els, els[1:])), els


@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_the_two_axes_are_independent(intr):
    """Moving along a row changes only azimuth; along a column, only elevation."""
    az_a, el_a = bearing_from_pixel(intr, intr.cx - 111.0, intr.cy + 77.0)
    az_b, el_b = bearing_from_pixel(intr, intr.cx + 333.0, intr.cy + 77.0)
    assert el_a == pytest.approx(el_b, abs=1e-15)
    assert az_a > az_b
    az_c, el_c = bearing_from_pixel(intr, intr.cx - 111.0, intr.cy - 250.0)
    assert az_a == pytest.approx(az_c, abs=1e-15)
    assert el_c > el_a


@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_bearing_agrees_with_the_body_frame_direction_it_claims_to_describe(intr):
    """Project a body-frame point the way the simulator does, then read it back.

    ``pursuit/sandbox.py`` renders with ``u = cx - fx * left / fwd`` and
    ``v = cy - fy * up / fwd`` (REP-103 forward/left/up). Recovering
    ``az = atan(left / fwd)`` and ``el = atan(up / fwd)`` from those pixels is
    the entire contract between the camera and the guidance law.
    """
    for fwd, left, up in [(50.0, 0.0, 0.0), (50.0, 8.0, 3.0), (50.0, -8.0, 3.0),
                          (50.0, 8.0, -3.0), (12.0, -1.5, -4.0), (200.0, 30.0, 20.0)]:
        u = intr.cx - intr.fx * (left / fwd)
        v = intr.cy - intr.fy * (up / fwd)
        az, el = bearing_from_pixel(intr, u, v)
        assert az == pytest.approx(math.atan2(left, fwd), abs=1e-12)
        assert el == pytest.approx(math.atan2(up, fwd), abs=1e-12)
        # A target to the left of the nose really does get a positive azimuth.
        assert (az > 0.0) == (left > 0.0)
        assert (el > 0.0) == (up > 0.0)


def test_elevation_is_the_tangent_plane_angle_not_the_spherical_one():
    """Documented convention check, so nobody "fixes" one half of the pair.

    ``el`` is ``atan(up / fwd)``, i.e. measured in the image column, *not*
    ``atan(up / hypot(fwd, left))``. The two differ once the target is off the
    boresight in azimuth, and this test pins which one the module produces.
    """
    fwd, left, up = 50.0, 40.0, 10.0
    u = CENTRED.cx - CENTRED.fx * (left / fwd)
    v = CENTRED.cy - CENTRED.fy * (up / fwd)
    _, el = bearing_from_pixel(CENTRED, u, v)
    tangent_plane = math.atan2(up, fwd)
    spherical = math.atan2(up, math.hypot(fwd, left))
    assert el == pytest.approx(tangent_plane, abs=1e-12)
    assert abs(el - spherical) > math.radians(2.0)


# --------------------------------------------------------------------------
# normalized_offset, field of view, from_dict
# --------------------------------------------------------------------------

@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_normalized_offset_is_image_convention_x_right_y_down(intr):
    assert normalized_offset(intr, intr.cx, intr.cy) == (0.0, 0.0)
    ox, oy = normalized_offset(intr, intr.cx + 0.25 * intr.width,
                               intr.cy + 0.5 * intr.height)
    assert ox == pytest.approx(0.5, abs=1e-12)   # right of centre -> positive x
    assert oy == pytest.approx(1.0, abs=1e-12)   # below centre    -> positive y
    # Opposite sign convention to azimuth/elevation, by design.
    az, el = bearing_from_pixel(intr, intr.cx + 100.0, intr.cy + 100.0)
    ox2, oy2 = normalized_offset(intr, intr.cx + 100.0, intr.cy + 100.0)
    assert ox2 > 0.0 > az
    assert oy2 > 0.0 > el


def test_normalized_offset_exceeds_one_when_principal_point_is_off_centre():
    """"Roughly [-1, 1]" -- the far edge of an off-centre frame overshoots."""
    ox, _ = normalized_offset(SIM, SIM.width - 1.0, SIM.cy)
    assert ox > 1.0
    _, oy = normalized_offset(SIM, SIM.cx, SIM.height - 1.0)
    assert oy > 1.3


@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_fov_equals_the_bearing_spread_across_the_frame(intr):
    """hfov/vfov must be the same camera that bearing_from_pixel describes."""
    az_left, _ = bearing_from_pixel(intr, 0.0, intr.cy)
    az_right, _ = bearing_from_pixel(intr, float(intr.width), intr.cy)
    assert math.degrees(az_left - az_right) == pytest.approx(intr.hfov_deg, abs=1e-12)
    _, el_top = bearing_from_pixel(intr, intr.cx, 0.0)
    _, el_bot = bearing_from_pixel(intr, intr.cx, float(intr.height))
    assert math.degrees(el_top - el_bot) == pytest.approx(intr.vfov_deg, abs=1e-12)


def test_sim_camera_is_the_seventy_six_degree_lens_the_docstring_describes():
    assert SIM.hfov_deg == pytest.approx(75.95178612231403, abs=1e-9)
    assert SIM.vfov_deg == pytest.approx(47.797636801751615, abs=1e-9)


def test_intrinsics_from_dict_round_trips_and_coerces_types():
    intr = Intrinsics.from_dict({"width": "1440", "height": 840.0, "fx": 921,
                                 "fy": "923.5", "cx": 691.6, "cy": 257})
    assert (intr.width, intr.height) == (1440, 840)
    assert isinstance(intr.width, int) and isinstance(intr.fx, float)
    assert intr.fx == 921.0 and intr.fy == 923.5
    assert intr.cx == pytest.approx(691.6) and intr.cy == 257.0


# --------------------------------------------------------------------------
# range_from_span / span_from_range
# --------------------------------------------------------------------------

@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_range_span_roundtrip(intr):
    """range -> px -> range, over the band where the span stays above the floor."""
    r_max = intr.fx * TARGET_SPAN_M / 2.0  # the range at which the span hits 2 px
    for frac in (0.01, 0.05, 0.2, 0.5, 0.9, 1.0):
        range_m = r_max * frac
        span_px = span_from_range(intr, range_m, TARGET_SPAN_M)
        assert span_px >= 2.0, "test band must stay above the 2 px floor"
        back = range_from_span(intr, span_px, TARGET_SPAN_M)
        assert back is not None
        assert back == pytest.approx(range_m, rel=1e-12)


@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_span_roundtrip_the_other_way(intr):
    for span_px in (2.0, 3.5, 17.0, 120.0, 900.0):
        rng = range_from_span(intr, span_px, TARGET_SPAN_M)
        assert rng is not None
        assert span_from_range(intr, rng, TARGET_SPAN_M) == pytest.approx(span_px,
                                                                         rel=1e-12)


def test_range_and_span_use_fx_only_and_scale_as_documented():
    assert range_from_span(SIM, 10.0, TARGET_SPAN_M) == pytest.approx(
        SIM.fx * TARGET_SPAN_M / 10.0, rel=1e-15)
    assert span_from_range(SIM, 30.0, TARGET_SPAN_M) == pytest.approx(
        SIM.fx * TARGET_SPAN_M / 30.0, rel=1e-15)
    # Inverse proportionality: half the span is twice the range.
    r1 = range_from_span(SIM, 8.0, TARGET_SPAN_M)
    r2 = range_from_span(SIM, 4.0, TARGET_SPAN_M)
    assert r2 == pytest.approx(2.0 * r1, rel=1e-12)
    # ...and a bigger assumed target at the same pixel span is further away.
    assert range_from_span(SIM, 8.0, 2 * TARGET_SPAN_M) == pytest.approx(2.0 * r1,
                                                                        rel=1e-12)


@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_range_from_span_is_none_below_two_pixels(intr):
    for bad in (1.9999999, 1.0, 0.5, 0.0, -3.0):
        assert range_from_span(intr, bad, TARGET_SPAN_M) is None, bad


@pytest.mark.parametrize("intr", ALL_CAMS, ids=CAM_IDS)
def test_two_pixels_exactly_is_accepted_and_sets_the_max_reported_range(intr):
    """The gate is ``< 2.0``, so 2.0 px itself still yields a range."""
    at_floor = range_from_span(intr, 2.0, TARGET_SPAN_M)
    assert at_floor is not None
    assert at_floor == pytest.approx(intr.fx * TARGET_SPAN_M / 2.0, rel=1e-15)
    just_above = range_from_span(intr, 2.0000001, TARGET_SPAN_M)
    assert just_above is not None and just_above < at_floor


def test_sim_camera_max_monocular_range_for_the_iris():
    """At the 2 px floor the rig can only claim ranges out to ~217 m."""
    assert range_from_span(SIM, 2.0, TARGET_SPAN_M) == pytest.approx(216.626, abs=1e-3)


def test_range_from_span_is_none_for_missing_span_or_nonpositive_target_size():
    assert range_from_span(SIM, None, TARGET_SPAN_M) is None
    assert range_from_span(SIM, 30.0, 0.0) is None
    assert range_from_span(SIM, 30.0, -0.47) is None


def test_range_from_span_rejects_non_finite_spans():
    """A NaN or infinite span is a missing measurement, not a range.

    ``span_px < 2.0`` is False for NaN, so a bare comparison lets one through
    and it leaves as a NaN range -- which then poisons the range filter's EMA
    permanently instead of being skipped. Infinity fails the other way: it
    reports a range of exactly zero, i.e. contact.
    """
    assert range_from_span(SIM, float("nan"), TARGET_SPAN_M) is None
    assert range_from_span(SIM, float("inf"), TARGET_SPAN_M) is None
    assert range_from_span(SIM, float("-inf"), TARGET_SPAN_M) is None


def test_span_from_range_returns_inf_for_nonpositive_range():
    assert span_from_range(SIM, 0.0, TARGET_SPAN_M) == float("inf")
    assert span_from_range(SIM, -10.0, TARGET_SPAN_M) == float("inf")


def test_span_from_range_shrinks_monotonically_with_range():
    spans = [span_from_range(SIM, r, TARGET_SPAN_M) for r in (1, 2, 5, 10, 50, 300)]
    assert all(b < a for a, b in zip(spans, spans[1:])), spans


def test_range_error_is_quadratic_in_range_as_the_docstring_claims():
    """One pixel of span error at 10 px costs ~10% of the range."""
    true_span = 10.0
    r_true = range_from_span(SIM, true_span, TARGET_SPAN_M)
    r_err = range_from_span(SIM, true_span - 1.0, TARGET_SPAN_M)
    assert (r_err - r_true) / r_true == pytest.approx(1.0 / 9.0, rel=1e-12)
    # At 4 px the same one-pixel error costs a third.
    r4 = range_from_span(SIM, 4.0, TARGET_SPAN_M)
    r3 = range_from_span(SIM, 3.0, TARGET_SPAN_M)
    assert (r3 - r4) / r4 == pytest.approx(1.0 / 3.0, rel=1e-12)


# --------------------------------------------------------------------------
# body_to_world / world_to_body
# --------------------------------------------------------------------------

YAWS = [0.0, 1e-6, math.pi / 6, math.pi / 4, math.pi / 2, 2.0, math.pi,
        -math.pi / 3, -2.5, 3.5 * math.pi]

VECTORS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
           (12.0, -3.0, 4.5), (-7.5, 0.25, -2.0), (0.0, 0.0, 0.0)]


@pytest.mark.parametrize("yaw", YAWS)
@pytest.mark.parametrize("vec", VECTORS)
def test_body_world_roundtrip(yaw, vec):
    w = body_to_world(yaw, *vec)
    b = world_to_body(yaw, *w)
    for got, want in zip(b, vec):
        assert got == pytest.approx(want, abs=1e-12)
    # ...and the other way round, since either may be applied first.
    b2 = world_to_body(yaw, *vec)
    w2 = body_to_world(yaw, *b2)
    for got, want in zip(w2, vec):
        assert got == pytest.approx(want, abs=1e-12)


def test_body_to_world_at_known_yaws():
    # Yaw 0: body axes are world axes.
    assert body_to_world(0.0, 3.0, -4.0, 5.0) == pytest.approx((3.0, -4.0, 5.0),
                                                               abs=1e-12)
    # Yaw +90 deg (CCW from world +x): nose points along world +y...
    fwd = body_to_world(math.pi / 2, 1.0, 0.0, 0.0)
    assert fwd == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    # ...and the body's LEFT wing points along world -x.
    left = body_to_world(math.pi / 2, 0.0, 1.0, 0.0)
    assert left == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)
    # Yaw 180 deg reverses the horizontal plane, not the vertical.
    assert body_to_world(math.pi, 1.0, 2.0, 9.0) == pytest.approx((-1.0, -2.0, 9.0),
                                                                 abs=1e-12)


def test_world_to_body_at_known_yaws():
    # Flying north (yaw +90) with a target due world-east: it is off the right wing.
    fwd, left, up = world_to_body(math.pi / 2, 1.0, 0.0, 0.0)
    assert (fwd, left, up) == pytest.approx((0.0, -1.0, 0.0), abs=1e-12)
    assert left < 0.0  # right of the nose
    assert world_to_body(math.pi / 2, 0.0, 5.0, -2.0) == pytest.approx(
        (5.0, 0.0, -2.0), abs=1e-12)


@pytest.mark.parametrize("yaw", YAWS)
def test_rotation_is_yaw_only_and_norm_preserving(yaw):
    vx, vy, vz = 11.0, -6.0, 2.75
    wx, wy, wz = body_to_world(yaw, vx, vy, vz)
    assert wz == vz                                   # z is untouched, exactly
    assert math.hypot(wx, wy) == pytest.approx(math.hypot(vx, vy), abs=1e-12)
    assert world_to_body(yaw, wx, wy, wz)[2] == vz


@pytest.mark.parametrize("yaw", YAWS)
def test_body_to_world_adds_yaw_to_the_horizontal_heading(yaw):
    """A body vector at angle phi off the nose sits at phi + yaw in the world."""
    for phi in (0.0, 0.7, -1.9, 3.0):
        vx, vy = math.cos(phi) * 4.0, math.sin(phi) * 4.0
        wx, wy, _ = body_to_world(yaw, vx, vy, 0.0)
        assert wrap_pi(math.atan2(wy, wx) - (phi + yaw)) == pytest.approx(0.0,
                                                                         abs=1e-9)


def test_rotation_is_periodic_in_yaw():
    a = body_to_world(0.9, 3.0, -1.0, 2.0)
    b = body_to_world(0.9 + 2.0 * math.pi, 3.0, -1.0, 2.0)
    assert a == pytest.approx(b, abs=1e-12)


def test_body_to_world_is_a_proper_rotation_not_a_mirror():
    """Chained rotations compose: R(a) . R(b) == R(a + b)."""
    a, b = 0.7, -1.3
    once = body_to_world(a + b, 2.0, 5.0, -1.0)
    twice = body_to_world(a, *body_to_world(b, 2.0, 5.0, -1.0))
    assert once == pytest.approx(twice, abs=1e-12)


# --------------------------------------------------------------------------
# wrap_pi
# --------------------------------------------------------------------------

def test_wrap_pi_leaves_interior_angles_alone():
    for a in (0.0, 0.5, -0.5, 3.0, -3.0, 1e-12, -1e-12, math.pi - 1e-6):
        assert wrap_pi(a) == pytest.approx(a, abs=1e-12)


def test_wrap_pi_is_periodic():
    for a in (0.0, 0.5, -2.2, 3.0, -3.1):
        for k in (-3, -2, -1, 1, 2, 3):
            assert wrap_pi(a + 2.0 * math.pi * k) == pytest.approx(wrap_pi(a),
                                                                  abs=1e-9)


def test_wrap_pi_known_values():
    assert wrap_pi(3.0 * math.pi / 2.0) == pytest.approx(-math.pi / 2.0, abs=1e-12)
    assert wrap_pi(-3.0 * math.pi / 2.0) == pytest.approx(math.pi / 2.0, abs=1e-12)
    assert wrap_pi(2.0 * math.pi) == pytest.approx(0.0, abs=1e-12)
    assert wrap_pi(-2.0 * math.pi) == pytest.approx(0.0, abs=1e-12)
    assert wrap_pi(0.0) == 0.0


def test_wrap_pi_output_always_within_half_turn():
    a = -40.0
    while a <= 40.0:
        r = wrap_pi(a)
        assert -math.pi <= r <= math.pi, (a, r)
        a += 1e-3


def test_wrap_pi_makes_heading_errors_take_the_short_way_round():
    """The reason it exists: a 20 deg error across the +/-180 seam."""
    want, have = math.radians(170.0), math.radians(-170.0)
    err = wrap_pi(want - have)
    assert err == pytest.approx(math.radians(-20.0), abs=1e-12)
    assert abs(err) < math.pi


def test_wrap_pi_maps_exactly_pi_to_minus_pi():
    # SUSPECTED BUG: the docstring promises the half-open interval (-pi, pi], but
    # ``(a + pi) % 2pi - pi`` yields the *other* half-open interval [-pi, pi):
    # exactly +pi comes back as -pi. Asserting what the code currently does.
    assert wrap_pi(math.pi) == -math.pi
    assert wrap_pi(3.0 * math.pi) == pytest.approx(-math.pi, abs=1e-12)
    # Harmless for the magnitude-based checks in guidance/dynamics (|-pi| == |pi|),
    # but a sign-sensitive caller sitting exactly on the seam flips direction.
    assert wrap_pi(math.pi - 1e-12) > 0.0
    assert wrap_pi(math.pi) < 0.0


# --------------------------------------------------------------------------
# los_unit
# --------------------------------------------------------------------------

def test_los_unit_direction_and_range():
    unit, r = los_unit((1.0, 2.0, 3.0), (4.0, 6.0, 3.0))
    assert r == pytest.approx(5.0, abs=1e-12)
    assert unit == pytest.approx((0.6, 0.8, 0.0), abs=1e-12)


def test_los_unit_points_from_chaser_to_target():
    unit, r = los_unit((10.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert unit == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)
    assert r == pytest.approx(10.0, abs=1e-12)


def test_los_unit_is_normalised_and_matches_euclidean_range():
    cases = [((0, 0, 0), (1, 1, 1)), ((-5.5, 3.25, 90.0), (12.0, -8.0, 41.5)),
             ((100.0, 100.0, 20.0), (100.0, 100.0, 55.0)),
             ((3.0, -4.0, 0.5), (-3.0, 4.0, -0.5))]
    for c, t in cases:
        unit, r = los_unit(c, t)
        assert math.sqrt(sum(x * x for x in unit)) == pytest.approx(1.0, abs=1e-12)
        assert r == pytest.approx(math.dist(c, t), rel=1e-12)
        # unit * r reconstructs the displacement.
        for u, want in zip(unit, (t[0] - c[0], t[1] - c[1], t[2] - c[2])):
            assert u * r == pytest.approx(want, abs=1e-9)


def test_los_unit_is_antisymmetric():
    c, t = (1.0, -2.0, 8.0), (-4.0, 7.0, 2.5)
    fwd, r1 = los_unit(c, t)
    back, r2 = los_unit(t, c)
    assert r1 == pytest.approx(r2, rel=1e-15)
    assert fwd == pytest.approx(tuple(-x for x in back), abs=1e-15)


def test_los_unit_degenerate_when_the_two_aircraft_coincide():
    """Zero range returns the boresight fallback rather than dividing by zero."""
    unit, r = los_unit((7.0, 7.0, 7.0), (7.0, 7.0, 7.0))
    assert unit == (1.0, 0.0, 0.0)
    assert r == 0.0
    # Strictly inside the 1e-9 m guard band, same fallback.
    unit, r = los_unit((0.0, 0.0, 0.0), (1e-10, 0.0, 0.0))
    assert unit == (1.0, 0.0, 0.0)
    assert r == 0.0


def test_los_unit_just_outside_the_guard_band_is_a_real_measurement():
    unit, r = los_unit((0.0, 0.0, 0.0), (0.0, 1e-8, 0.0))
    assert r == pytest.approx(1e-8, rel=1e-9)
    assert unit == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_los_unit_accepts_any_sequence_including_numpy_and_ints():
    import numpy as np

    unit_a, r_a = los_unit([0, 0, 0], [0, 0, 4])
    assert unit_a == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)
    assert r_a == pytest.approx(4.0, abs=1e-12)
    unit_b, r_b = los_unit(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 4.0]))
    assert unit_b == pytest.approx(unit_a, abs=1e-12)
    assert r_b == pytest.approx(r_a, abs=1e-12)
    assert all(isinstance(x, float) for x in unit_b)


def test_los_unit_composes_with_world_to_body_and_bearing_the_way_the_loop_does():
    """End-to-end: two world poses -> pixel -> bearing, signs intact.

    Chaser at the origin heading world +x, target 100 m ahead, 10 m to the world
    left (+y) and 5 m below. Left of the nose must read positive azimuth, below
    must read negative elevation, and the monocular range must recover the true
    slant range from the pixel span it implies.
    """
    chaser = (0.0, 0.0, 50.0)
    target = (100.0, 10.0, 45.0)
    yaw = 0.0
    (lx, ly, lz), rng = los_unit(chaser, target)
    fwd, left, up = world_to_body(yaw, lx * rng, ly * rng, lz * rng)
    assert (fwd, left, up) == pytest.approx((100.0, 10.0, -5.0), abs=1e-12)

    u, v = SIM.cx - SIM.fx * (left / fwd), SIM.cy - SIM.fy * (up / fwd)
    az, el = bearing_from_pixel(SIM, u, v)
    assert az > 0.0 and el < 0.0
    assert az == pytest.approx(math.atan2(10.0, 100.0), abs=1e-12)
    assert el == pytest.approx(math.atan2(-5.0, 100.0), abs=1e-12)
    assert u < SIM.cx and v > SIM.cy  # left of centre, below centre, in pixels

    span_px = span_from_range(SIM, rng, TARGET_SPAN_M)
    assert range_from_span(SIM, span_px, TARGET_SPAN_M) == pytest.approx(rng,
                                                                        rel=1e-12)
    assert rng == pytest.approx(math.sqrt(100.0 ** 2 + 10.0 ** 2 + 5.0 ** 2),
                                rel=1e-12)
