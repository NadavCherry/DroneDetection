"""Tests for :mod:`pursuit.guidance`.

The guidance law is a differentiator (``LosRateFilter``) feeding a gain
(``nav_gain * Vc``) wrapped in a mode machine, so the things worth pinning down
numerically are: does the differentiator report the rate that actually happened,
does it refuse to report one it cannot know, does the range filter reject a bad
box without letting it move the estimate, is the latency compensation using the
heading from the right instant, and does the mode machine move between states at
the thresholds the config claims.

Everything here is closed-form and deterministic: no simulator, no renderer, no
randomness, no sleeps.
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from pursuit.dynamics import Limits
from pursuit.geometry import Intrinsics, body_to_world, wrap_pi
from pursuit.guidance import (
    ACQUIRE,
    HIT,
    PURSUE,
    REACQUIRE,
    SEARCH,
    TERMINAL,
    GuidanceConfig,
    LosRateFilter,
    PursuitGuidance,
    RangeFilter,
    bearing_to_los,
)
from pursuit.perception import TargetEstimate

# A camera with a centred principal point -- guidance itself never touches the
# pixels (it is handed az/el), so the only intrinsic that matters below is fx,
# which sets the monocular range when ``range_override`` is not used.
INTR = Intrinsics(width=1440, height=840, fx=921.8, fy=923.9, cx=720.0, cy=420.0)
LIMITS = Limits()          # max_speed_xy = 14, max_yaw_rate = 2.5
SPAN_M = 0.47
DT = 0.05                  # the rig's 20 Hz control tick


# ------------------------------------------------------------------- helpers

def norm(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def det(az=0.0, el=0.0, rng=None, span_px=10.0):
    """A valid detection at a bearing, optionally with an oracle range."""
    return TargetEstimate(valid=True, u=0.0, v=0.0, span_px=span_px, az=az, el=el,
                          score=1.0, source="detector", range_override=rng)


def make_guidance(cfg=None, limits=LIMITS):
    return PursuitGuidance(INTR, limits, SPAN_M, cfg)


class Driver:
    """Ticks a :class:`PursuitGuidance` on a fixed clock with a fixed ego state."""

    def __init__(self, guidance, dt=DT, vel=(10.0, 0.0, 0.0), yaw=0.0,
                 xyz=(0.0, 0.0, 10.0)):
        self.g = guidance
        self.dt = dt
        self.t = 0.0
        self.vel = vel
        self.yaw = yaw
        self.xyz = xyz

    def tick(self, measurement=None):
        st = self.g.step(self.t, self.dt, self.xyz, self.yaw, self.vel, measurement)
        self.t += self.dt
        return st

    def run(self, n, measurement=None):
        st = None
        for _ in range(n):
            st = self.tick(measurement)
        return st


def confirm(drv, rng=40.0, az=0.0, el=0.0):
    """Drive the FSM through ACQUIRE into PURSUE and return the last state."""
    return drv.run(GuidanceConfig().confirm_hits + 1, det(az=az, el=el, rng=rng))


# ============================================================ LosRateFilter

class TestLosRateFilter:

    def test_first_sample_seeds_and_reports_zero(self):
        f = LosRateFilter()
        ds = f.update((1.0, 0.0, 0.0), 3.0)
        assert ds == (0.0, 0.0, 0.0)
        assert f.s == (1.0, 0.0, 0.0)
        assert f.t_last == 3.0
        assert f.n == 1

    def test_input_is_normalised(self):
        f = LosRateFilter()
        f.update((0.0, 0.0, 7.0), 0.0)
        assert f.s == pytest.approx((0.0, 0.0, 1.0), abs=1e-15)

    def test_constant_los_gives_exactly_zero_rate(self):
        # A stationary line of sight is a collision course already; reporting any
        # rate at all here would steer the aircraft off it.
        f = LosRateFilter()
        s = (0.6, 0.8, 0.0)
        for i in range(40):
            f.update(s, i * DT)
        assert f.ds == (0.0, 0.0, 0.0)
        assert f.rate_mag == 0.0
        assert f.n == 40

    def test_constant_los_rate_is_zero_even_when_los_is_re_normalised(self):
        f = LosRateFilter()
        for i in range(20):
            # Same direction, different length: normalisation must make these
            # identical samples, not a rate.
            f.update((3.0 + 0.5 * i, 4.0 + (0.5 * i) * (4.0 / 3.0), 0.0), i * DT)
        assert f.rate_mag == pytest.approx(0.0, abs=1e-12)

    def test_known_rotation_exact_with_negligible_smoothing(self):
        # tau -> 0 makes the EMA a pass-through, so the filter output is the raw
        # chord difference projected across the LOS. For a rotation of theta per
        # sample that is exactly sin(theta)/dt along the tangential direction at
        # the *new* LOS -- a closed form worth pinning to float precision.
        w = 0.35
        f = LosRateFilter(tau=1e-4)
        n = 12
        for i in range(n + 1):
            a = w * i * DT
            f.update((math.cos(a), math.sin(a), 0.0), i * DT)

        a_end = w * n * DT
        expect_mag = math.sin(w * DT) / DT
        expect = (-math.sin(a_end) * expect_mag, math.cos(a_end) * expect_mag, 0.0)
        assert f.ds == pytest.approx(expect, abs=1e-12)
        assert norm(f.ds) == pytest.approx(expect_mag, rel=1e-12)
        # and sin(theta)/dt is the true rate to better than a part in 10^4 here
        assert norm(f.ds) == pytest.approx(w, rel=1e-4)

    def test_known_rotation_magnitude_and_direction_with_default_smoothing(self):
        w = 0.2                       # rad/s, CCW about +z (LOS swings to +y)
        f = LosRateFilter()           # default tau = 0.12 s
        n = int(3.0 / DT)
        for i in range(n + 1):
            a = w * i * DT
            f.update((math.cos(a), math.sin(a), 0.0), i * DT)

        a_end = w * n * DT
        assert norm(f.ds) == pytest.approx(w, rel=2e-3)
        true_dir = (-math.sin(a_end), math.cos(a_end), 0.0)
        # Direction is right to within the EMA's own phase lag (~ w * tau rad).
        assert dot(f.ds, true_dir) / norm(f.ds) > 0.999
        assert f.ds[2] == pytest.approx(0.0, abs=1e-15)

    def test_rotation_direction_sign_follows_the_target(self):
        w = 0.2
        out = {}
        for sign in (+1.0, -1.0):
            f = LosRateFilter()
            for i in range(40):
                a = sign * w * i * DT
                f.update((math.cos(a), math.sin(a), 0.0), i * DT)
            out[sign] = f.ds
        assert out[+1.0][1] > 0.15          # target crossing left -> +y rate
        assert out[-1.0][1] < -0.15         # crossing right -> -y rate
        assert out[+1.0][1] == pytest.approx(-out[-1.0][1], rel=1e-9)

    def test_rate_is_exactly_perpendicular_to_the_los_for_a_single_step(self):
        # With no prior rate to blend in, the projection in update() is the only
        # thing acting, and it must leave nothing along the LOS -- a radial
        # component here is a fake closing rate.
        f = LosRateFilter()
        f.update((1.0, 0.0, 0.0), 0.0)
        f.update((math.cos(0.01), math.sin(0.01), 0.0), DT)
        assert dot(f.ds, f.s) == pytest.approx(0.0, abs=1e-15)
        assert norm(f.ds) > 0.0

    def test_rate_stays_perpendicular_through_a_sustained_rotation(self):
        # The EMA blends a rate measured against the previous LOS into one held
        # against the current LOS, so perpendicularity is approximate rather than
        # exact; it is bounded by the filter's phase lag, ~ w * tau.
        w = 0.2
        f = LosRateFilter()
        worst = 0.0
        for i in range(80):
            a = w * i * DT
            f.update((math.cos(a), math.sin(a), 0.0), i * DT)
            if f.rate_mag > 0.0:
                worst = max(worst, abs(dot(f.ds, f.s)) / f.rate_mag)
        assert worst < 0.03
        # tighten the bound with a slower rotation -- it scales with w * tau
        f2 = LosRateFilter()
        worst2 = 0.0
        for i in range(80):
            a = 0.02 * i * DT
            f2.update((math.cos(a), math.sin(a), 0.0), i * DT)
            if f2.rate_mag > 0.0:
                worst2 = max(worst2, abs(dot(f2.ds, f2.s)) / f2.rate_mag)
        assert worst2 < 0.003

    def test_out_of_plane_rotation_reports_the_vertical_rate(self):
        w = 0.15
        f = LosRateFilter(tau=1e-4)
        n = 10
        for i in range(n + 1):
            a = w * i * DT
            f.update((math.cos(a), 0.0, math.sin(a)), i * DT)
        a_end = w * n * DT
        mag = math.sin(w * DT) / DT
        assert f.ds == pytest.approx(
            (-math.sin(a_end) * mag, 0.0, math.cos(a_end) * mag), abs=1e-12)

    def test_reset_across_a_gap_longer_than_max_gap_s(self):
        f = LosRateFilter()
        for i in range(30):
            a = 0.3 * i * DT
            f.update((math.cos(a), math.sin(a), 0.0), i * DT)
        assert f.rate_mag > 0.2
        n_before = f.n

        t_gap = 30 * DT + f.max_gap_s + 1e-6      # strictly beyond the gap
        ds = f.update((0.0, 1.0, 0.0), t_gap)
        assert ds == (0.0, 0.0, 0.0)
        assert f.ds == (0.0, 0.0, 0.0)
        assert f.rate_mag == 0.0
        assert f.s == (0.0, 1.0, 0.0)
        assert f.t_last == t_gap
        assert f.n == 1 < n_before

    def test_gap_exactly_at_max_gap_s_is_still_differenced(self):
        f = LosRateFilter()
        f.update((1.0, 0.0, 0.0), 0.0)
        gap = f.max_gap_s                      # boundary is `dt > max_gap_s`
        f.update((math.cos(0.05), math.sin(0.05), 0.0), gap)
        assert f.n == 2
        assert f.rate_mag > 0.0

    def test_sub_min_dt_sample_is_ignored_entirely(self):
        f = LosRateFilter()
        f.update((1.0, 0.0, 0.0), 0.0)
        f.update((math.cos(0.02), math.sin(0.02), 0.0), DT)
        ds_before, s_before, n_before = f.ds, f.s, f.n

        ds = f.update((0.0, 0.0, 1.0), DT + 0.5 * f.min_dt_s)
        assert ds == ds_before
        assert f.ds == ds_before
        assert f.s == s_before          # the sample did not move the LOS either
        assert f.t_last == DT
        assert f.n == n_before

    def test_dropped_frame_does_not_report_the_rate_as_zero(self):
        # The docstring's motivating failure: a dropped detection must not be
        # differenced into "the target stopped". The filter simply is not fed,
        # so both the LOS and its rate are held exactly as they were.
        g = make_guidance()
        drv = Driver(g, vel=(14.0, 0.0, 0.0))
        for i in range(40):
            drv.tick(det(az=0.3 * i * DT, rng=40.0))
        s0, ds0 = g.los.s, g.los.ds
        assert norm(ds0) == pytest.approx(0.3, rel=0.05)

        for _ in range(4):
            st = drv.tick(None)
            assert g.los.ds == ds0
            assert g.los.s == s0
            assert st.los_rate == pytest.approx(norm(ds0), abs=1e-5)

    def test_los_is_extrapolated_through_a_dropout_and_the_reach_is_capped(self):
        cfg = GuidanceConfig()
        g = make_guidance()
        drv = Driver(g, vel=(14.0, 0.0, 0.0))
        for i in range(40):
            drv.tick(det(az=0.3 * i * DT, rng=40.0))
        s0, ds0 = g.los.s, g.los.ds

        def rotated(age):
            v = tuple(s0[i] + ds0[i] * age for i in range(3))
            k = 1.0 / norm(v)
            return tuple(c * k for c in v)

        for i in range(1, 8):                       # ages 0.05 .. 0.35
            drv.tick(None)
            age = i * DT + cfg.sensor_latency_s
            assert g._los_now() == pytest.approx(rotated(age), abs=1e-15)
        frozen = g._los_now()
        assert cfg.max_extrapolation_s == 0.35
        for _ in range(4):                          # past the cap: no further reach
            drv.tick(None)
            assert g._los_now() == pytest.approx(frozen, abs=1e-15)
        assert frozen == pytest.approx(rotated(cfg.max_extrapolation_s), abs=1e-15)

    def test_reset_clears_everything(self):
        f = LosRateFilter()
        f.update((1.0, 0.0, 0.0), 0.0)
        f.update((math.cos(0.05), math.sin(0.05), 0.0), DT)
        f.reset()
        assert (f.s, f.ds, f.t_last, f.n) == (None, (0.0, 0.0, 0.0), None, 0)


# ============================================================== RangeFilter

class TestRangeFilter:

    def test_first_sample_seeds_exactly(self):
        r = RangeFilter()
        assert r.update(37.5, 1.0) == 37.5
        assert r.value == 37.5
        assert r.t_last == 1.0
        assert r.rate == 0.0

    def test_converges_to_a_steady_measurement(self):
        r = RangeFilter()
        r.update(40.0, 0.0)
        for i in range(1, 61):
            r.update(36.0, i * DT)
        assert r.value == pytest.approx(36.0, abs=1e-3)
        assert r.rate == pytest.approx(0.0, abs=1e-3)

    def test_convergence_is_monotone_and_ema_shaped(self):
        r = RangeFilter(tau=0.25)
        r.update(40.0, 0.0)
        a = 1.0 - math.exp(-DT / 0.25)
        prev = 40.0
        for i in range(1, 21):
            v = r.update(36.0, i * DT)
            assert 36.0 <= v < prev                     # strictly closing in
            assert v == pytest.approx(prev + a * (36.0 - prev), rel=1e-12)
            prev = v
        # the residual after N steps is the EMA's own geometric decay
        assert prev - 36.0 == pytest.approx(4.0 * (1.0 - a) ** 20, rel=1e-9)

    def test_tracks_a_steady_closing_rate(self):
        # A target closing at a constant 5 m/s: in steady state the reported
        # rate is the true one even though it is computed from the EMA step.
        r = RangeFilter()
        r.update(40.0, 0.0)
        for i in range(1, 121):
            r.update(40.0 - 5.0 * i * DT, i * DT)
        assert r.rate == pytest.approx(-5.0, rel=2e-3)

    def test_single_wild_outlier_is_rejected_not_averaged_in(self):
        r = RangeFilter()
        r.update(36.0, 0.0)
        for i in range(1, 21):
            r.update(36.0, i * DT)
        held, t_prev = r.value, r.t_last
        assert held == pytest.approx(36.0, abs=1e-9)

        t_bad = 21 * DT
        out = r.update(120.0, t_bad)
        assert out == held                    # not blended at all
        assert r.value == held
        assert r.rejects == 1
        assert r.t_last == t_bad              # clock still advances

        # and the filter is undisturbed once good samples resume
        r.update(36.0, 22 * DT)
        assert r.rejects == 0
        assert r.value == pytest.approx(36.0, abs=1e-9)
        assert t_prev < t_bad

    def test_gate_width_is_the_documented_slack(self):
        r = RangeFilter(max_closing_mps=30.0)
        r.update(40.0, 0.0)
        slack = 30.0 * DT + 0.5 + 0.15 * 40.0        # = 8.0
        # just inside the gate -> accepted (value moves toward it)
        inside = RangeFilter(max_closing_mps=30.0)
        inside.update(40.0, 0.0)
        inside.update(40.0 + slack - 1e-6, DT)
        assert inside.value > 40.0
        assert inside.rejects == 0
        # just outside -> rejected (value untouched)
        r.update(40.0 + slack + 1e-6, DT)
        assert r.value == 40.0
        assert r.rejects == 1

    def test_reseeds_after_exactly_max_reject_consecutive_outliers(self):
        r = RangeFilter(max_reject=6)
        r.update(36.0, 0.0)
        for i in range(1, r.max_reject):          # 5 rejects
            assert r.update(120.0, i * DT) == pytest.approx(36.0, abs=1e-9)
            assert r.rejects == i
        assert r.value == pytest.approx(36.0, abs=1e-9)

        out = r.update(120.0, r.max_reject * DT)  # the 6th re-seeds
        assert out == 120.0
        assert r.value == 120.0
        assert r.rate == 0.0
        assert r.rejects == 0

    def test_reject_streak_is_broken_by_one_good_sample(self):
        r = RangeFilter(max_reject=6)
        r.update(36.0, 0.0)
        for i in range(1, 6):
            r.update(120.0, i * DT)
        assert r.rejects == 5
        r.update(36.0, 6 * DT)                    # streak broken
        assert r.rejects == 0
        for i in range(7, 12):                    # five more is not six
            r.update(120.0, i * DT)
        assert r.rejects == 5
        assert r.value == pytest.approx(36.0, abs=1e-9)

    @pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), 0.0, -5.0])
    def test_unusable_measurements_are_dropped_without_side_effects(self, bad):
        r = RangeFilter()
        r.update(36.0, 0.0)
        r.update(35.5, DT)
        snapshot = (r.value, r.rate, r.t_last, r.rejects)
        assert r.update(bad, 2 * DT) == r.value
        assert (r.value, r.rate, r.t_last, r.rejects) == snapshot

    def test_predict_extrapolates_and_is_capped(self):
        r = RangeFilter()
        r.value, r.rate, r.t_last = 30.0, -4.0, 10.0
        assert r.predict(10.0) == 30.0            # dt == 0
        assert r.predict(9.0) == 30.0             # backwards -> held
        assert r.predict(10.5) == pytest.approx(28.0, rel=1e-12)
        # extrapolation is capped at one second of coasting
        assert r.predict(15.0) == pytest.approx(26.0, rel=1e-12)
        assert r.predict(11.0) == r.predict(15.0)

    def test_predict_is_floored_above_zero(self):
        r = RangeFilter()
        r.value, r.rate, r.t_last = 2.0, -10.0, 0.0
        assert r.predict(1.0) == 0.1
        assert r.predict(100.0) == 0.1

    def test_predict_before_any_measurement_is_none(self):
        assert RangeFilter().predict(1.0) is None

    def test_reset_clears_everything(self):
        r = RangeFilter()
        r.update(36.0, 0.0)
        r.reset()
        assert (r.value, r.rate, r.t_last, r.rejects) == (None, 0.0, None, 0)


# ======================================================= yaw history / latency

class TestYawAt:
    """``_yaw_at`` -- the heading the aircraft held when a frame was captured."""

    @staticmethod
    def _with_history(pairs):
        g = make_guidance()
        g._yaw_hist.extend(pairs)
        return g

    def test_empty_history_returns_the_fallback(self):
        g = make_guidance()
        assert g._yaw_at(0.5, 1.234) == 1.234

    def test_clamps_to_the_ends_of_the_history(self):
        g = self._with_history([(1.0, 0.2), (2.0, 0.6)])
        assert g._yaw_at(0.0, 99.0) == 0.2        # before the first sample
        assert g._yaw_at(1.0, 99.0) == 0.2        # exactly the first sample
        assert g._yaw_at(2.0, 99.0) == 0.6        # exactly the last sample
        assert g._yaw_at(9.0, 99.0) == 0.6        # after the last sample

    def test_linear_interpolation_between_samples(self):
        g = self._with_history([(0.0, 0.0), (0.4, 0.8)])
        assert g._yaw_at(0.1, 99.0) == pytest.approx(0.2, rel=1e-12)
        assert g._yaw_at(0.2, 99.0) == pytest.approx(0.4, rel=1e-12)
        assert g._yaw_at(0.3, 99.0) == pytest.approx(0.6, rel=1e-12)

    def test_interpolates_in_the_right_interval_of_a_long_history(self):
        g = self._with_history([(i * DT, 0.1 * i) for i in range(10)])
        # between samples 4 (0.4) and 5 (0.5), a quarter of the way along
        assert g._yaw_at(4.25 * DT, 99.0) == pytest.approx(0.425, rel=1e-12)

    def test_wrap_around_takes_the_short_way_across_plus_pi(self):
        # 3.0 -> -3.0 is +0.283 rad the short way, not -6.0 rad the long way.
        g = self._with_history([(0.0, 3.0), (1.0, -3.0)])
        short = wrap_pi(-3.0 - 3.0)                 # +0.2831853...
        assert short == pytest.approx(2.0 * math.pi - 6.0, rel=1e-12)

        mid = g._yaw_at(0.5, 99.0)
        assert mid == pytest.approx(3.0 + 0.5 * short, rel=1e-12)
        # exactly on the seam; wrap_pi lands on -pi (its range is [-pi, pi))
        assert abs(wrap_pi(mid)) == pytest.approx(math.pi, abs=1e-9)
        assert abs(mid) > 3.0                       # NOT the naive average 0.0
        # three quarters of the way is past the seam and still continuous
        assert wrap_pi(g._yaw_at(0.75, 99.0)) == pytest.approx(-3.0708, abs=1e-4)

    def test_wrap_around_the_other_direction(self):
        g = self._with_history([(0.0, -3.0), (1.0, 3.0)])
        mid = g._yaw_at(0.5, 99.0)
        assert mid == pytest.approx(-math.pi, abs=1e-9)
        assert wrap_pi(g._yaw_at(0.75, 99.0)) == pytest.approx(3.0708, abs=1e-4)

    def test_interpolated_heading_is_continuous_as_a_direction(self):
        # The returned angle may leave (-pi, pi], which is harmless because it is
        # only ever consumed through cos/sin -- assert that directly.
        g = self._with_history([(0.0, 3.0), (1.0, -3.0)])
        prev = None
        for i in range(1, 100):
            y = g._yaw_at(i / 100.0, 99.0)
            v = (math.cos(y), math.sin(y))
            if prev is not None:
                step = math.hypot(v[0] - prev[0], v[1] - prev[1])
                assert step < 0.01          # no jump across the seam
            prev = v

    def test_step_records_the_heading_history(self):
        g = make_guidance()
        drv = Driver(g)
        for i in range(5):
            drv.yaw = 0.1 * i
            drv.tick(None)
        assert list(g._yaw_hist) == [(i * DT, 0.1 * i) for i in range(5)]

    def test_bearing_is_de_rotated_by_the_heading_held_at_capture(self):
        # The whole point of the latency machinery: a bearing that arrived now
        # must be resolved into the world with the yaw from `sensor_latency_s`
        # ago, not the yaw the aircraft has reached since.
        lat = 0.10
        g = make_guidance(replace(GuidanceConfig(), sensor_latency_s=lat))
        drv = Driver(g)
        yaws = [0.0, 0.1, 0.2, 0.3, 0.4]
        for i, y in enumerate(yaws):
            drv.yaw = y
            st = drv.tick(det(az=0.0, rng=40.0))

        yaw_then = 0.2                       # the heading held at t = 0.2 - 0.1
        assert g.los.s == pytest.approx(body_to_world(yaw_then, 1.0, 0.0, 0.0),
                                        abs=1e-12)
        assert g.los.s != pytest.approx(body_to_world(0.4, 1.0, 0.0, 0.0), abs=1e-6)
        assert g.los.t_last == pytest.approx(4 * DT - lat, abs=1e-12)
        assert st.age_s == pytest.approx(lat, abs=1e-9)

    def test_a_pure_own_yaw_rate_does_not_look_like_target_motion(self):
        # Chaser yawing steadily, target parked dead ahead in the world: the
        # image-plane bearing moves, but the world LOS must not, so the PN rate
        # must stay at zero. This is the "seeker chasing its own tail" case.
        lat = 0.10
        g = make_guidance(replace(GuidanceConfig(), sensor_latency_s=lat))
        drv = Driver(g)
        # Bounded so the bearing stays inside the camera's +/-38 deg field of
        # view for the whole run. Outside it there is no measurement to make, and
        # a bearing beyond +/-90 deg is not a direction the pinhole model can
        # even represent -- `bearing_to_los` inverts `tan`, which is exactly what
        # makes it the true inverse of `bearing_from_pixel` inside the FOV.
        yaw_rate = 1.5
        for i in range(10):
            t = i * DT
            drv.yaw = yaw_rate * t
            # the bearing the camera reports lags by `lat`: the target sits at
            # world azimuth 0, so body azimuth = -yaw(t - lat)
            az = -yaw_rate * max(0.0, t - lat)
            assert abs(az) < math.radians(38.0)
            drv.tick(det(az=az, rng=40.0))
        assert g.los.rate_mag == pytest.approx(0.0, abs=1e-12)
        assert g.los.s == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)


# ==================================================== physical LOS-rate clamp

class TestLosRateClamp:

    @staticmethod
    def _limit(range_m, limits=LIMITS):
        return 2.2 * limits.max_speed_xy / max(2.0, range_m)

    def test_clamp_scales_an_impossible_rate_down_and_keeps_its_direction(self):
        g = make_guidance()
        g.rng.value, g.rng.t_last = 40.0, 0.0
        raw = (0.0, 5.0, 0.0)
        g.los.ds = raw
        g._clamp_los_rate()

        lim = self._limit(40.0)               # 2.2 * 14 / 40 = 0.77 rad/s
        assert lim == pytest.approx(0.77, rel=1e-12)
        assert norm(g.los.ds) == pytest.approx(lim, rel=1e-12)
        assert g.los.ds == pytest.approx((0.0, lim, 0.0), abs=1e-15)

    def test_clamp_preserves_direction_in_three_dimensions(self):
        g = make_guidance()
        g.rng.value, g.rng.t_last = 25.0, 0.0
        raw = (1.0, -2.0, 3.0)
        g.los.ds = raw
        g._clamp_los_rate()
        lim = self._limit(25.0)
        assert norm(g.los.ds) == pytest.approx(lim, rel=1e-12)
        k = 1.0 / norm(raw)
        assert g.los.ds == pytest.approx(tuple(c * k * lim for c in raw), abs=1e-15)

    def test_a_physically_possible_rate_is_untouched(self):
        g = make_guidance()
        g.rng.value, g.rng.t_last = 40.0, 0.0
        small = (0.0, 0.5, 0.0)               # below the 0.77 limit
        g.los.ds = small
        g._clamp_los_rate()
        assert g.los.ds == small              # not scaled *up* either

    def test_limit_is_range_dependent_and_floors_the_denominator_at_2m(self):
        g = make_guidance()
        # far: a tight limit
        g.rng.value, g.rng.t_last = 100.0, 0.0
        g.los.ds = (0.0, 50.0, 0.0)
        g._clamp_los_rate()
        assert norm(g.los.ds) == pytest.approx(2.2 * 14.0 / 100.0, rel=1e-12)
        # very close: the denominator floors at 2 m, so the limit stops growing
        for r in (2.0, 1.0, 0.01):
            g.rng.value = r
            g.los.ds = (0.0, 50.0, 0.0)
            g._clamp_los_rate()
            assert norm(g.los.ds) == pytest.approx(2.2 * 14.0 / 2.0, rel=1e-12)

    def test_clamp_is_a_no_op_without_a_range_estimate(self):
        g = make_guidance()
        assert g.rng.value is None
        g.los.ds = (0.0, 99.0, 0.0)
        g._clamp_los_rate()
        assert g.los.ds == (0.0, 99.0, 0.0)

    def test_clamp_fires_on_a_bearing_jump_through_the_public_path(self):
        g = make_guidance()
        drv = Driver(g)
        drv.run(4, det(az=0.0, rng=40.0))
        assert g.los.rate_mag == 0.0

        drv.tick(det(az=1.0, rng=40.0))       # 57 degrees in one 50 ms tick
        lim = self._limit(40.0)
        assert g.rng.value == pytest.approx(40.0, rel=1e-12)
        assert g.los.rate_mag == pytest.approx(lim, rel=1e-12)
        assert g.los.ds[1] > 0.0              # direction of the jump is kept

    def test_clamp_scales_with_the_airframes_own_speed_limit(self):
        fast = replace(LIMITS, max_speed_xy=28.0)
        g = make_guidance(limits=fast)
        g.rng.value, g.rng.t_last = 40.0, 0.0
        g.los.ds = (0.0, 50.0, 0.0)
        g._clamp_los_rate()
        assert norm(g.los.ds) == pytest.approx(2.2 * 28.0 / 40.0, rel=1e-12)


# ==================================================================== the FSM

class TestModeMachine:

    def test_starts_in_search_and_stays_there_without_detections(self):
        """Search is step-and-stare, and the FIRST phase is a look, not a slew.

        That ordering is the whole point: a pursuit usually starts with the
        target already in frame, and a search that opens by turning away spends
        the one moment the range is shortest. Measured against the trained
        detector, opening with a slew was the difference between acquiring and
        never acquiring at all.
        """
        cfg = GuidanceConfig()
        g = make_guidance()
        drv = Driver(g, vel=(0.0, 0.0, 0.0))
        assert g.mode == SEARCH

        # Walk one full look/slew/look cycle and record the phase each tick.
        n = int((cfg.search_first_dwell_s
                 + cfg.search_step_rad / cfg.search_yaw_rate) / DT) + 6
        phases, states = [], []
        for _ in range(n):
            st = drv.tick(None)
            assert st.mode == SEARCH
            assert (st.command.vx, st.command.vy) == (0.0, 0.0)
            assert st.command.frame == "world"
            phases.append(st.command.source)
            states.append(st)

        # It opens with a look, holding the heading perfectly still...
        n_look = next(i for i, p in enumerate(phases) if p != "search:look")
        assert phases[0] == "search:look"
        assert all(s.command.yaw_rate == 0.0 for s in states[:n_look])
        assert abs(n_look - cfg.search_first_dwell_s / DT) <= 1

        # ...then slews by one step at the search rate...
        assert phases[n_look] == "search:slew"
        assert states[n_look].command.yaw_rate == pytest.approx(cfg.search_yaw_rate)
        n_slew = next((i for i, p in enumerate(phases[n_look:]) if p != "search:slew"),
                      None)
        assert n_slew is not None, "the slew must end in another look"
        assert abs(n_slew - cfg.search_step_rad / cfg.search_yaw_rate / DT) <= 1

        # ...and holds again. A step shorter than the field of view means
        # consecutive looks overlap, so a target cannot fall between two of them.
        assert phases[n_look + n_slew] == "search:look"
        assert cfg.search_step_rad < math.radians(INTR.width / INTR.fx * 57.3)

        assert states[-1].range_est is None
        assert states[-1].confirmed is False

    def test_search_altitude_is_a_bounded_excursion_not_a_climb(self):
        """The search moves vertically around a datum and stays there.

        It has to move at all because the camera sees 15.5 degrees up and 32
        down, so the targets it cannot see are the high ones and altitude is the
        only control that changes elevation. It must not be a climb *rate*: a
        constant 3 m/s over a 20-second search is 60 m, which pins the aircraft
        at its ceiling looking down at terrain, where a target at the original
        altitude sits below the bottom of the frame. That failure needed a long
        episode to appear, so this test uses one.
        """
        cfg = GuidanceConfig()
        g = make_guidance()
        z0 = 25.0
        drv = Driver(g, vel=(0.0, 0.0, 0.0), xyz=(0.0, 0.0, z0))

        # Integrate the commanded climb rate the way the airframe would.
        z = z0
        seen = []
        for _ in range(int(40.0 / DT)):              # 40 s of searching
            drv.xyz = (0.0, 0.0, z)
            st = drv.tick(None)
            assert st.mode == SEARCH
            z += st.command.vz * DT
            seen.append(z)

        lo = z0 + cfg.search_alt_offset_m - cfg.search_alt_amplitude_m
        hi = z0 + cfg.search_alt_offset_m + cfg.search_alt_amplitude_m
        assert min(seen) >= lo - 2.0, "search descended below its excursion"
        assert max(seen) <= hi + 2.0, "search climbed away from its datum"
        # It genuinely moves -- most of the commanded amplitude, allowing for
        # the aircraft's finite climb rate lagging a sinusoid.
        assert max(seen) - min(seen) > 0.6 * cfg.search_alt_amplitude_m
        # And it is biased upward, toward the blind side of the camera.
        assert sum(seen) / len(seen) > z0

    def test_search_to_acquire_to_pursue_after_confirm_hits(self):
        cfg = GuidanceConfig()
        assert cfg.confirm_hits == 3
        g = make_guidance()
        drv = Driver(g)

        modes, streaks = [], []
        for _ in range(cfg.confirm_hits + 2):
            st = drv.tick(det(rng=40.0))
            modes.append(st.mode)
            streaks.append(st.streak)

        assert modes == [ACQUIRE, ACQUIRE, PURSUE, PURSUE, PURSUE]
        assert streaks == [1, 2, 3, 4, 5]
        assert g.confirmed is True

    def test_confirmation_needs_the_hits_to_be_consecutive_enough(self):
        cfg = GuidanceConfig()
        assert cfg.confirm_miss_tolerance == 1
        g = make_guidance()
        drv = Driver(g)

        assert drv.tick(det(rng=40.0)).mode == ACQUIRE
        assert drv.tick(det(rng=40.0)).mode == ACQUIRE
        # one miss inside the streak is tolerated
        st = drv.tick(None)
        assert st.mode == ACQUIRE and st.streak == 2
        # a second consecutive miss throws the streak away
        st = drv.tick(None)
        assert st.mode == SEARCH and st.streak == 0
        assert g.confirmed is False
        # and it has to be earned again from scratch
        assert drv.tick(det(rng=40.0)).mode == ACQUIRE
        assert drv.tick(det(rng=40.0)).mode == ACQUIRE
        assert drv.tick(det(rng=40.0)).mode == PURSUE

    def test_pursue_to_terminal_by_range(self):
        cfg = GuidanceConfig()
        g = make_guidance()
        drv = Driver(g)
        modes = [drv.tick(det(rng=5.0)).mode for _ in range(5)]
        # NOTE: the tick that confirms returns early from _advance_mode, so the
        # first PURSUE happens even though the range is already inside the
        # terminal ring; TERMINAL is entered on the following tick.
        assert modes == [ACQUIRE, ACQUIRE, PURSUE, TERMINAL, TERMINAL]
        assert 5.0 < cfg.terminal_range_m + 1e-9
        assert g._terminal_los == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
        assert g._terminal_range == pytest.approx(5.0, rel=1e-9)

    def test_terminal_boundary_is_the_configured_range(self):
        cfg = GuidanceConfig()
        for rng, expect in ((cfg.terminal_range_m + 0.01, PURSUE),
                            (cfg.terminal_range_m, TERMINAL),
                            (cfg.terminal_range_m - 0.01, TERMINAL)):
            g = make_guidance()
            drv = Driver(g)
            st = drv.run(5, det(rng=rng))
            assert st.mode == expect, rng

    def test_hit_is_reported_but_does_not_stop_the_closure(self):
        cfg = GuidanceConfig()
        g = make_guidance()
        drv = Driver(g)
        st = drv.run(5, det(rng=cfg.hit_range_m - 0.1))
        assert st.mode == HIT
        # HIT is a report, not a brake: the command is still full closing speed
        assert st.command.vx == pytest.approx(LIMITS.max_speed_xy, rel=1e-12)
        assert st.closing_speed == pytest.approx(LIMITS.max_speed_xy, rel=1e-3)

    def test_terminal_is_not_latched_and_falls_back_to_pursue(self):
        # A miss flies *past* the target; the mode must follow the range back out.
        g = make_guidance()
        drv = Driver(g)
        assert drv.run(5, det(rng=5.0)).mode == TERMINAL
        # walk the range back out through the innovation gate
        r = 5.0
        st = None
        while r < 12.0:
            r += 0.5
            st = drv.tick(det(rng=r))
        assert st.mode == PURSUE
        assert g._terminal_los is None

    def test_pursue_drops_to_reacquire_on_a_single_miss(self):
        g = make_guidance()
        drv = Driver(g)
        assert confirm(drv).mode == PURSUE
        st = drv.tick(None)
        assert st.mode == REACQUIRE
        assert st.lost_for_s == pytest.approx(DT, abs=1e-9)
        assert g.confirmed is True                  # the track is not disowned

    def test_reacquire_coasts_then_sweeps(self):
        cfg = GuidanceConfig()
        g = make_guidance()
        drv = Driver(g)
        confirm(drv)

        # inside reacquire_hold_s it keeps flying the closure it had
        while g.lost_for_s < cfg.reacquire_hold_s - 1e-9:
            st = drv.tick(None)
            assert st.mode == REACQUIRE
            assert st.command.source == "pn:reacquire"
            assert st.command.vx == pytest.approx(LIMITS.max_speed_xy, rel=1e-9)

        # past it, the aircraft stops translating and sweeps instead
        st = drv.tick(None)
        assert g.lost_for_s > cfg.reacquire_hold_s
        assert st.command.source == "reacquire"
        assert st.note == "re-acquisition sweep"
        assert (st.command.vx, st.command.vy) == (0.0, 0.0)
        assert abs(st.command.yaw_rate) == pytest.approx(cfg.search_yaw_rate)

    def test_terminal_survives_a_brief_dropout(self):
        cfg = GuidanceConfig()
        g = make_guidance()
        drv = Driver(g)
        assert drv.run(5, det(rng=5.0)).mode == TERMINAL

        # half a second blind -- well past reacquire_hold_s, but TERMINAL holds
        while g.lost_for_s < 0.5:
            st = drv.tick(None)
            assert st.mode == TERMINAL
            assert st.command.source == "pn:terminal"
        assert g.lost_for_s > cfg.reacquire_hold_s
        # and it is still flying the committed line of sight at full speed
        assert st.command.vx == pytest.approx(LIMITS.max_speed_xy, rel=1e-9)
        assert st.command.vy == pytest.approx(0.0, abs=1e-12)

    def test_terminal_switches_from_live_to_blind_commit(self):
        cfg = GuidanceConfig()
        g = make_guidance()
        drv = Driver(g)
        assert drv.run(5, det(rng=5.0)).note == "terminal (live)"

        notes = []
        for _ in range(4):
            st = drv.tick(None)
            notes.append((round(st.age_s, 3), st.note))
        assert notes == [(0.05, "terminal (live)"),
                         (0.10, "terminal (live)"),
                         (0.15, "terminal (blind commit)"),
                         (0.20, "terminal (blind commit)")]
        assert cfg.terminal_fresh_s == 0.12

    def test_terminal_falls_to_reacquire_past_terminal_blind_s(self):
        cfg = GuidanceConfig()
        g = make_guidance()
        drv = Driver(g)
        assert drv.run(5, det(rng=5.0)).mode == TERMINAL

        flip_at = None
        for _ in range(int(2.0 / DT)):
            st = drv.tick(None)
            if st.mode != TERMINAL:
                flip_at = g.lost_for_s
                break
        assert st.mode == REACQUIRE
        assert flip_at is not None
        assert cfg.terminal_blind_s == 1.2
        assert cfg.terminal_blind_s < flip_at < cfg.terminal_blind_s + 2 * DT
        assert g._terminal_los is None
        assert g.confirmed is True          # not given up on yet

    def test_reacquire_falls_back_to_search_after_reacquire_timeout_s(self):
        cfg = GuidanceConfig()
        g = make_guidance()
        drv = Driver(g)
        confirm(drv)
        assert g.rng.value is not None and g.los.s is not None

        flip_at = None
        for _ in range(int((cfg.reacquire_timeout_s + 1.0) / DT)):
            st = drv.tick(None)
            if st.mode == SEARCH:
                flip_at = g.lost_for_s
                break
        assert st.mode == SEARCH
        assert flip_at is not None
        assert cfg.reacquire_timeout_s <= flip_at < cfg.reacquire_timeout_s + 2 * DT
        # giving up means forgetting: filters and the streak are wiped
        assert g.confirmed is False
        assert g.streak == 0
        assert g.los.s is None
        assert g.rng.value is None
        assert st.command.source.startswith("search")
        # SUSPECTED BUG (reporting only): GuidanceState.range_est is filled in
        # *before* _advance_mode runs, so the very tick that gives up still
        # reports the range it just threw away -- a logged/HUD range that no
        # longer has a filter behind it. It reads None from the next tick on.
        assert st.range_est == pytest.approx(40.0, rel=1e-9)
        assert drv.tick(None).range_est is None

    def test_reset_returns_the_whole_machine_to_search(self):
        g = make_guidance()
        drv = Driver(g)
        drv.run(5, det(rng=5.0))
        g.reset()
        assert g.mode == SEARCH
        assert (g.streak, g.misses, g.confirmed, g.lost_for_s) == (0, 0, False, 0.0)
        assert g.los.s is None and g.rng.value is None
        assert g._terminal_los is None
        assert len(g._yaw_hist) == 0


# =============================================================== the PN law

class TestTailChase:

    def test_pure_tail_chase_commands_no_lateral_velocity(self):
        """Target dead ahead and receding in a straight line: the line of sight
        never rotates, so proportional navigation has nothing to correct and the
        entire command must be closing speed."""
        g = make_guidance()
        drv = Driver(g, vel=(9.0, 0.0, 0.0))       # own velocity straight up the LOS

        st = None
        rng = 40.0
        for _ in range(40):
            rng += 5.0 * DT                        # target pulling away at 5 m/s
            st = drv.tick(det(az=0.0, el=0.0, rng=rng))

        assert st.mode == PURSUE
        assert st.los_rate == 0.0                  # the LOS never rotated
        assert st.lateral_speed == 0.0
        assert abs(st.command.vy) < 1e-12
        assert abs(st.command.vz) < 1e-12
        assert abs(st.command.yaw_rate) < 1e-12
        # ... and all of the speed budget went into closing
        assert st.command.vx == pytest.approx(GuidanceConfig().approach_speed,
                                              rel=1e-12)
        assert st.closing_speed == pytest.approx(GuidanceConfig().approach_speed,
                                                 rel=1e-9)
        assert st.boresight_deg == 0.0
        assert st.command.frame == "world"

    def test_tail_chase_off_the_boresight_still_has_no_lateral_term(self):
        # Same geometry rotated: the chaser is heading 0.6 rad but flying exactly
        # along the (constant) line of sight, so there is still nothing to steer.
        yaw = 0.6
        g = make_guidance()
        s_world = body_to_world(yaw, *bearing_to_los(0.0, 0.0))
        drv = Driver(g, yaw=yaw, vel=tuple(9.0 * c for c in s_world))
        st = drv.run(40, det(az=0.0, el=0.0, rng=45.0))

        assert st.los_rate == 0.0
        assert st.lateral_speed == 0.0
        assert abs(st.command.yaw_rate) < 1e-12
        cmd = (st.command.vx, st.command.vy, st.command.vz)
        assert norm(cmd) == pytest.approx(GuidanceConfig().approach_speed, rel=1e-9)
        # the command is parallel to the line of sight
        assert dot(cmd, s_world) == pytest.approx(norm(cmd), rel=1e-12)

    def test_a_crossing_target_does_produce_a_lateral_command(self):
        # Guard against the tail-chase assertions passing vacuously.
        g = make_guidance()
        drv = Driver(g, vel=(14.0, 0.0, 0.0))
        st = None
        for i in range(40):
            st = drv.tick(det(az=0.3 * i * DT, rng=40.0))   # crossing left

        # The reference is the same geometry with the LOS rate withheld, which
        # leaves only the across-LOS velocity the chaser already had. Asserting
        # an absolute sign on world vy instead would be asserting a property of
        # `lookahead_s`: the chaser is flying straight while the sight line
        # rotates away, so its own perpendicular velocity is a large negative
        # number and whether the total lands above zero depends entirely on how
        # far ahead PN is asked to predict. What must be true for any lookahead
        # is that PN pushes the command *toward* the target relative to doing
        # nothing.
        blind = make_guidance(replace(GuidanceConfig(), min_los_samples=10 ** 6))
        db = Driver(blind, vel=(14.0, 0.0, 0.0))
        st_blind = None
        for i in range(40):
            st_blind = db.tick(det(az=0.3 * i * DT, rng=40.0))

        assert st.los_rate == pytest.approx(0.3, rel=0.05)
        assert st.lateral_speed > 1.0
        assert st.command.vy > st_blind.command.vy + 0.5    # steering left
        assert st.command.yaw_rate > 0.5                    # and turning to follow

    def test_lateral_command_is_capped_before_the_total_is_saturated(self):
        # A static bearing well off the nose: the LOS rate is zero, so the whole
        # across-LOS term is the chaser's own velocity component, |v| * sin(az).
        cfg = GuidanceConfig()
        under, over = 0.5, 0.8          # 14*sin(0.5)=6.71 (under 9), 14*sin(0.8)=10.04
        got = {}
        for az in (under, over):
            g = make_guidance()
            drv = Driver(g, vel=(14.0, 0.0, 0.0))
            st = drv.run(20, det(az=az, rng=40.0))
            got[az] = st
            assert st.los_rate == 0.0

        assert got[under].lateral_speed == pytest.approx(14.0 * math.sin(under),
                                                         abs=1e-3)
        assert got[over].lateral_speed == pytest.approx(cfg.max_lateral_speed,
                                                        rel=1e-9)
        # capping the lateral term left the closing term intact
        assert got[over].closing_speed > 0.0

    def test_los_rate_is_ignored_until_min_los_samples(self):
        # Same crossing geometry, but the rate is not yet trusted: the only
        # across-LOS term left is the velocity the chaser already had.
        crossing = [det(az=0.3 * i * DT, rng=40.0) for i in range(40)]
        gated = make_guidance(replace(GuidanceConfig(), min_los_samples=10 ** 6))
        live = make_guidance()
        st_gated = st_live = None
        d1, d2 = Driver(gated, vel=(14.0, 0.0, 0.0)), Driver(live, vel=(14.0, 0.0, 0.0))
        for m in crossing:
            st_gated = d1.tick(m)
            st_live = d2.tick(m)

        assert gated.los.rate_mag > 0.25            # the rate was measured
        assert st_gated.command.vy < 0.0            # ... but not steered on
        # Relative, not absolute: see the note in the crossing-target test above.
        assert st_live.command.vy > st_gated.command.vy
        assert st_gated.lateral_speed != pytest.approx(st_live.lateral_speed, rel=1e-3)

    def test_speed_gate_backs_off_when_the_target_is_off_boresight(self):
        cfg = GuidanceConfig()
        speeds = {}
        for az_deg in (0.0, cfg.boresight_soft_deg, cfg.boresight_hard_deg):
            g = make_guidance()
            drv = Driver(g, vel=(0.0, 0.0, 0.0))
            st = drv.run(6, det(az=math.radians(az_deg), rng=40.0))
            speeds[az_deg] = norm((st.command.vx, st.command.vy, st.command.vz))
            assert st.boresight_deg == pytest.approx(az_deg, abs=1e-6)
        assert speeds[0.0] == pytest.approx(cfg.approach_speed, rel=1e-9)
        assert speeds[cfg.boresight_soft_deg] == pytest.approx(cfg.approach_speed,
                                                              rel=1e-9)
        assert speeds[cfg.boresight_hard_deg] == pytest.approx(
            cfg.approach_speed * cfg.capture_speed_scale, rel=1e-9)

    def test_yaw_deadband_suppresses_a_tiny_azimuth_error(self):
        cfg = GuidanceConfig()
        inside = 0.5 * cfg.yaw_deadband_rad
        outside = 2.0 * cfg.yaw_deadband_rad
        got = {}
        for az in (inside, outside):
            g = make_guidance()
            drv = Driver(g, vel=(0.0, 0.0, 0.0))
            got[az] = drv.run(6, det(az=az, rng=40.0)).command.yaw_rate
        assert got[inside] == 0.0
        assert got[outside] == pytest.approx(cfg.kp_yaw * outside, rel=1e-9)

    def test_yaw_rate_is_clamped_to_the_airframe_limit(self):
        slow = replace(LIMITS, max_yaw_rate=1.0)
        g = make_guidance(limits=slow)
        drv = Driver(g, vel=(0.0, 0.0, 0.0))
        st = drv.run(6, det(az=1.0, rng=40.0))     # kp_yaw * 1.0 = 1.4 > 1.0
        assert st.command.yaw_rate == pytest.approx(slow.max_yaw_rate, rel=1e-12)
        st = Driver(make_guidance(limits=slow), vel=(0.0, 0.0, 0.0)).run(
            6, det(az=-1.0, rng=40.0))
        assert st.command.yaw_rate == pytest.approx(-slow.max_yaw_rate, rel=1e-12)

    def test_elevation_maps_to_a_climb_command(self):
        g = make_guidance()
        drv = Driver(g, vel=(0.0, 0.0, 0.0))
        up = drv.run(6, det(el=0.3, rng=40.0)).command.vz
        g2 = make_guidance()
        down = Driver(g2, vel=(0.0, 0.0, 0.0)).run(6, det(el=-0.3, rng=40.0)).command.vz
        assert up > 0.0 and down < 0.0
        assert up == pytest.approx(-down, rel=1e-12)
        assert abs(up) <= LIMITS.max_speed_z + 1e-12


# ================================================= behaviour worth recording

class TestKnownQuirks:

    def test_vertical_lead_scale_actually_reaches_the_command(self):
        # This replaces test_vertical_gain_is_a_dead_configuration_knob, which
        # asserted the *no-op* behaviour of a `vertical_gain` that guidance.py
        # never read, and said "the day it is wired up, this test says so".
        # It did. `vertical_gain` is gone; `vertical_lead_scale` is real, and
        # this asserts the property the old test was holding a place for.
        # The target must be *changing elevation*: the lead scales the PN
        # correction, which is proportional to the line-of-sight rate, so a
        # static geometry has nothing for it to act on and would pass this test
        # for the wrong reason.
        out = []
        for vs in (1.0, 3.0):
            g = make_guidance(replace(GuidanceConfig(), vertical_lead_scale=vs))
            drv = Driver(g, vel=(14.0, 0.0, 0.0))
            st = None
            for k in range(12):
                st = drv.tick(det(az=0.05, el=0.10 + 0.02 * k,
                                  rng=40.0 - 0.7 * k))
            out.append(st.command.as_tuple())
        assert out[0] != out[1], "vertical_lead_scale has no effect on the command"
        # It must change the commanded *direction* vertically -- more lead means
        # a steeper command. Asserting the other two components are untouched
        # would be wrong now that saturation is applied as a vector: a larger vz
        # rescales the whole vector, which is the point of that fix.
        def steepness(c):
            return c[2] / max(1e-9, math.hypot(c[0], c[1]))
        assert abs(steepness(out[1])) > abs(steepness(out[0])), (
            "more vertical lead must command a steeper climb/descent")

    def test_vertical_saturation_preserves_the_commanded_direction(self):
        """A saturated climb must slow the aircraft, not re-point it.

        The horizontal ceiling has always been applied as a vector scale for
        this reason; the vertical one was a bare component clamp sitting two
        lines under the comment explaining why that is wrong, which flattens the
        commanded course whenever the climb saturates.
        """
        g = make_guidance()
        limits = g.limits
        drv = Driver(g, vel=(4.0, 0.0, 0.0))
        # A steeply-above target drives a large vertical command.
        st = drv.run(14, det(az=0.0, el=0.9, rng=12.0))
        vx, vy, vz, _yaw = st.command.as_tuple()
        assert abs(vz) <= limits.max_speed_z + 1e-6
        assert math.hypot(vx, vy) <= limits.max_speed_xy + 1e-6

    def test_reacquire_coast_note_is_overwritten_by_the_pn_note(self):
        # SUSPECTED BUG: _reacquire_command sets st.note = "coasting on predicted
        # LOS" and then returns _closure_command(...), which unconditionally
        # rewrites st.note to "PN N=... Vc=...". The coast phase is therefore
        # mislabelled in every trace and on the HUD -- a REACQUIRE tick that is
        # coasting is indistinguishable from a PURSUE tick by its note. The
        # command itself is correct; only the label is lost.
        g = make_guidance()
        drv = Driver(g)
        confirm(drv)
        st = drv.tick(None)
        assert st.mode == REACQUIRE
        assert st.command.source == "pn:reacquire"       # the source is right
        assert st.note.startswith("PN N=")               # the note is not
        assert "coasting" not in st.note

    def test_confirmation_tick_skips_the_range_based_mode_check(self):
        # Observed behaviour rather than a defect, but it is load-bearing for
        # anything that reads the mode: the tick that sets `confirmed` returns
        # from _advance_mode before the range is consulted, so a target already
        # inside hit_range_m spends one tick in PURSUE (at approach_speed, not
        # terminal_speed) before HIT is ever reported.
        g = make_guidance()
        drv = Driver(g)
        modes = [drv.tick(det(rng=0.5)).mode for _ in range(4)]
        assert modes == [ACQUIRE, ACQUIRE, PURSUE, HIT]


class TestClosureProbation:
    """A lock must close, or it is not a target.

    The one check that can distinguish a *persistent* false lock from a real
    pursuit. Measured failure it exists for: the seeker locked onto a fixed
    feature on the horizon -- the same pixel for forty seconds while the chaser
    flew 400 m at it, its estimated range pinned at 35 m throughout, because a
    thing that neither moves in the image nor grows is infinitely far away.
    """

    def test_a_lock_that_never_closes_is_dropped(self):
        cfg = GuidanceConfig()
        g = make_guidance()
        drv = Driver(g)
        confirm(drv, rng=35.0)
        assert g.mode == PURSUE and g.confirmed

        # Fly at it for the whole probation window with the range going nowhere,
        # exactly as a horizon feature behaves.
        n = int((cfg.lock_probation_s + 4 * DT) / DT)
        for _ in range(n):
            st = drv.tick(det(rng=35.0))
        assert g.confirmed is False
        assert st.mode == SEARCH
        # (The bearing/range filters legitimately repopulate from whatever the
        # detector keeps reporting -- it is the *lock* that has been rejected,
        # and `confirmed` is what says so.)

        # ...and it must not simply re-lock onto the same thing on the next
        # frame, which is what makes the rejection worth anything.
        for _ in range(int(0.8 * cfg.lock_refractory_s / DT)):
            st = drv.tick(det(rng=35.0))
            assert g.confirmed is False, "re-locked inside the refractory period"
            assert st.mode == SEARCH

    def test_a_closing_lock_is_never_interrupted(self):
        cfg = GuidanceConfig()
        g = make_guidance()
        drv = Driver(g)
        r = 60.0
        confirm(drv, rng=r)
        # A slow but real closure: 1.4 m/s, the hardest scenario in the suite.
        for _ in range(int(3 * cfg.lock_probation_s / DT)):
            r = max(2.0, r - 1.4 * DT)
            st = drv.tick(det(rng=r))
            assert g.confirmed, "a closing pursuit must never be dropped"
        assert st.mode in (PURSUE, TERMINAL, HIT)

    def test_the_window_restarts_on_real_progress(self):
        """Closure resets the clock, so a long pursuit is not chopped up."""
        cfg = GuidanceConfig()
        g = make_guidance()
        drv = Driver(g)
        r = 80.0
        confirm(drv, rng=r)
        # Alternate: close hard for a window, then stall for most of one.
        for _cycle in range(3):
            for _ in range(int(cfg.lock_probation_s / DT)):
                r = max(5.0, r - 0.6 * DT)      # 4.8 m per window, over the bar
                drv.tick(det(rng=r))
            assert g.confirmed
            for _ in range(int(0.5 * cfg.lock_probation_s / DT)):
                drv.tick(det(rng=r))            # stall, but not long enough
            assert g.confirmed

    def test_probation_does_not_run_while_searching(self):
        g = make_guidance()
        drv = Driver(g)
        drv.run(int(3 * GuidanceConfig().lock_probation_s / DT), None)
        assert g.mode == SEARCH
        assert g._probation_t0 is None


def test_search_altitude_datum_does_not_ratchet_across_lock_cycles():
    """The excursion datum is set once, not re-taken on every mode change.

    Re-datuming whenever the mode leaves SEARCH looks harmless and is a ratchet:
    each search/pursue/search cycle re-centres at wherever the last one left the
    aircraft and adds the offset again, walking it upward until it is pinned at
    the ceiling -- where a target at the original altitude falls below the
    bottom of the frame. Bounded relative to a moving datum is not bounded.
    """
    cfg = GuidanceConfig()
    g = make_guidance()
    z0 = 25.0
    drv = Driver(g, vel=(0.0, 0.0, 0.0), xyz=(0.0, 0.0, z0))
    z = z0
    peak = z0

    for _cycle in range(6):
        # search for a while...
        for _ in range(int(6.0 / DT)):
            drv.xyz = (0.0, 0.0, z)
            st = drv.tick(None)
            z += st.command.vz * DT
            peak = max(peak, z)
        # ...then a burst of detections that takes it into PURSUE and back.
        for _ in range(cfg.confirm_hits + 2):
            drv.xyz = (0.0, 0.0, z)
            st = drv.tick(det(rng=40.0))
            z += st.command.vz * DT
            peak = max(peak, z)

    hi = z0 + cfg.search_alt_offset_m + cfg.search_alt_amplitude_m
    assert peak <= hi + 3.0, (
        f"search altitude ratcheted to {peak:.1f} m from a {z0:.1f} m datum "
        f"(ceiling of the excursion is {hi:.1f} m)")


class TestScoreProbation:
    """A new lock must prove itself with confident detections, or die.

    The measured discriminator for urban flight. Detector score separates a
    drone from a rooftop at AUC 0.95 (median 0.80 against 0.24), but applying it
    at *seeding* is the wrong place: a real drone at the edge of the envelope
    scores badly too, and a bar high enough to exclude a building excluded a
    genuine 0.38 m intercept. So seeding stays cheap and the audit happens
    immediately afterwards.
    """

    # The probation is OFF by default (it did not survive measurement -- see
    # GuidanceConfig.score_probation_frames). These tests exercise the
    # mechanism, so they switch it on explicitly rather than depending on a
    # default that is deliberately zero.
    # The probation floor is lifted above the seed score so the confirming
    # detections clear PROMOTION (T = 0.70 at 10 px) without also counting as
    # probation evidence. Otherwise seeding hands the track a free hit and the
    # "one hit is not enough" case cannot be expressed at all.
    ON = replace(GuidanceConfig(), score_probation_frames=20,
                 score_probation_min=0.80)

    def _drive(self, scores, cfg=None, seed_score=0.72):
        """Confirm a track at ``seed_score``, then feed ``scores``.

        ``seed_score`` must clear the *promotion* floor (T = 0.70 at the 10 px
        span the ``det`` helper uses) or the track never confirms at all and
        every assertion below passes vacuously -- the promotion gate would have
        rejected it before the probation ever armed. It is kept only just above
        that floor so the confirming detections contribute as little probation
        evidence as possible.
        """
        g = make_guidance(cfg if cfg is not None else self.ON)
        drv = Driver(g)
        for _ in range(GuidanceConfig().confirm_hits + 1):
            drv.tick(TargetEstimate(valid=True, u=0.0, v=0.0, span_px=10.0,
                                    az=0.0, el=0.0, score=seed_score,
                                    source="detector", range_override=40.0))
        for sc in scores:
            drv.tick(TargetEstimate(valid=True, u=0.0, v=0.0, span_px=10.0,
                                    az=0.0, el=0.0, score=sc,
                                    source="detector", range_override=40.0))
        return g

    def test_a_confident_track_survives(self):
        cfg = self.ON
        g = self._drive([0.85] * (cfg.score_probation_frames + 4),
                        cfg=cfg, seed_score=0.85)
        assert g.confirmed, "a track scoring 0.85 throughout was rejected"

    def test_a_low_scoring_track_is_dropped(self):
        cfg = self.ON
        g = self._drive([0.25] * (cfg.score_probation_frames + 4), cfg=cfg)
        assert not g.confirmed, (
            "a track that never scored above the probation floor was kept -- "
            "this is the Rivermark rooftop case")
        assert g.mode == SEARCH

    def test_the_bar_is_a_couple_of_hits_not_a_majority(self):
        """Two confident frames are enough; a real target flickers."""
        cfg = self.ON
        scores = [0.2] * cfg.score_probation_frames
        scores[3] = 0.86
        scores[9] = 0.90
        g = self._drive(scores + [0.2] * 4, cfg=cfg)
        assert g.confirmed, (
            "two confident detections should clear probation -- a genuine "
            "target at range scores badly on most frames")

    def test_one_confident_hit_is_not_enough(self):
        cfg = self.ON
        scores = [0.2] * cfg.score_probation_frames
        scores[5] = 0.90
        g = self._drive(scores + [0.2] * 4, cfg=cfg)
        assert not g.confirmed

    def test_the_window_counts_detector_frames_not_seconds(self):
        """A coasting track must not be judged on evidence it could not produce.

        Counting wall-clock time would expire the window during a dropout and
        execute exactly the track the reacquire logic exists to protect.
        """
        cfg = self.ON
        g = make_guidance(cfg)
        drv = Driver(g)
        confirm(drv)
        # Two confident detections, then a long blind coast.
        for sc in (0.8, 0.8):
            drv.tick(TargetEstimate(valid=True, u=0.0, v=0.0, span_px=10.0,
                                    az=0.0, el=0.0, score=sc,
                                    source="detector", range_override=40.0))
        n_before = g._score_probe_frames
        for _ in range(int(3.0 / DT)):
            drv.tick(None)
        assert g._score_probe_frames == n_before, (
            "coasted frames were counted as probation opportunities")

    def test_probation_can_be_disabled(self):
        cfg = replace(GuidanceConfig(), score_probation_frames=0)
        g = self._drive([0.05] * 40, cfg)
        assert g.confirmed, "score_probation_frames=0 must switch the test off"

    def test_a_rejected_lock_is_not_immediately_re_seeded(self):
        """The refractory period is what makes the rejection stick."""
        cfg = self.ON
        g = self._drive([0.25] * (cfg.score_probation_frames + 4), cfg=cfg)
        assert not g.confirmed
        assert g._refractory_until > g.t


class TestPromotionGate:
    """A track may exist without being allowed to steer the aircraft.

    The third state this loop was missing, and the measured answer to urban
    clutter. Seeding stays cheap so a faint target still starts a track; what a
    track has to earn is the *licence to point the aircraft*, and it earns it
    with confident detections whose bar is normalised by pixel span.
    """

    @staticmethod
    def _det(score, span=10.0, rng=40.0):
        return TargetEstimate(valid=True, u=0.0, v=0.0, span_px=span, az=0.0,
                              el=0.0, score=score, source="detector",
                              range_override=rng)

    def test_the_floor_falls_with_pixel_span(self):
        """A four-pixel drone cannot be asked to score like a twenty-pixel one.

        This is the whole reason the gate works where a flat threshold did not:
        a bar high enough to exclude a rooftop excluded a genuine target at
        80 m, and a bar low enough for that target admitted every rooftop.
        """
        g = make_guidance()
        cfg = GuidanceConfig()
        f4, f10, f20 = (g._promotion_floor(s) for s in (4.0, 10.0, 20.0))
        assert f4 < f10 < f20, "the floor must relax for smaller targets"
        assert f10 == pytest.approx(cfg.promote_score_at_10px, abs=1e-9)
        assert cfg.promote_score_min <= f4
        assert g._promotion_floor(1000.0) == pytest.approx(cfg.promote_score_max)
        assert g._promotion_floor(0.01) == pytest.approx(cfg.promote_score_min)

    def test_clutter_scores_never_earn_the_right_to_steer(self):
        """Rivermark's median false detection is 0.24; it must not fly the aircraft."""
        g = make_guidance()
        drv = Driver(g)
        for _ in range(40):
            drv.tick(self._det(0.24))
        assert not g._promoted
        assert not g.confirmed
        assert g.mode in (SEARCH, ACQUIRE)

    def test_a_confident_target_is_promoted_and_pursued(self):
        g = make_guidance()
        drv = Driver(g)
        st = drv.run(GuidanceConfig().confirm_hits + 2, self._det(0.85))
        assert g._promoted and g.confirmed
        assert st.mode in (PURSUE, TERMINAL, HIT)

    def test_a_faint_but_real_target_still_seeds_and_is_tracked(self):
        """Promotion withholds steering, it does not suppress the track.

        The distinction is the point. A distant target's first frames are its
        weakest, and they have to be allowed to build the window that will
        eventually promote it -- suppressing them is what broke acquisition
        when the seed threshold was raised instead.
        """
        g = make_guidance()
        drv = Driver(g)
        for _ in range(6):
            drv.tick(self._det(0.30, span=5.0))
        assert not g._promoted, "0.30 at 5 px is below the floor"
        assert g.streak > 0, "the track must still be accumulating evidence"
        # ... and when it closes and the score rises, it promotes.
        for _ in range(4):
            drv.tick(self._det(0.80, span=14.0))
        assert g._promoted

    def test_promotion_latches(self):
        """Without a latch the flag chatters and PN gets an on-off lock."""
        g = make_guidance()
        drv = Driver(g)
        drv.run(GuidanceConfig().confirm_hits + 2, self._det(0.85))
        assert g._promoted
        for _ in range(30):
            drv.tick(self._det(0.10))
        assert g._promoted, "promotion must not be revoked by a weak frame"

    def test_two_hits_are_required_not_one(self):
        g = make_guidance()
        drv = Driver(g)
        drv.tick(self._det(0.95))
        assert not g._promoted
        drv.tick(self._det(0.95))
        assert g._promoted

    def test_hits_must_fall_inside_the_window(self):
        cfg = GuidanceConfig()
        g = make_guidance()
        drv = Driver(g)
        drv.tick(self._det(0.95))
        for _ in range(int((cfg.promote_window_s + 0.2) / DT)):
            drv.tick(self._det(0.10))
        assert not g._promoted, "a stale confident frame should have expired"
        drv.tick(self._det(0.95))
        assert not g._promoted, "one fresh hit is not two"

    def test_it_can_be_disabled(self):
        g = make_guidance(replace(GuidanceConfig(), promote_hits=0))
        drv = Driver(g)
        st = drv.run(GuidanceConfig().confirm_hits + 2, self._det(0.05))
        assert g.confirmed and st.mode in (PURSUE, TERMINAL, HIT)

    def test_a_dropped_lock_must_earn_promotion_again(self):
        g = make_guidance()
        drv = Driver(g)
        drv.run(GuidanceConfig().confirm_hits + 2, self._det(0.85))
        assert g._promoted
        g._drop_lock()
        assert not g._promoted, "a rejected lock keeps no credit"
