"""Tests for :mod:`pursuit.episode` scoring and :mod:`pursuit.perception` tracking.

Two things get exercised here, and both are places where the "obvious"
implementation is quietly wrong:

``segment_cpa``
    The whole reason the function exists is that a 20 Hz sampled range is not
    the geometry that happened. Every case below is analytic -- the answer is
    computed by hand, not read off the implementation -- and the headline case
    is a crossing pass whose true closest approach is strictly between the two
    samples that bracket it.

``SingleTargetTracker``
    Confirmation, coasting, death, gating, re-seeding, and the one line that
    decides whether guidance sees a raw box centre or a filtered one.

Everything is deterministic: no renderer, no network, no randomness that is not
explicitly seeded.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pursuit.episode import segment_cpa
from pursuit.geometry import Intrinsics, bearing_from_pixel
from pursuit.perception import (
    Box,
    OracleDetector,
    Perception,
    SingleTargetTracker,
    TrackerConfig,
)

# The rig's real camera (mirrors pursuit.sandbox.SIM_INTRINSICS).
INTR = Intrinsics(width=1440, height=840, fx=921.8145952785566,
                  fy=923.9695163260498, cx=691.6137045337061,
                  cy=257.22911647658873)

DT = 0.05  # the 20 Hz control tick the whole loop runs at


def box_at(cx: float, cy: float, span: float = 20.0, score: float = 0.9) -> Box:
    """A square detection of ``span`` px centred on ``(cx, cy)``."""
    h = 0.5 * span
    return Box(cx - h, cy - h, cx + h, cy + h, score, "drone")


def sep(a, b) -> float:
    """Euclidean separation, computed independently of ``segment_cpa``."""
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


# ==================================================================== segment_cpa

class TestSegmentCpa:
    """The scoring primitive: closest approach *between* two samples."""

    def test_crossing_pass_collides_between_the_samples(self):
        """The case the function exists for: both samples say 11.2 m, truth is 0.

        Chaser runs along +x, target crosses along +y, and they occupy the same
        point at exactly half a tick. A scorer that looked at the endpoints
        would record a comfortable miss for a dead-centre collision.
        """
        p0, p1 = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
        q0, q1 = (5.0, -10.0, 0.0), (5.0, 10.0, 0.0)

        d, u = segment_cpa(p0, p1, q0, q1)

        assert u == pytest.approx(0.5, abs=1e-12)
        assert d == pytest.approx(0.0, abs=1e-12)
        # ... while both sampled ranges are two orders of magnitude larger.
        assert sep(p0, q0) == pytest.approx(math.sqrt(125.0), abs=1e-12)
        assert sep(p1, q1) == pytest.approx(math.sqrt(125.0), abs=1e-12)

    def test_crossing_pass_with_a_known_interior_miss(self):
        """Same crossing, lifted 20 cm in z: the analytic minimum is that 20 cm."""
        p0, p1 = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
        q0, q1 = (5.0, -10.0, 0.2), (5.0, 10.0, 0.2)

        d, u = segment_cpa(p0, p1, q0, q1)

        assert u == pytest.approx(0.5, abs=1e-12)
        assert d == pytest.approx(0.2, abs=1e-12)
        assert 0.0 < u < 1.0
        assert d < min(sep(p0, q0), sep(p1, q1)) - 10.0

    def test_20hz_tick_hides_a_hit_between_two_samples(self):
        """The docstring's own scenario, made exact.

        A head-on merge at 31 m/s closing. Both bracketing samples read 0.80 m,
        so a 0.5 m hit radius applied to the *sampled* range scores a miss; the
        true closest approach is 0.20 m, which is a hit.
        """
        half_gap = 2.0 * math.sqrt(0.6)          # relative displacement per tick
        p0 = (0.0, 0.0, 25.0)
        p1 = (0.75, 0.0, 25.0)                   # chaser at 15 m/s over 50 ms
        d0 = np.array([half_gap / 2.0, 0.2, 0.0])
        dv = np.array([-half_gap, 0.0, 0.0])
        q0 = tuple(np.array(p0) + d0)
        q1 = tuple(np.array(p1) + d0 + dv)

        d, u = segment_cpa(p0, p1, q0, q1)

        assert sep(p0, q0) == pytest.approx(0.8, abs=1e-12)
        assert sep(p1, q1) == pytest.approx(0.8, abs=1e-12)
        assert u == pytest.approx(0.5, abs=1e-12)
        assert d == pytest.approx(0.2, abs=1e-12)
        hit_radius = 0.5
        assert d <= hit_radius                    # scored on the geometry ...
        assert min(sep(p0, q0), sep(p1, q1)) > hit_radius   # ... not the sampling

    def test_parallel_motion_holds_its_separation(self):
        """Equal velocities: relative motion is zero, so the gap never changes."""
        d, u = segment_cpa((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                           (0.0, 5.0, 0.0), (1.0, 5.0, 0.0))
        assert d == 5.0
        assert u == 0.0

    def test_zero_relative_velocity_when_both_are_stationary(self):
        d, u = segment_cpa((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                           (3.0, 4.0, 0.0), (3.0, 4.0, 0.0))
        assert d == 5.0
        assert u == 0.0

    def test_relative_velocity_below_the_degenerate_threshold_short_circuits(self):
        """``denom <= 1e-12`` takes the constant-separation branch verbatim.

        The relative displacement here is 2e-7 m over the tick, so the branch
        costs at most 2e-7 m of accuracy -- which is the point of the guard.
        """
        d, u = segment_cpa((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0), (1.0 - 2e-7, 0.0, 0.0))
        assert d == 1.0          # exactly |d0|, not the true 0.9999998
        assert u == 0.0

    def test_u_clamps_to_zero_when_the_minimum_is_in_the_past(self):
        """Target receding: unclamped the parabola minimises at u = -1."""
        d, u = segment_cpa((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        assert u == 0.0
        assert d == pytest.approx(1.0, abs=1e-12)   # the u=0 separation

    def test_u_clamps_to_one_when_the_minimum_is_beyond_the_tick(self):
        """Closing but not yet past: unclamped the minimum sits at u = 3."""
        d, u = segment_cpa((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                           (3.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        assert u == 1.0
        assert d == pytest.approx(2.0, abs=1e-12)   # the u=1 separation

    def test_u_is_always_in_the_unit_interval_and_d_never_exceeds_the_samples(self):
        rng = np.random.default_rng(20250807)
        for _ in range(300):
            p0, p1, q0, q1 = (rng.normal(0.0, 12.0, 3) for _ in range(4))
            d, u = segment_cpa(p0, p1, q0, q1)
            assert 0.0 <= u <= 1.0
            assert d >= 0.0
            assert d <= min(sep(p0, q0), sep(p1, q1)) + 1e-9

    def test_matches_a_dense_brute_force_search(self):
        """The analytic minimum is the true minimum over the segment."""
        rng = np.random.default_rng(11)
        grid = np.linspace(0.0, 1.0, 40001)
        for _ in range(60):
            p0, p1, q0, q1 = (rng.normal(0.0, 12.0, 3) for _ in range(4))
            d, u = segment_cpa(p0, p1, q0, q1)
            P = p0[None, :] + (p1 - p0)[None, :] * grid[:, None]
            Q = q0[None, :] + (q1 - q0)[None, :] * grid[:, None]
            brute = np.linalg.norm(Q - P, axis=1)
            assert d == pytest.approx(float(brute.min()), abs=1e-6)
            assert d <= float(brute.min()) + 1e-9
            assert abs(u - float(grid[int(brute.argmin())])) < 2e-3

    def test_is_symmetric_in_chaser_and_target(self):
        p0, p1 = (1.0, -2.0, 3.0), (4.0, 1.0, 2.0)
        q0, q1 = (-3.0, 5.0, 0.0), (2.0, -1.0, 4.0)
        a = segment_cpa(p0, p1, q0, q1)
        b = segment_cpa(q0, q1, p0, p1)
        assert a[0] == pytest.approx(b[0], abs=1e-12)
        assert a[1] == pytest.approx(b[1], abs=1e-12)

    def test_returns_plain_python_floats(self):
        """``EpisodeResult`` is round()ed and JSON-dumped; numpy scalars leak."""
        d, u = segment_cpa(np.zeros(3), np.ones(3), np.array([0.0, 2.0, 0.0]),
                           np.array([1.0, 2.0, 1.0]))
        assert type(d) is float
        assert type(u) is float


# ============================================================ SingleTargetTracker

class TestTrackerLifecycle:

    def test_confirms_only_after_confirm_hits(self):
        tr = SingleTargetTracker(TrackerConfig(init_hits=1, confirm_hits=3))

        for i in range(2):
            tr.step([box_at(400.0, 300.0)], i * DT)
            est = tr.estimate(INTR)
            assert tr.kf is not None and tr.alive          # the track exists ...
            assert not tr.confirmed                        # ... but is not reported
            assert est.valid is False
            assert est.source == "none"
            assert est.u is None and est.v is None

        tr.step([box_at(400.0, 300.0)], 2 * DT)
        est = tr.estimate(INTR)
        assert tr.confirmed
        assert tr.hits == 3 and tr.misses == 0 and tr.age == 3
        assert est.valid is True
        assert est.source == "detector"
        assert est.u == pytest.approx(400.0, abs=1e-12)
        assert est.v == pytest.approx(300.0, abs=1e-12)
        assert est.age_frames == 3

    def test_first_detection_seeds_on_the_most_confident_box(self):
        """With no track there is no prediction to be near, so score decides."""
        tr = SingleTargetTracker(TrackerConfig(init_hits=1, confirm_hits=1))
        tr.step([box_at(100.0, 100.0, score=0.30),
                 box_at(900.0, 700.0, score=0.80),
                 box_at(120.0, 110.0, score=0.55)], 0.0)
        est = tr.estimate(INTR)
        assert est.u == pytest.approx(900.0, abs=1e-12)
        assert est.v == pytest.approx(700.0, abs=1e-12)
        assert est.score == pytest.approx(0.80, abs=1e-12)

    def test_coasts_through_a_dropout_and_extrapolates_forward(self):
        """One dropped frame: the estimate survives it and moves with the target."""
        tr = SingleTargetTracker(TrackerConfig(init_hits=1, confirm_hits=2))
        x0, per_frame, n = 400.0, 10.0, 12
        for i in range(n):
            tr.step([box_at(x0 + per_frame * i, 300.0)], i * DT)

        measured = tr.estimate(INTR)
        assert measured.source == "detector"
        assert measured.u == pytest.approx(x0 + per_frame * (n - 1), abs=1e-12)
        frozen = tr.kf.pos[0]              # what a zero-velocity model would say

        tr.step([], n * DT)                # the detector misses this frame
        est = tr.estimate(INTR)
        truth = x0 + per_frame * n

        assert tr.misses == 1 and tr.alive
        assert est.valid is True
        assert est.source == "coast"
        assert est.age_frames == n + 1
        # It extrapolated past the last measurement rather than freezing on it.
        assert est.u > x0 + per_frame * (n - 1)
        assert abs(est.u - truth) < 6.0
        assert abs(est.u - truth) < abs(frozen - truth)   # velocity state earns it
        assert est.v == pytest.approx(300.0, abs=1e-9)
        # A coasted estimate is deliberately reported at half confidence,
        # and it carries the *stale* box it was last measured from.
        assert est.score == pytest.approx(0.45, abs=1e-12)
        assert est.bbox == pytest.approx((x0 + per_frame * (n - 1) - 10.0, 290.0,
                                          x0 + per_frame * (n - 1) + 10.0, 310.0),
                                         abs=1e-12)
        assert est.span_px == pytest.approx(20.0, abs=1e-9)

    def test_dies_after_max_coast_frames(self):
        """Survives exactly ``max_coast_frames`` misses, dies on the next one."""
        cfg = TrackerConfig(init_hits=1, confirm_hits=1, max_coast_frames=3)
        tr = SingleTargetTracker(cfg)
        tr.step([box_at(100.0, 100.0)], 0.0)

        for i in range(1, cfg.max_coast_frames + 1):
            assert tr.step([], i * DT) is None
            assert tr.misses == i
            assert tr.alive
            assert tr.estimate(INTR).source == "coast"
            assert tr.estimate(INTR).valid is True

        # One miss too many: the track is declared lost and wiped.
        assert tr.step([], (cfg.max_coast_frames + 1) * DT) is None
        assert tr.alive is False
        assert tr.confirmed is False
        assert tr.kf is None
        assert (tr.hits, tr.misses, tr.age) == (0, 0, 0)
        assert tr.span is None and tr.last_box is None and tr.t_last is None
        est = tr.estimate(INTR)
        assert est.valid is False and est.source == "none"

    def test_reset_clears_every_field(self):
        tr = SingleTargetTracker(TrackerConfig(init_hits=1, confirm_hits=1))
        tr.step([box_at(500.0, 400.0, span=33.0, score=0.7)], 0.0)
        assert tr.confirmed
        tr.reset()
        assert tr.kf is None and tr.span is None and tr.last_box is None
        assert (tr.hits, tr.misses, tr.age) == (0, 0, 0)
        assert tr.score == 0.0 and tr.t_last is None
        assert tr.alive is False and tr.confirmed is False


class TestTrackerGating:

    GATE = 50.0

    def _seeded(self):
        tr = SingleTargetTracker(TrackerConfig(init_hits=1, 
            confirm_hits=1, gate_base_px=self.GATE, gate_per_miss_px=0.0,
            gate_span_scale=0.0, reseed_after_misses=10 ** 6,
            max_coast_frames=10 ** 6))
        tr.step([box_at(640.0, 360.0)], 0.0)
        return tr

    @pytest.mark.parametrize("offset", [0.0, 25.0, 49.0, 50.0])
    def test_gate_accepts_a_detection_inside_it(self, offset):
        tr = self._seeded()
        assert tr._gate() == pytest.approx(self.GATE, abs=1e-12)
        pick = tr.step([box_at(640.0 + offset, 360.0)], DT)
        assert pick is not None
        assert tr.misses == 0 and tr.hits == 2
        est = tr.estimate(INTR)
        assert est.source == "detector"
        assert est.u == pytest.approx(640.0 + offset, abs=1e-12)

    @pytest.mark.parametrize("offset", [50.001, 51.0, 200.0])
    def test_gate_rejects_a_far_detection(self, offset):
        """A detection outside the gate is not the thing being chased."""
        tr = self._seeded()
        pick = tr.step([box_at(640.0 + offset, 360.0)], DT)
        assert pick is None
        assert tr.misses == 1 and tr.hits == 1
        est = tr.estimate(INTR)
        assert est.source == "coast"
        # The estimate stayed on the prediction, it did not jump to the intruder.
        assert est.u == pytest.approx(640.0, abs=1e-9)
        assert est.u == pytest.approx(tr.kf.pos[0], abs=1e-12)

    def test_gate_prefers_the_nearest_candidate_not_the_most_confident(self):
        """Once there *is* a track, proximity beats score -- see the module doc."""
        tr = self._seeded()
        tr.step([box_at(640.0 + 40.0, 360.0, score=0.20),
                 box_at(640.0 + 10.0, 360.0, score=0.95)], DT)
        assert tr.estimate(INTR).u == pytest.approx(650.0, abs=1e-12)

        tr2 = self._seeded()
        tr2.step([box_at(640.0 + 10.0, 360.0, score=0.20),
                  box_at(640.0 + 40.0, 360.0, score=0.95)], DT)
        assert tr2.estimate(INTR).u == pytest.approx(650.0, abs=1e-12)
        assert tr2.score == pytest.approx(0.20, abs=1e-12)

    def test_gate_widens_by_gate_per_miss_px_each_consecutive_miss(self):
        """A track that blinked has genuinely larger uncertainty.

        Gate is ``60 + 25 * misses``, so a detection 130 px from the prediction
        is rejected at 0, 1 and 2 misses and accepted at 3 (gate 135 px).
        """
        cfg = TrackerConfig(init_hits=1, confirm_hits=1, gate_base_px=60.0,
                            gate_per_miss_px=25.0, gate_span_scale=0.0,
                            gate_max_px=1000.0, reseed_after_misses=10 ** 6,
                            max_coast_frames=10 ** 6)
        tr = SingleTargetTracker(cfg)
        tr.step([box_at(300.0, 300.0)], 0.0)

        for i in range(1, 4):
            assert tr._gate() == pytest.approx(60.0 + 25.0 * (i - 1), abs=1e-12)
            assert tr.step([box_at(430.0, 300.0)], i * DT) is None
            assert tr.misses == i

        assert tr._gate() == pytest.approx(135.0, abs=1e-12)
        assert tr.step([box_at(430.0, 300.0)], 4 * DT) is not None
        assert tr.misses == 0
        assert tr.estimate(INTR).u == pytest.approx(430.0, abs=1e-12)

    def test_gate_is_capped_by_gate_max_px(self):
        cfg = TrackerConfig(init_hits=1, confirm_hits=1, gate_base_px=60.0,
                            gate_per_miss_px=25.0, gate_span_scale=0.0,
                            gate_max_px=100.0, reseed_after_misses=10 ** 6,
                            max_coast_frames=10 ** 6)
        tr = SingleTargetTracker(cfg)
        tr.step([box_at(300.0, 300.0)], 0.0)
        for i in range(1, 8):
            tr.step([], i * DT)
        assert tr.misses == 7
        assert tr._gate() == pytest.approx(100.0, abs=1e-12)   # not 60 + 175


class TestTrackerReseed:
    """REGRESSION: the gate used to be a one-way trap."""

    RESEED_AFTER = 4

    def _cfg(self, reseed_after: int) -> TrackerConfig:
        return TrackerConfig(init_hits=1, confirm_hits=1, reseed_after_misses=reseed_after,
                             max_coast_frames=10 ** 6)

    def test_reseeds_on_the_best_detection_after_reseed_after_misses(self):
        """A drifted prediction must not exile a target that is in plain view.

        The track sits at (200, 200); the real target is at (1000, 600), 894 px
        away and outside any gate the config can grow. Before the re-seed the
        tracker coasted confidently into nothing forever.
        """
        tr = SingleTargetTracker(self._cfg(self.RESEED_AFTER))
        tr.step([box_at(200.0, 200.0, span=20.0, score=0.9)], 0.0)
        far = box_at(1000.0, 600.0, span=60.0, score=0.7)

        # Not yet: the gate is still doing its job for `reseed_after_misses`
        # frames, and the estimate stays on the (wrong) prediction.
        for i in range(1, self.RESEED_AFTER + 1):
            assert tr.step([far], i * DT) is None
            assert tr.misses == i
            est = tr.estimate(INTR)
            assert est.source == "coast"
            assert est.u == pytest.approx(200.0, abs=1e-9)
            assert tr.span == pytest.approx(20.0, abs=1e-12)

        # One frame later the prediction is judged to be the thing that is wrong.
        pick = tr.step([far], (self.RESEED_AFTER + 1) * DT)
        assert pick is far
        assert tr.misses == 0 and tr.hits == 2
        est = tr.estimate(INTR)
        assert est.source == "detector"
        assert est.u == pytest.approx(1000.0, abs=1e-12)
        assert est.v == pytest.approx(600.0, abs=1e-12)
        # A *fresh* filter, not an update: state snapped onto the detection,
        # velocity zeroed, and the span taken outright instead of EMA-blended.
        assert tr.kf.pos == pytest.approx((1000.0, 600.0), abs=1e-12)
        assert tr.kf.vel == pytest.approx((0.0, 0.0), abs=1e-12)
        assert tr.span == pytest.approx(60.0, abs=1e-12)

    def test_reseed_takes_the_highest_scoring_candidate(self):
        tr = SingleTargetTracker(self._cfg(2))
        tr.step([box_at(200.0, 200.0)], 0.0)
        far = [box_at(1000.0, 600.0, score=0.3), box_at(20.0, 800.0, score=0.9)]
        for i in range(1, 3):
            assert tr.step(far, i * DT) is None
        pick = tr.step(far, 3 * DT)
        assert pick is far[1]
        assert tr.estimate(INTR).u == pytest.approx(20.0, abs=1e-12)

    def test_without_the_reseed_the_gate_traps_the_track_forever(self):
        """The failure the re-seed exists to prevent, held in place for 60 frames.

        Same feed as above with re-seeding disabled: the target is visible on
        every single frame and the tracker never once sees it.
        """
        tr = SingleTargetTracker(self._cfg(10 ** 6))
        tr.step([box_at(200.0, 200.0)], 0.0)
        far = box_at(1000.0, 600.0, score=0.7)
        sources = set()
        for i in range(1, 60):
            assert tr.step([far], i * DT) is None
            sources.add(tr.estimate(INTR).source)

        assert sources == {"coast"}
        assert tr.misses == 59
        assert tr.hits == 1
        assert tr.kf.pos == pytest.approx((200.0, 200.0), abs=1e-9)
        assert math.hypot(1000.0 - 200.0, 600.0 - 200.0) > tr._gate()


class TestTrackerEstimate:

    def _drifted(self):
        """A confirmed track running at 10 px/frame, then a 25 px jump.

        The jump is inside the gate but far enough from the prediction that the
        filtered position and the raw box centre cannot be confused.
        """
        tr = SingleTargetTracker(TrackerConfig(init_hits=1, confirm_hits=2))
        for i in range(8):
            tr.step([box_at(400.0 + 10.0 * i, 300.0)], i * DT)
        jump_cx = 400.0 + 10.0 * 8 + 25.0
        assert tr.step([box_at(jump_cx, 300.0)], 8 * DT) is not None
        assert tr.misses == 0
        return tr, jump_cx

    def test_estimate_reports_the_raw_box_centre_when_measured(self):
        """The single most important line in the class: no filter on a measurement.

        A filtered bearing is a bearing from a moment when the aircraft was
        pointing somewhere else, and de-rotating it manufactures a world-frame
        LOS rate out of nothing.
        """
        tr, jump_cx = self._drifted()
        est = tr.estimate(INTR)

        assert est.source == "detector"
        assert est.u == jump_cx                     # exactly, not approximately
        assert est.v == 300.0
        # ... and the filter really does disagree, so this is not a tautology.
        assert abs(tr.kf.pos[0] - jump_cx) > 5.0
        assert est.score == pytest.approx(0.9, abs=1e-12)
        # The bearing is derived from the reported pixel, not the filtered one.
        assert (est.az, est.el) == pytest.approx(
            bearing_from_pixel(INTR, jump_cx, 300.0), abs=1e-15)

    def test_estimate_reports_the_filtered_position_while_coasting(self):
        """No measurement: the filter is exactly what it is there for."""
        tr, jump_cx = self._drifted()
        tr.step([], 9 * DT)
        est = tr.estimate(INTR)

        assert est.source == "coast"
        assert tr.misses == 1
        assert est.u == tr.kf.pos[0]                # exactly the filtered state
        assert est.v == tr.kf.pos[1]
        assert est.u != jump_cx
        assert (est.az, est.el) == pytest.approx(
            bearing_from_pixel(INTR, tr.kf.pos[0], tr.kf.pos[1]), abs=1e-15)

    def test_span_is_ema_smoothed_on_update_with_the_configured_tau(self):
        """Span drives range drives the speed schedule, so it is smoothed hard."""
        cfg = TrackerConfig(init_hits=1, confirm_hits=1, span_tau=0.15)
        tr = SingleTargetTracker(cfg)
        tr.step([box_at(100.0, 100.0, span=20.0)], 0.0)
        assert tr.span == pytest.approx(20.0, abs=1e-12)   # seeded outright

        tr.step([box_at(100.0, 100.0, span=40.0)], DT)
        a = 1.0 - math.exp(-DT / cfg.span_tau)
        assert tr.span == pytest.approx(20.0 + a * 20.0, abs=1e-12)
        assert tr.estimate(INTR).span_px == pytest.approx(tr.span, abs=1e-12)

    def test_span_of_a_non_square_box_is_the_larger_side(self):
        b = Box(100.0, 200.0, 140.0, 210.0, 0.5)
        assert (b.w, b.h) == (40.0, 10.0)
        assert b.span == 40.0
        assert (b.cx, b.cy) == (120.0, 205.0)

    def test_estimate_never_populates_range_override(self):
        """SUSPECTED BUG: ``TargetEstimate.range_override`` is documented as
        "Set only by the oracle sensor", and ``guidance.py`` reads it (line 610)
        to bypass the monocular range -- but nothing in the codebase ever writes
        it. ``OracleDetector`` emits a plain ``Box``, which carries no range, and
        ``SingleTargetTracker.estimate`` builds the ``TargetEstimate`` without
        the field. So the oracle flies on monocular range like every other
        detector and the guidance branch is unreachable. Asserting what the code
        currently does.
        """
        per = Perception(OracleDetector(), INTR, TrackerConfig(init_hits=1, confirm_hits=1))
        est = per.step(None, 0, 0.0,
                       {"bbox": [100.0, 200.0, 140.0, 220.0], "range_m": 30.0})
        assert est.valid is True
        assert est.range_override is None


# =========================================================== detectors / frontend

class TestOracleDetector:

    GT = {"bbox": [100.0, 200.0, 140.0, 220.0], "range_m": 30.0, "visible": True}

    def test_reproduces_the_ground_truth_box_exactly(self):
        (b,) = OracleDetector().detect(None, 0, self.GT)
        assert (b.x1, b.y1, b.x2, b.y2) == (100.0, 200.0, 140.0, 220.0)
        assert b.score == 1.0 and b.label == "drone"
        assert b.span == 40.0

    def test_no_box_without_ground_truth(self):
        det = OracleDetector()
        assert det.detect(None, 0, None) == []
        assert det.detect(None, 0, {}) == []
        assert det.detect(None, 0, {"bbox": None, "range_m": 1.0}) == []

    def test_span_bias_scales_about_the_box_centre(self):
        """A biased span is a biased *range*; the bearing must be untouched."""
        (b,) = OracleDetector(span_bias=2.0).detect(None, 0, self.GT)
        assert (b.cx, b.cy) == (120.0, 210.0)
        assert b.span == pytest.approx(80.0, abs=1e-12)
        assert b.h == pytest.approx(40.0, abs=1e-12)

    def test_max_range_m_makes_a_distant_target_undetectable(self):
        assert OracleDetector(max_range_m=20.0).detect(None, 0, self.GT) == []
        assert len(OracleDetector(max_range_m=40.0).detect(None, 0, self.GT)) == 1

    def test_latency_frames_delays_the_report_by_exactly_n_frames(self):
        det = OracleDetector(latency_frames=2)
        seen = []
        for i in range(6):
            gt = {"bbox": [float(i), 0.0, float(i) + 10.0, 5.0], "range_m": 1.0}
            out = det.detect(None, i, gt)
            seen.append(None if not out else out[0].x1)
        assert seen == [None, None, 0.0, 1.0, 2.0, 3.0]

    def _dropout_pattern(self, seed: int, n: int = 60):
        det = OracleDetector(dropout=0.5, seed=seed)
        return [len(det.detect(None, i, self.GT)) for i in range(n)]

    def test_dropout_is_deterministic_for_a_given_seed(self):
        a = self._dropout_pattern(3)
        b = self._dropout_pattern(3)
        c = self._dropout_pattern(9)
        assert a == b                       # a seeded scenario is one flight ...
        assert a != c                       # ... and a different seed a different one
        assert 0 < sum(a) < 60              # a real coin, not a stuck one
        assert abs(sum(a) / 60.0 - 0.5) < 0.2

    def test_dropout_of_one_never_reports_anything(self):
        det = OracleDetector(dropout=1.0, seed=1)
        assert all(det.detect(None, i, self.GT) == [] for i in range(20))


class TestPerception:

    GT = {"bbox": [100.0, 200.0, 140.0, 220.0], "range_m": 30.0, "visible": True}

    def test_wires_detector_and_tracker_into_one_call(self):
        per = Perception(OracleDetector(), INTR, TrackerConfig(init_hits=1, confirm_hits=2))
        first = per.step(None, 0, 0.0, self.GT)
        assert first.valid is False and first.source == "none"

        est = per.step(None, 1, DT, self.GT)
        assert est.valid is True and est.source == "detector"
        assert est.u == pytest.approx(120.0, abs=1e-12)
        assert est.v == pytest.approx(210.0, abs=1e-12)
        assert est.span_px == pytest.approx(40.0, abs=1e-12)
        assert est.offset(INTR)[0] == pytest.approx(
            (120.0 - INTR.cx) / (0.5 * INTR.width), abs=1e-12)
        assert len(per.last_boxes) == 1

    def test_a_detector_that_needs_pixels_is_not_handed_none(self):
        class NeedsFrame:
            name, needs_frame = "nf", True

            def detect(self, frame, idx, gt=None):      # pragma: no cover
                raise AssertionError("must not be called with frame=None")

        per = Perception(NeedsFrame(), INTR, TrackerConfig(init_hits=1, confirm_hits=1))
        est = per.step(None, 0, 0.0, self.GT)
        assert est.valid is False and est.source == "none"
        assert per.last_boxes == []

    def test_min_score_filters_before_association(self):
        class TwoBoxes:
            name, needs_frame = "two", False

            def detect(self, frame, idx, gt=None):
                return [Box(0.0, 0.0, 10.0, 10.0, 0.90),
                        Box(600.0, 400.0, 610.0, 410.0, 0.10)]

        per = Perception(TwoBoxes(), INTR, TrackerConfig(init_hits=1, confirm_hits=1),
                         min_score=0.5)
        est = per.step(None, 0, 0.0, None)
        assert len(per.last_boxes) == 1
        assert est.u == pytest.approx(5.0, abs=1e-12)   # the 0.10 box never existed

    def test_reset_clears_the_track(self):
        per = Perception(OracleDetector(), INTR, TrackerConfig(init_hits=1, confirm_hits=1))
        per.step(None, 0, 0.0, self.GT)
        assert per.tracker.confirmed
        per.reset()
        assert per.tracker.kf is None
        assert per.last_boxes == []
        assert per.step(None, 0, 0.0, None).valid is False

    def test_reset_clears_the_timing_counters(self):
        """``reset()`` must zero the stage timings, not just the tracker.

        One ``Perception`` is reused across a whole scenario matrix and
        ``Episode.run`` resets it per episode, so counters that survived the
        reset would make every episode's reported timing a running mean over
        that episode *and every one before it* -- which hides the thing the
        number exists to show, a detector that slows down as the target grows.
        """
        per = Perception(OracleDetector(), INTR, TrackerConfig(init_hits=1, confirm_hits=1))
        for i in range(5):
            per.step(None, i, i * DT, self.GT)
        assert per.n == 5
        assert per.timings["track_ms"] > 0.0

        assert len(per.samples["track_ms"]) == 5

        per.reset()
        assert per.n == 0
        assert per.timings == {"detect_ms": 0.0, "track_ms": 0.0}
        # The per-frame samples are the same hazard as the counters and have to
        # be cleared with them: a p95 accumulated over every episode so far is
        # not this episode's p95, and it is the statistic the frame rate is
        # judged on.
        assert per.samples == {"detect_ms": [], "track_ms": []}

        per.step(None, 0, 0.0, self.GT)
        assert per.n == 1
        report = per.stage_report()
        assert {"detect_ms", "track_ms", "detect_p95_ms", "track_p95_ms",
                "perception_ms", "perception_fps"} <= set(report)
        assert report["track_ms"] == pytest.approx(
            round(per.timings["track_ms"] / 1.0, 2), abs=1e-12)
        assert report["perception_ms"] == pytest.approx(
            report["detect_ms"] + report["track_ms"], abs=1e-9)
        assert report["perception_fps"] == pytest.approx(
            round(1000.0 / report["perception_ms"], 1), abs=1e-9)


class TestCorroboratedSeeding:
    """A lock may only start on evidence that repeats.

    This is a *safety* property, not an accuracy one. With a single-detection
    seed the closed loop locked onto scene clutter and flew at it for 724
    frames -- the tracker's belief a median 311 px from the drone, the range
    filter collapsed to 0.1 m on an oversized spurious box, and the episode
    ending having never closed. The asymmetry that fixes it: clutter false
    positives are sparse and incoherent, a real target repeats in place.
    """

    def cfg(self, **kw):
        base = dict(init_hits=2, init_score=0.20, init_window_s=0.12,
                    init_gate_px=25.0, gate_span_scale=0.0, confirm_hits=1)
        base.update(kw)
        return TrackerConfig(**base)

    def test_one_confident_detection_is_not_enough(self):
        tr = SingleTargetTracker(self.cfg())
        assert tr.step([box_at(600.0, 300.0, score=0.9)], 0.0) is None
        assert tr.kf is None and not tr.confirmed
        assert tr.estimate(INTR).valid is False

    def test_two_agreeing_detections_seed_the_track(self):
        # 10 px apart: what a 9 m/s crossing target at 40 m moves in one frame.
        tr = SingleTargetTracker(self.cfg())
        tr.step([box_at(600.0, 300.0, score=0.9)], 0.0)
        pick = tr.step([box_at(610.0, 300.0, score=0.9)], DT)
        assert pick is not None
        assert tr.kf is not None and tr.confirmed
        # It seeds on the NEW box, which is the more recent evidence.
        assert tr.estimate(INTR).u == pytest.approx(610.0, abs=1e-9)

    def test_two_detections_far_apart_do_not_seed(self):
        """Scattered false positives are exactly this: strong, and incoherent."""
        tr = SingleTargetTracker(self.cfg())
        tr.step([box_at(200.0, 200.0, score=0.9)], 0.0)
        assert tr.step([box_at(1200.0, 700.0, score=0.9)], DT) is None
        assert tr.kf is None

    def test_a_stale_candidate_expires(self):
        tr = SingleTargetTracker(self.cfg(init_window_s=0.12))
        tr.step([box_at(600.0, 300.0, score=0.9)], 0.0)
        assert tr.step([box_at(602.0, 301.0, score=0.9)], 0.5) is None
        assert tr.kf is None
        # ...but the late one is itself a fresh candidate, so the next agreeing
        # detection seeds normally.
        assert tr.step([box_at(603.0, 302.0, score=0.9)], 0.55) is not None

    def test_weak_detections_never_seed_however_many(self):
        tr = SingleTargetTracker(self.cfg())
        for i in range(20):
            assert tr.step([box_at(600.0, 300.0, score=0.15)], i * DT) is None
        assert tr.kf is None

    def test_a_scattered_false_positive_stream_at_the_measured_rate(self):
        """Clutter at the rate this detector actually produces it.

        Measured at the deployed operating point: 0.15 confident false positives
        per frame, scattered. Over a 15-second search that is ~45 chances, and
        corroboration has to hold across all of them. It is not a guarantee --
        two of them landing within the gate inside the window is possible and
        this test pins how rare -- but it turns a near-certainty into an
        exception, and the span bound below removes the damaging kind outright.
        """
        rng = np.random.default_rng(4)
        locks = 0
        trials = 40
        for trial in range(trials):
            tr = SingleTargetTracker(self.cfg())
            for i in range(300):                       # 15 s of searching
                boxes = []
                if rng.random() < 0.30:
                    boxes = [box_at(float(rng.uniform(50, 1390)),
                                    float(rng.uniform(50, 790)),
                                    score=float(rng.uniform(0.36, 0.9)))]
                tr.step(boxes, i * DT)
                if tr.kf is not None:
                    break
            locks += int(tr.kf is not None)
        assert locks <= trials * 0.10, f"{locks}/{trials} searches captured by clutter"

    def test_an_oversized_box_can_never_seed(self):
        """The specific box that caused the measured false lock.

        A first sighting is never a box filling the frame: to be that large the
        target would have to be metres away, which cannot happen without having
        been tracked in from further out. Left unchecked it drives the monocular
        range to its floor and latches the terminal phase on nothing.
        """
        tr = SingleTargetTracker(self.cfg())
        for i in range(30):
            tr.step([box_at(700.0, 300.0, span=400.0, score=0.95)], i * DT)
        assert tr.kf is None
        # The same position at a plausible size seeds immediately.
        tr.step([box_at(700.0, 300.0, span=20.0, score=0.95)], 30 * DT)
        assert tr.step([box_at(701.0, 301.0, span=20.0, score=0.95)],
                       31 * DT) is not None

    def test_a_real_target_still_locks_promptly_at_a_partial_hit_rate(self):
        """25 % of frames carry a detection -- the measured rate at 9-14 px."""
        rng = np.random.default_rng(7)
        tr = SingleTargetTracker(self.cfg())
        seeded_at = None
        for i in range(60):
            boxes = []
            if rng.random() < 0.25:
                boxes = [box_at(700.0 + rng.normal(0, 3), 300.0 + rng.normal(0, 3),
                                score=0.6)]
            tr.step(boxes, i * DT)
            if tr.kf is not None and seeded_at is None:
                seeded_at = i
        assert seeded_at is not None and seeded_at < 40

    def test_reseeding_after_a_lost_track_needs_the_same_corroboration(self):
        tr = SingleTargetTracker(self.cfg(reseed_after_misses=2,
                                          max_coast_frames=10 ** 6,
                                          gate_base_px=5.0))
        tr.step([box_at(600.0, 300.0, score=0.9)], 0.0)
        tr.step([box_at(601.0, 300.0, score=0.9)], DT)
        assert tr.kf is not None
        for i in range(2, 6):                       # drive it out of the gate
            tr.step([], i * DT)
        assert tr.misses >= 2
        # One far-away confident box must not move the lock on its own.
        assert tr.step([box_at(1200.0, 700.0, score=0.9)], 6 * DT) is None
        # A second, agreeing one may.
        assert tr.step([box_at(1205.0, 702.0, score=0.9)], 7 * DT) is not None
