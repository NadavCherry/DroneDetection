"""Intruder-entry engagements: the frame starts empty and something arrives.

These guard the property the suite exists for, which is easy to break silently.
Every earlier suite placed the target where the camera was already pointing, so
the runs began *after* acquisition -- the system scored well on a problem it was
not being asked. The checks here are that an ingress scenario really does start
with nothing in frame, that the intruder really does cross the field of view,
and that it does not begin evading before it has been found.

The constant-bearing test is the important one. A target flying exactly at an
observer holds a fixed bearing -- that is the definition of a collision course --
so it would sit frozen at the frame edge forever. Getting a target to *cross* the
frame is a geometric requirement, not a detail, and it is the requirement that
was got wrong first.
"""
from __future__ import annotations

import math

import pytest

from pursuit.dynamics import Airframe
from pursuit.episode import ScenarioConfig, place_engagement
from pursuit.evader import EvaderConfig, evader_limits, make_evader
from pursuit.geometry import world_to_body
from pursuit.sandbox import INGRESS_ENTRIES, SIM_INTRINSICS, build_suite

INTR = SIM_INTRINSICS
ECFG = EvaderConfig()


def _in_frame(chaser_xyz, chaser_yaw, target_xyz) -> bool:
    d = [target_xyz[i] - chaser_xyz[i] for i in range(3)]
    fwd, left, up = world_to_body(chaser_yaw, *d)
    if fwd <= 0.05:
        return False
    u = INTR.cx - INTR.fx * (left / fwd)
    v = INTR.cy - INTR.fy * (up / fwd)
    return 0.0 <= u < INTR.width and 0.0 <= v < INTR.height


def _fly_transit(sc: ScenarioConfig):
    """Open-loop transit against a stationary chaser -> (entered, seconds_seen)."""
    c, cyaw, t0, tyaw, aim = place_engagement(sc, (0.0, 0.0), 0.0,
                                              ECFG.altitude_band[0])
    assert aim is not None
    tgt = Airframe(xyz=t0, yaw=tyaw, ground_z=0.0)
    tgt.limits = evader_limits(ECFG)
    ev = make_evader(sc.policy, sc.seed, 0.0, ECFG, heading0=tyaw,
                     centre_xy=(c[0], c[1]))
    ev.arm_ingress(aim, sc.transit_speed or ECFG.speed)
    entered, seen = None, 0
    for k in range(int(sc.max_seconds / 0.05)):
        tgt.step(ev.command(k * 0.05, tgt, c), 0.05)
        if _in_frame(c, cyaw, tgt.xyz):
            seen += 1
            if entered is None:
                entered = k * 0.05
    return c, cyaw, t0, entered, seen * 0.05


ING = build_suite("ingress", ScenarioConfig())


@pytest.mark.parametrize("sc", ING, ids=[s.name for s in ING])
def test_frame_starts_empty(sc):
    """The whole point: at t=0 the camera sees nothing."""
    c, cyaw, t0, _tyaw, _aim = place_engagement(sc, (0.0, 0.0), 0.0,
                                                ECFG.altitude_band[0])
    assert not _in_frame(c, cyaw, t0), (
        f"{sc.name} starts with the intruder already in frame, which skips "
        f"acquisition -- the thing the suite exists to test")


@pytest.mark.parametrize("sc", ING, ids=[s.name for s in ING])
def test_intruder_transits_into_view(sc):
    """It must actually arrive, and linger long enough to be findable."""
    _c, _cyaw, _t0, entered, seen = _fly_transit(sc)
    assert entered is not None, f"{sc.name} never enters the field of view"
    assert seen >= 1.0, (
        f"{sc.name} is only in frame for {seen:.2f}s -- too brief for a "
        f"detector needing consecutive hits to seed a track")


def test_collision_course_holds_constant_bearing():
    """Why a zero-offset aim point cannot work, stated as a test.

    A target flying straight at the observer does not move in the image. If this
    ever stops being true the projection model is wrong, and the ingress table's
    signed offsets -- which exist purely to defeat this -- would be cargo cult.
    """
    chaser = (0.0, 0.0, 25.0)
    yaw = 0.0
    start = (60.0, 45.0, 25.0)          # well off the boresight
    vel = [chaser[i] - start[i] for i in range(3)]
    n = math.sqrt(sum(v * v for v in vel))
    vel = [v / n * 9.0 for v in vel]

    bearings = []
    for k in range(60):
        p = [start[i] + vel[i] * k * 0.1 for i in range(3)]
        d = [p[i] - chaser[i] for i in range(3)]
        fwd, left, _up = world_to_body(yaw, *d)
        if fwd > 1.0:
            bearings.append(math.atan2(left, fwd))
    assert len(bearings) > 20
    assert max(bearings) - min(bearings) < 1e-6, (
        "an inbound target changed bearing; the collision-course geometry the "
        "ingress aim points are built around does not hold")


def test_ingress_entries_are_aimed_to_cross_the_boresight():
    """A target entering from the left must be aimed to pass on the right.

    Asserted as geometry rather than as sign arithmetic on purpose: the two
    conventions here point opposite ways -- ``start_bearing_deg`` is positive to
    the *left* (yaw is CCW) while ``transit_miss_m`` is positive to the *right*
    of the boresight -- so a sign rule is exactly the kind of thing that reads
    plausibly and is backwards. Computing both bearings and comparing them
    cannot be got backwards.
    """
    for label, bearing, _elev, lateral, _vert, ahead in INGRESS_ENTRIES:
        assert ahead > 0.0, f"{label}: aim point must be ahead of the chaser"
        # Chaser at the origin facing +x, so a bearing is just an angle.
        aim_bearing = math.degrees(math.atan2(-lateral, ahead))
        if abs(bearing) < 175.0 and abs(lateral) > 1e-9:
            assert bearing * aim_bearing < 0.0, (
                f"{label}: starts {bearing:+.0f} deg off the nose and is aimed "
                f"at {aim_bearing:+.0f} deg -- same side of the boresight, so "
                f"the course never sweeps across the frame")


def test_transit_then_evade_switches_only_on_reveal():
    ev = make_evader("evasive", seed=3, ground_z=0.0, cfg=ECFG, heading0=0.0)
    ev.arm_ingress((0.0, 0.0, 25.0), 9.0)
    assert not ev.revealed

    own, chaser = (80.0, 0.0, 25.0), (0.0, 0.0, 25.0)
    v_transit = ev.desired_velocity(0.0, own, chaser)
    # Transiting means closing on the aim point, i.e. moving -x here.
    assert v_transit[0] < 0.0

    ev.reveal(1.0)
    assert ev.revealed and ev.revealed_at == 1.0
    v_evade = ev.desired_velocity(1.0, own, chaser)
    assert v_evade != v_transit, "reveal did not change the target's behaviour"

    ev.reveal(5.0)
    assert ev.revealed_at == 1.0, "reveal must be idempotent, not re-stamped"


def test_reveal_is_a_noop_for_ordinary_scenarios():
    """Non-ingress evaders are revealed from birth and must be unaffected."""
    ev = make_evader("weave", seed=1, ground_z=0.0, cfg=ECFG, heading0=0.0)
    assert ev.revealed
    before = ev.desired_velocity(0.5, (40.0, 0.0, 25.0), (0.0, 0.0, 25.0))
    ev.reveal(0.5)
    after = ev.desired_velocity(0.5, (40.0, 0.0, 25.0), (0.0, 0.0, 25.0))
    assert before == after


def test_chaser_start_is_scattered_across_scenarios():
    """Flying every engagement from one spot measures the scenery, not the drone."""
    starts = {(round(s.chaser_offset_xy[0], 2), round(s.chaser_offset_xy[1], 2))
              for s in ING}
    yaws = {round(s.chaser_yaw_deg, 1) for s in ING}
    assert len(starts) >= 8, f"only {len(starts)} distinct chaser positions"
    assert len(yaws) >= 8, f"only {len(yaws)} distinct chaser headings"


def test_placement_is_shared_between_sandbox_and_sim():
    """One trigonometry, used by both loops.

    A sandbox that places its aircraft differently from the simulator stops
    predicting it, quietly, and every sweep run in it becomes a measurement of
    something else.
    """
    import inspect

    from pursuit import episode, sandbox

    assert "place_engagement" in inspect.getsource(sandbox.run_episode)
    assert "place_engagement" in inspect.getsource(episode.Episode._initial_poses)


def test_default_scenario_geometry_is_unchanged():
    """The new fields must not move any pre-existing scenario."""
    sc = ScenarioConfig(start_range_m=40.0, start_bearing_deg=30.0,
                        start_elevation_deg=-10.0, altitude_m=25.0)
    c, cyaw, t, _tyaw, aim = place_engagement(sc, (7.0, -3.0), 5.0, 8.0)
    assert aim is None and cyaw == 0.0
    assert c == pytest.approx((7.0, -3.0, 30.0))
    b, e, r = math.radians(30.0), math.radians(-10.0), 40.0
    assert t == pytest.approx((7.0 + r * math.cos(e) * math.cos(b),
                               -3.0 + r * math.cos(e) * math.sin(b),
                               30.0 + r * math.sin(e)))


# ============================================================== pass geometry

class TestPassGeometry:
    """Signs on :func:`pass_geometry`, because every aim-bias claim rests on them.

    The whole "we pass N cm above/below" result is one sign away from being
    exactly wrong, and it is not self-checking: a flipped vertical convention
    produces numbers that look just as plausible and point the fix in the
    opposite direction. Each case below is a geometry whose answer is obvious by
    inspection.
    """

    def test_chaser_passing_over_the_top_reads_positive(self):
        from pursuit.episode import pass_geometry
        # Chaser 2 m higher, flying +x at 10; target level, stationary.
        g = pass_geometry(p=(0.0, 0.0, 27.0), vp=(10.0, 0.0, 0.0),
                          q=(50.0, 0.0, 25.0), vq=(0.0, 0.0, 0.0), yaw=0.0)
        assert g["vertical_m"] == pytest.approx(2.0, abs=1e-9)
        assert g["lateral_m"] == pytest.approx(0.0, abs=1e-9)
        assert g["cpa_m"] == pytest.approx(2.0, abs=1e-9)

    def test_chaser_passing_underneath_reads_negative(self):
        from pursuit.episode import pass_geometry
        g = pass_geometry(p=(0.0, 0.0, 23.0), vp=(10.0, 0.0, 0.0),
                          q=(50.0, 0.0, 25.0), vq=(0.0, 0.0, 0.0), yaw=0.0)
        assert g["vertical_m"] == pytest.approx(-2.0, abs=1e-9)

    def test_target_to_the_right_reads_positive_lateral(self):
        from pursuit.episode import pass_geometry
        # Facing +x, world -y is the chaser's right (yaw CCW, body +y is left).
        g = pass_geometry(p=(0.0, 0.0, 25.0), vp=(10.0, 0.0, 0.0),
                          q=(50.0, -3.0, 25.0), vq=(0.0, 0.0, 0.0), yaw=0.0)
        assert g["lateral_m"] == pytest.approx(3.0, abs=1e-9)
        assert g["vertical_m"] == pytest.approx(0.0, abs=1e-9)

    def test_lateral_sign_is_relative_to_heading_not_the_world(self):
        from pursuit.episode import pass_geometry
        # Same world geometry, chaser turned 180 deg: right becomes left.
        kw = dict(q=(0.0, -3.0, 25.0), vq=(0.0, 0.0, 0.0))
        a = pass_geometry(p=(-50.0, 0.0, 25.0), vp=(10.0, 0.0, 0.0), yaw=0.0, **kw)
        b = pass_geometry(p=(50.0, 0.0, 25.0), vp=(-10.0, 0.0, 0.0),
                          yaw=math.pi, **kw)
        assert a["lateral_m"] == pytest.approx(3.0, abs=1e-9)
        assert b["lateral_m"] == pytest.approx(-3.0, abs=1e-9)

    def test_it_extrapolates_past_the_sampled_instant(self):
        """The point of the function: the pass has not happened yet at the break.

        At the tick the hit is declared the aircraft are still closing, so the
        sampled separation is dominated by un-flown forward distance. A true
        head-on pass must report ~0 cross-track, not the range still remaining.
        """
        from pursuit.episode import pass_geometry
        g = pass_geometry(p=(0.0, 0.0, 25.0), vp=(15.0, 0.0, 0.0),
                          q=(0.9, 0.0, 25.0), vq=(-9.0, 0.0, 0.0), yaw=0.0)
        assert g["cpa_m"] == pytest.approx(0.0, abs=1e-9)
        assert g["dt_s"] > 0.0, "must look forward in time, not report the sample"

    def test_it_never_looks_backwards(self):
        """Already separating -> the closest approach was in the past, report now."""
        from pursuit.episode import pass_geometry
        g = pass_geometry(p=(0.0, 0.0, 25.0), vp=(-15.0, 0.0, 0.0),
                          q=(5.0, 0.0, 25.0), vq=(9.0, 0.0, 0.0), yaw=0.0)
        assert g["dt_s"] == 0.0
        assert g["cpa_m"] == pytest.approx(5.0, abs=1e-9)

    def test_components_sum_in_quadrature_to_the_distance(self):
        from pursuit.episode import pass_geometry
        g = pass_geometry(p=(1.0, -2.0, 24.0), vp=(11.0, 3.0, 1.0),
                          q=(9.0, 4.0, 26.5), vq=(-2.0, 5.0, -1.0), yaw=0.7)
        got = math.sqrt(g["along_m"] ** 2 + g["lateral_m"] ** 2
                        + g["vertical_m"] ** 2)
        assert got == pytest.approx(g["cpa_m"], rel=1e-12), (
            "the along/lateral/vertical triad is not orthonormal")


# ============================================================ approach cases

class TestApproachSuite:
    """The four relative-motion cases must behave the way their names claim.

    A suite whose scenario names do not match what the aircraft actually do is
    worse than no suite: every table built from it reads as evidence about
    crossing targets or tail chases while measuring something else. These are
    cheap assertions against the thing the name promises.
    """

    @staticmethod
    def _fly(sc, ticks=80):
        c, cyaw, t0, tyaw, aim = place_engagement(sc, (0.0, 0.0), 0.0,
                                                  ECFG.altitude_band[0])
        tgt = Airframe(xyz=t0, yaw=tyaw, ground_z=0.0)
        tgt.limits = evader_limits(ECFG)
        ev = make_evader(sc.policy, sc.seed, 0.0, ECFG, heading0=tyaw,
                         centre_xy=(c[0], c[1]))
        if aim is not None:
            ev.arm_ingress(aim, sc.transit_speed or ECFG.speed)
        us = []
        for k in range(ticks):
            tgt.step(ev.command(k * 0.05, tgt, c), 0.05)
            d = [tgt.xyz[i] - c[i] for i in range(3)]
            fwd, left, _up = world_to_body(cyaw, *d)
            if fwd > 0.05:
                u = INTR.cx - INTR.fx * (left / fwd)
                if 0.0 <= u < INTR.width:
                    us.append(u)
        return c, cyaw, t0, tgt, us

    def _cases(self, prefix):
        return [s for s in build_suite("approach", ScenarioConfig())
                if s.name.startswith(prefix)]

    def test_inbound_closes_and_starts_unseen(self):
        for sc in self._cases("inbound"):
            c, cyaw, t0, tgt, _us = self._fly(sc, ticks=60)
            assert not _in_frame(c, cyaw, t0), f"{sc.name} starts in frame"
            assert math.dist(c, tgt.xyz) < math.dist(c, t0), (
                f"{sc.name} did not close on the chaser")

    def test_outbound_is_a_genuine_tail_chase(self):
        """It must open the range and start in view -- it is already running."""
        for sc in self._cases("outbound"):
            c, cyaw, t0, tgt, us = self._fly(sc, ticks=60)
            assert _in_frame(c, cyaw, t0), (
                f"{sc.name}: a fleeing target is already in the engagement, "
                f"not entering it")
            assert math.dist(c, tgt.xyz) > math.dist(c, t0), (
                f"{sc.name} did not open the range, so it is not outbound")

    def test_crossings_sweep_in_opposite_directions(self):
        """L-to-R must enter near the left edge and drift right, and vice versa."""
        drifts = {}
        for prefix in ("left-to-right", "right-to-left"):
            for sc in self._cases(prefix):
                _c, _y, _t0, _tg, us = self._fly(sc)
                if len(us) > 2:
                    drifts.setdefault(prefix, []).append(us[-1] - us[0])
        assert drifts.get("left-to-right"), "no left-to-right crossing was visible"
        assert drifts.get("right-to-left"), "no right-to-left crossing was visible"
        assert all(d > 0 for d in drifts["left-to-right"]), (
            f"left-to-right did not move right: {drifts['left-to-right']}")
        assert all(d < 0 for d in drifts["right-to-left"]), (
            f"right-to-left did not move left: {drifts['right-to-left']}")

    def test_every_approach_case_is_flown_at_several_ranges(self):
        names = [s.name for s in build_suite("approach", ScenarioConfig())]
        for prefix in ("inbound", "outbound", "left-to-right", "right-to-left"):
            got = [n for n in names if n.startswith(prefix)]
            assert len(got) >= 3, f"{prefix} flown at only {len(got)} range(s)"

    def test_mission_suite_covers_entries_motions_and_the_ladder(self):
        """The suite that gets recorded must contain all three families."""
        names = {s.name for s in build_suite("mission", ScenarioConfig())}
        assert any(n.startswith("in-") for n in names), "no ingress entries"
        assert any(n.startswith("inbound") for n in names), "no approach cases"
        assert any(n.startswith("L1-") for n in names), "no difficulty ladder"
        assert len(names) >= 28


# ============================================================= point defence

class TestDefendSuite:
    """The intruder is going for a building, and the clock is the adversary.

    This suite is the only one the interceptor can lose by being *slow*, so the
    properties worth pinning are the ones that would quietly turn a loss back
    into a draw: the intruder must actually press its attack, an undetected one
    must reach the building, and a struck building must be reported as a struck
    building rather than as whatever else was also true at the time.
    """

    def _suite(self):
        return build_suite("defend", ScenarioConfig())

    def test_every_case_defends_a_real_asset_away_from_the_chaser(self):
        for sc in self._suite():
            assert sc.defend_xy is not None, f"{sc.name} defends nothing"
            d = math.hypot(*sc.defend_xy)
            assert 30.0 < d < 90.0, (
                f"{sc.name}: asset {d:.0f} m from station — too close is not a "
                f"defence problem, too far is outside the arena")
            assert sc.strike_radius_m > 0.0

    def test_the_intruder_aims_at_the_asset_not_the_chaser(self):
        for sc in self._suite():
            c, _yaw, t0, _tyaw, aim = place_engagement(
                sc, (0.0, 0.0), 0.0, ECFG.altitude_band[0])
            assert aim is not None
            assert math.hypot(aim[0] - sc.defend_xy[0],
                              aim[1] - sc.defend_xy[1]) < 1e-6
            # ... and the asset is not simply where the chaser is standing
            assert math.dist((aim[0], aim[1]), (c[0], c[1])) > 20.0

    def test_the_intruder_presses_its_attack(self):
        """No proximity reveal: an undetected intruder has no reason to break off.

        Letting it break off because the interceptor happens to be nearby saves
        the asset in exactly the runs where the system failed, converting the
        only failure that matters into a quiet non-event.
        """
        for sc in self._suite():
            assert sc.reveal_range_m == 0.0, (
                f"{sc.name} would abandon its attack on proximity alone")

    def test_an_undetected_intruder_actually_reaches_the_building(self):
        """Open loop, chaser blind: the threat must be real."""
        for sc in self._suite()[:4]:
            c, _y, t0, tyaw, aim = place_engagement(
                sc, (0.0, 0.0), 0.0, ECFG.altitude_band[0])
            tgt = Airframe(xyz=t0, yaw=tyaw, ground_z=0.0)
            tgt.limits = evader_limits(ECFG)
            ev = make_evader(sc.policy, sc.seed, 0.0, ECFG, heading0=tyaw,
                             centre_xy=(c[0], c[1]))
            ev.arm_ingress(aim, sc.transit_speed or ECFG.speed)
            struck = False
            for k in range(int(sc.max_seconds / 0.05)):
                tgt.step(ev.command(k * 0.05, tgt, c), 0.05)
                if math.dist(tgt.xyz, aim) <= sc.strike_radius_m:
                    struck = True
                    break
            assert struck, (
                f"{sc.name}: an unopposed intruder never reaches its target, so "
                f"the scenario poses no threat and cannot be failed")

    def test_a_struck_asset_is_reported_as_a_struck_asset(self):
        """`target_struck` must survive the post-loop outcome bookkeeping.

        It can be simultaneously true that the asset was hit and that the
        interceptor never acquired anything; reporting the cause instead of the
        consequence hides a lost building behind a tracking statistic.
        """
        from pursuit.guidance import GuidanceConfig
        from pursuit.sandbox import (SIM_INTRINSICS, SyntheticCamera,
                                     run_episode)

        struck = []
        for sc in self._suite():
            cam = SyntheticCamera(SIM_INTRINSICS, seed=sc.seed, span_bias=0.92,
                                  noise_px=0.6, span_noise=0.10)
            r = run_episode(sc, GuidanceConfig(), ECFG, cam)
            if r.struck_asset:
                struck.append(r)
                assert r.outcome == "target_struck", (
                    f"{sc.name}: asset struck but reported as {r.outcome!r}")
                assert not r.success
        assert struck, "no scenario in the suite is currently losable"

    def test_an_intercept_records_how_much_time_was_left(self):
        from pursuit.guidance import GuidanceConfig
        from pursuit.sandbox import (SIM_INTRINSICS, SyntheticCamera,
                                     run_episode)

        margins = []
        for sc in self._suite():
            cam = SyntheticCamera(SIM_INTRINSICS, seed=sc.seed, span_bias=0.92,
                                  noise_px=0.6, span_noise=0.10)
            r = run_episode(sc, GuidanceConfig(), ECFG, cam)
            if r.success and r.strike_margin_s is not None:
                margins.append(r.strike_margin_s)
                assert r.strike_margin_s >= 0.0
        assert margins, ("no intercept reported a strike margin — the metric "
                         "the mission is judged on is not being recorded")
