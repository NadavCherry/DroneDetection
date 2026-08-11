"""Tests for :mod:`pursuit.evader`.

The evader is the thing the whole pursuit experiment is measured against, so the
properties that matter are not "does it produce a velocity" but "is the scenario
still winnable at all". Two containment guarantees carry that weight:

* the evader stays inside ``arena_radius_m``, so ``flee`` is an evasion and not a
  straight line to infinity that quietly measures the speed difference, and
* the evader stays inside ``altitude_band``, so it cannot leave upward and end
  the episode out of the camera's reach.

The second one is a **regression**. An earlier ``_altitude_guard`` *added* a
bounded push instead of replacing the vertical command, and a bounded push loses
to an unbounded policy: ``climb_flee``'s 3.5 m/s against a 3 m/s push-back is a
net +0.5 m/s, which took the target to 68 m against a 45 m ceiling and made the
scenario unwinnable by construction. Every ceiling assertion below is written so
that the additive version fails it by tens of metres.

The closed-loop tests drive the evader through its own
:class:`~pursuit.dynamics.Airframe` -- containment is a claim about where the
aircraft ends up, not about what the policy asked for -- but the chaser is a
scripted kinematic point rather than :class:`~pursuit.guidance.PursuitGuidance`,
so a failure here is a failure in ``evader.py`` and nothing else.
"""
from __future__ import annotations

import math
import random

import pytest

from pursuit.dynamics import Airframe, BodyCommand
from pursuit.evader import (
    POLICIES,
    Evader,
    EvaderConfig,
    evader_limits,
    make_evader,
)

DT = 0.05
CFG = EvaderConfig()
LOW, HIGH = CFG.altitude_band

# The ceiling is overshot by the vertical stopping distance and no more:
# v^2 / 2a, plus one tick of discretisation, from the evader's own limits.
# The additive-guard bug overshot it by ~23 m, so this bound is the regression.
CEILING_SLOP_M = 1.5


# -- harness ----------------------------------------------------------------


def _fly(policy, *, seed=0, cfg=CFG, secs=60.0, dt=DT, start=(30.0, 0.0, 20.0),
         chaser0=(0.0, 0.0, 20.0), chaser_speed=8.0, heading0=0.0,
         centre_xy=(0.0, 0.0), ground_z=0.0, vel0=(0.0, 0.0, 0.0)):
    """Fly the evader's own airframe for ``secs`` and return its trajectory.

    The chaser is a point that walks straight at the evader at a constant speed.
    It is deliberately *slower* than the evader's cruise (8 vs 9 m/s) so the
    tail chase never closes and the policies are exercised for the whole run.
    """
    ev = make_evader(policy, seed, ground_z, cfg,
                     heading0=heading0, centre_xy=centre_xy)
    target = Airframe(xyz=start, yaw=float(heading0 or 0.0), ground_z=ground_z,
                      vel=vel0)
    target.limits = evader_limits(cfg)

    chaser = list(chaser0)
    traj = []
    t = 0.0
    for _ in range(int(round(secs / dt))):
        target.step(ev.command(t, target, tuple(chaser)), dt)
        traj.append(target.xyz)
        sep = math.dist(target.xyz, chaser) or 1.0
        for k in range(3):
            chaser[k] += chaser_speed * dt * (target.xyz[k] - chaser[k]) / sep
        t += dt
    return traj


def _radius(p, centre=(0.0, 0.0)):
    return math.hypot(p[0] - centre[0], p[1] - centre[1])


def _path_length(traj):
    return sum(math.dist(a, b) for a, b in zip(traj, traj[1:]))


# -- construction -----------------------------------------------------------


def test_policies_cover_the_documented_names_and_the_ladder():
    """Every documented policy exists, and the ladder is drawn from them.

    The three added last (`sweep`, `barrel`, `evasive`) exist because the
    original seven were dominated by *radial* flight -- running straight away
    from the chaser, which produces almost no line-of-sight rate and is both the
    easiest case for the seeker and invisible on video.
    """
    from pursuit.evader import LADDER
    assert POLICIES == ("straight", "flee", "weave", "break_turn", "jink",
                        "orbit", "climb_flee", "sweep", "barrel", "evasive")
    assert set(LADDER) <= set(POLICIES)
    assert LADDER[0] == "sweep" and LADDER[-1] == "evasive"
    # Every policy must be constructible and produce a finite velocity.
    for pol in POLICIES:
        ev = make_evader(pol, 1, 0.0, EvaderConfig(), heading0=0.0,
                         centre_xy=(0.0, 0.0))
        v = ev.desired_velocity(0.5, (30.0, 0.0, 25.0), (0.0, 0.0, 25.0))
        assert len(v) == 3 and all(math.isfinite(c) for c in v)

def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError) as exc:
        Evader("corkscrew")
    assert "corkscrew" in str(exc.value)


def test_make_evader_wires_its_arguments_through():
    cfg = EvaderConfig(speed=11.0)
    ev = make_evader("weave", 12, 7.5, cfg, heading0=0.25,
                     centre_xy=[3.0, -4.0])
    assert ev.policy == "weave"
    assert ev.cfg is cfg
    assert ev.ground_z == 7.5
    assert ev.heading == 0.25
    assert ev.centre_xy == (3.0, -4.0)


def test_default_centre_is_the_world_origin():
    assert Evader("orbit").centre_xy == (0.0, 0.0)


def test_random_initial_heading_is_the_first_draw_of_the_seeded_rng():
    # Not just "in range": the exact draw, so the seed really is the only
    # source of the heading and the rng is not consumed before it.
    for seed in (0, 1, 5, 99):
        expected = random.Random(seed).uniform(-math.pi, math.pi)
        assert make_evader("straight", seed, 0.0, CFG).heading == expected


def test_evader_limits_track_the_config():
    cfg = EvaderConfig(speed=11.0, climb_mps=4.25)
    lim = evader_limits(cfg)
    assert lim.max_speed_xy == 11.0
    # Vertical authority is sized from the most demanding policy, not the
    # mildest: `barrel` and `evasive` both climb harder than `climb_mps`, and
    # sizing from `climb_mps` alone silently clipped their altitude changes to a
    # few metres -- present in the telemetry, invisible in the engagement.
    assert lim.max_speed_z == pytest.approx(
        max(3.0, cfg.climb_mps, cfg.barrel_climb_mps, 1.2 * cfg.climb_mps))
    assert lim.max_speed_z >= cfg.barrel_climb_mps
    assert lim.max_accel_xy == 10.0
    assert lim.max_accel_z == 7.0
    assert lim.max_yaw_rate == 3.0
    assert lim.max_yaw_accel == 12.0
    assert lim.min_agl == 3.0
    # A slow climber still gets a 3 m/s vertical floor, which is what makes the
    # altitude guard's 3 m/s return command flyable.
    assert evader_limits(EvaderConfig(climb_mps=0.5, barrel_climb_mps=0.5)
                         ).max_speed_z == 3.0


# -- altitude guard, in isolation -------------------------------------------


def test_altitude_guard_leaves_the_policy_alone_inside_the_band():
    ev = Evader("climb_flee")
    for z in (LOW, 20.0, 33.3, HIGH):
        for vz in (-4.0, -0.25, 0.0, 3.5):
            assert ev._altitude_guard(z, vz) == vz


def test_altitude_guard_replaces_the_climb_at_the_ceiling():
    """REGRESSION: the band wins outright; it does not merely subtract.

    The old guard added a bounded push, so ``climb_flee`` at the ceiling came
    out at ``3.5 - 3.0 = +0.5`` m/s and kept climbing. The replacement form
    returns the descent rate and ignores the policy entirely.
    """
    ev = Evader("climb_flee")
    climb = CFG.climb_mps  # 3.5 m/s, the thing that used to win
    assert ev._altitude_guard(HIGH + 1.0, climb) == -1.0
    assert ev._altitude_guard(HIGH + 0.25, climb) == -0.25
    assert ev._altitude_guard(HIGH + 10.0, climb) == -3.0
    # The additive version would have produced +0.5 for every one of those.
    assert ev._altitude_guard(HIGH + 1.0, climb) != pytest.approx(climb - 3.0)


def test_altitude_guard_never_fights_a_policy_that_is_already_returning():
    ev = Evader("straight")
    # Above the ceiling and already descending faster than the guard asks for:
    # the guard must not slow the return down.
    assert ev._altitude_guard(HIGH + 1.0, -6.0) == -6.0
    # Below the floor and already climbing faster than the guard asks for.
    assert ev._altitude_guard(LOW - 1.0, 6.0) == 6.0


def test_altitude_guard_lifts_a_descent_off_the_floor():
    ev = Evader("straight")
    assert ev._altitude_guard(LOW - 1.0, -2.0) == 1.0
    assert ev._altitude_guard(LOW - 0.5, 0.0) == 0.5
    assert ev._altitude_guard(LOW - 20.0, -2.0) == 3.0


def test_altitude_guard_is_measured_from_ground_z():
    ev = Evader("climb_flee", ground_z=100.0)
    assert ev._altitude_guard(100.0 + HIGH + 1.0, CFG.climb_mps) == -1.0
    assert ev._altitude_guard(100.0 + LOW - 1.0, -2.0) == 1.0
    assert ev._altitude_guard(100.0 + 20.0, 3.5) == 3.5
    # ...and the same absolute height is fine 100 m lower down.
    assert Evader("climb_flee")._altitude_guard(100.0 + 20.0, 3.5) == -3.0


# -- arena guard, in isolation ----------------------------------------------


def test_arena_guard_is_a_no_op_well_inside_the_arena():
    ev = Evader("straight")
    inner = CFG.arena_radius_m - CFG.arena_margin_m  # 65 m
    for rr in (0.0, 1.0, 40.0, inner):
        assert ev._arena_guard((rr, 0.0, 20.0), 0.6, -0.8) == (0.6, -0.8)


def test_arena_guard_is_fully_inward_at_and_beyond_the_edge():
    ev = Evader("straight")
    # Heading straight out along +y at the +x edge: u saturates at 1, so the
    # policy's own heading is discarded entirely.
    assert ev._arena_guard((90.0, 0.0, 20.0), 0.0, 1.0) == (-1.0, 0.0)
    assert ev._arena_guard((150.0, 0.0, 20.0), 0.0, 1.0) == (-1.0, 0.0)
    out = ev._arena_guard((0.0, -90.0, 20.0), 1.0, 0.0)
    assert out == pytest.approx((0.0, 1.0), abs=1e-12)


def test_arena_guard_blend_is_linear_in_the_margin():
    ev = Evader("straight")
    # Tangential heading (0, 1) at radius rr on the +x axis: the blend is
    # unit((-u, 1 - u)) with u = (rr - 65) / 25.
    for rr in (70.0, 77.5, 85.0):
        u = (rr - 65.0) / 25.0
        n = math.hypot(u, 1.0 - u)
        got = ev._arena_guard((rr, 0.0, 20.0), 0.0, 1.0)
        assert got == pytest.approx((-u / n, (1.0 - u) / n), abs=1e-12)


def test_arena_guard_is_relative_to_centre_xy():
    ev = Evader("straight", centre_xy=(100.0, -50.0))
    assert ev._arena_guard((100.0 + 40.0, -50.0, 20.0), 0.6, -0.8) == (0.6, -0.8)
    u = 0.2
    n = math.hypot(u, 1.0 - u)
    got = ev._arena_guard((100.0 + 70.0, -50.0, 20.0), 0.0, 1.0)
    assert got == pytest.approx((-u / n, (1.0 - u) / n), abs=1e-12)


def test_arena_guard_turns_smoothly_when_the_heading_is_not_radial():
    """The docstring's claim: a blend, not a reflection -- no corner in the path."""
    ev = Evader("straight")
    h = (math.cos(math.radians(45.0)), math.sin(math.radians(45.0)))
    angles = []
    rr = 60.0
    while rr <= 95.0:
        gx, gy = ev._arena_guard((rr, 0.0, 20.0), *h)
        angles.append(math.atan2(gy, gx))
        rr += 0.1
    steps = [abs(math.degrees(b - a)) for a, b in zip(angles, angles[1:])]
    assert max(steps) < 2.0


def test_arena_guard_reverses_instead_of_turning_when_the_heading_is_radial():
    """SUSPECTED BUG: the blend degenerates for an exactly outward heading.

    ``_arena_guard`` blends the policy heading with the inward unit vector. When
    the policy is flying exactly radially outward the two are anti-parallel, so
    the blend has no lateral component to turn into: it shrinks to zero at
    ``u == 0.5`` and flips sign. That is a hard 180 degree reflection -- exactly
    the "corner in the flight path that is not something an aircraft does" the
    method's own docstring says it was written to avoid -- and at the crossing
    point ``_unit_xy`` falls back to world ``+x`` (here: *outward*) because the
    blended vector's norm is below its 1e-9 guard.

    Asserted as the code behaves today, not as documented.
    """
    ev = Evader("straight")
    assert ev._arena_guard((77.4, 0.0, 20.0), 1.0, 0.0) == (1.0, 0.0)
    assert ev._arena_guard((77.5, 0.0, 20.0), 1.0, 0.0) == (1.0, 0.0)  # u == 0.5
    assert ev._arena_guard((77.6, 0.0, 20.0), 1.0, 0.0) == (-1.0, 0.0)


def test_radial_flee_bounces_along_its_own_track_and_stops_dead():
    """SUSPECTED BUG (closed-loop consequence of the reflection above).

    Fleeing a stationary chaser sitting at the arena centre is exactly radial,
    so the guard never produces a lateral component: the evader has no turn to
    fly, it decelerates to a complete stop at 81.75 m and reverses back down the
    same line. Containment still holds, which is why this is a fidelity problem
    rather than a scenario-validity one -- but a real aircraft does not hover.
    """
    traj = _fly("flee", start=(30.0, 0.0, 20.0), chaser0=(0.0, 0.0, 20.0),
                chaser_speed=0.0, secs=30.0)
    assert max(abs(p[1]) for p in traj) == 0.0          # never turns at all
    radii = [_radius(p) for p in traj]
    assert max(radii) == pytest.approx(81.75, abs=1e-3)
    assert max(radii) - min(radii) > 5.0                # bang-bang, not a curve
    speeds = [math.dist(a, b) / DT for a, b in zip(traj, traj[1:])]
    assert min(speeds) == pytest.approx(0.0, abs=1e-9)  # dead stop mid-arena


# -- per-policy desired velocity --------------------------------------------


def test_straight_holds_its_heading_and_ignores_the_chaser():
    ev = make_evader("straight", 0, 0.0, CFG, heading0=math.radians(30.0))
    want = (CFG.speed * math.cos(math.radians(30.0)),
            CFG.speed * math.sin(math.radians(30.0)), 0.0)
    for t in (0.0, 1.7, 40.0):
        for chaser in ((0.0, 0.0, 20.0), (-50.0, 30.0, 5.0), (10.0, 0.0, 20.0)):
            assert ev.desired_velocity(t, (10.0, 0.0, 20.0), chaser) == \
                pytest.approx(want, abs=1e-12)


def test_flee_runs_exactly_down_the_line_of_sight():
    ev = make_evader("flee", 0, 0.0, CFG, heading0=0.0)
    v = ev.desired_velocity(0.0, (30.0, 40.0, 20.0), (0.0, 0.0, 20.0))
    assert v == pytest.approx((CFG.speed * 0.6, CFG.speed * 0.8, 0.0), abs=1e-12)
    # Altitude is not part of "away": the horizontal bearing is all that counts.
    v2 = ev.desired_velocity(0.0, (30.0, 40.0, 20.0), (0.0, 0.0, -400.0))
    assert v2 == pytest.approx(v, abs=1e-12)


def test_flee_falls_back_to_world_x_when_the_chaser_is_co_located():
    ev = make_evader("flee", 0, 0.0, CFG, heading0=2.0)
    v = ev.desired_velocity(0.0, (10.0, 10.0, 20.0), (10.0, 10.0, 20.0))
    assert v == pytest.approx((CFG.speed, 0.0, 0.0), abs=1e-12)


def test_weave_slides_across_the_line_of_sight_and_crosses_it_twice():
    ev = make_evader("weave", 0, 0.0, CFG, heading0=0.0)
    own, chaser = (30.0, 0.0, 20.0), (0.0, 0.0, 20.0)
    p = CFG.weave_period_s

    # Zero crossings: on the line of sight at t = 0 and t = period / 2.
    assert ev.desired_velocity(0.0, own, chaser) == \
        pytest.approx((CFG.speed, 0.0, 0.0), abs=1e-9)
    assert ev.desired_velocity(p / 2.0, own, chaser) == \
        pytest.approx((CFG.speed, 0.0, 0.0), abs=1e-9)

    # Extremes: away + amplitude * (left-of-away), normalised. Positive first,
    # because the lateral axis is +90 degrees from the run-away direction.
    n = math.hypot(1.0, CFG.weave_amplitude)
    peak = (CFG.speed / n, CFG.speed * CFG.weave_amplitude / n, 0.0)
    assert ev.desired_velocity(p / 4.0, own, chaser) == pytest.approx(peak, abs=1e-9)
    assert ev.desired_velocity(3.0 * p / 4.0, own, chaser) == \
        pytest.approx((peak[0], -peak[1], 0.0), abs=1e-9)

    # The whole excursion is 2 * atan(0.8) ~ 77.3 degrees across the LOS.
    off = math.degrees(math.atan2(peak[1], peak[0]))
    assert off == pytest.approx(38.6598, abs=1e-3)


def test_weave_keeps_cruise_speed_throughout():
    ev = make_evader("weave", 0, 0.0, CFG, heading0=0.0)
    for i in range(120):
        v = ev.desired_velocity(i * 0.05, (30.0, 0.0, 20.0), (0.0, 0.0, 20.0))
        assert math.hypot(v[0], v[1]) == pytest.approx(CFG.speed, abs=1e-12)


def test_climb_flee_flees_horizontally_and_climbs_at_climb_mps():
    ev = make_evader("climb_flee", 0, 0.0, CFG, heading0=0.0)
    v = ev.desired_velocity(0.0, (30.0, 40.0, 20.0), (0.0, 0.0, 20.0))
    assert v == pytest.approx((CFG.speed * 0.6, CFG.speed * 0.8, CFG.climb_mps),
                              abs=1e-12)
    # ...but only while it is inside the band.
    v_high = ev.desired_velocity(0.0, (30.0, 40.0, HIGH + 2.0), (0.0, 0.0, 20.0))
    assert v_high[2] == pytest.approx(-2.0, abs=1e-12)


def test_orbit_is_purely_tangential_at_the_nominal_radius():
    for seed in range(6):
        ev = make_evader("orbit", seed, 0.0, CFG, heading0=0.0,
                         centre_xy=(0.0, 0.0))
        v = ev.desired_velocity(0.0, (CFG.orbit_radius_m, 0.0, 20.0),
                                (0.0, 0.0, 20.0))
        assert v[0] == pytest.approx(0.0, abs=1e-12)          # no radial part
        assert v[1] == pytest.approx(CFG.speed * ev._orbit_sign, abs=1e-12)
        assert v[2] == 0.0


def test_orbit_uses_its_heading_at_the_exact_centre():
    ev = make_evader("orbit", 0, 0.0, CFG, heading0=math.radians(30.0),
                     centre_xy=(5.0, -5.0))
    v = ev.desired_velocity(0.0, (5.0, -5.0, 20.0), (0.0, 0.0, 20.0))
    assert v == pytest.approx((CFG.speed * math.cos(math.radians(30.0)),
                               CFG.speed * math.sin(math.radians(30.0)), 0.0),
                              abs=1e-12)


def test_orbit_radial_correction_points_the_wrong_way():
    """SUSPECTED BUG: the sign of the radius-hold term is inverted.

    The comment says "steer back toward the nominal radius so the circle is
    stable", but ``err = (R - rr) / R`` is then applied as ``-err`` along the
    *outward* radial unit vector. Inside the nominal radius that pushes further
    in; outside it pushes further out. ``orbit_radius_m`` is therefore an
    unstable equilibrium and the policy never flies the circle it advertises:
    from 15 m it spirals in to ~1.8 m, and from 30 m it spirals out until the
    arena guard stops it near 78 m.

    Asserted as the code behaves today.
    """
    ev = make_evader("orbit", 4, 0.0, CFG, heading0=0.0, centre_xy=(0.0, 0.0))
    assert ev._orbit_sign == 1.0

    # Inside the nominal 25 m: the radial component is *negative* (inward).
    inside = ev.desired_velocity(0.0, (15.0, 0.0, 20.0), (0.0, 0.0, 20.0))
    assert inside[0] == pytest.approx(-5.622255, abs=1e-6)
    assert inside[0] < 0.0

    # Outside it: *positive* (outward), so it runs away from the circle.
    outside = ev.desired_velocity(0.0, (40.0, 0.0, 20.0), (0.0, 0.0, 20.0))
    assert outside[0] == pytest.approx(6.913992, abs=1e-6)
    assert outside[0] > 0.0

    # And the closed loop confirms it: started on the nominal circle, it leaves.
    traj = _fly("orbit", seed=4, start=(CFG.orbit_radius_m, 0.0, 20.0),
                chaser_speed=0.0, secs=40.0)
    assert _radius(traj[-1]) > 3.0 * CFG.orbit_radius_m


# -- break_turn -------------------------------------------------------------


def test_break_turn_flies_straight_until_the_trigger_range():
    ev = make_evader("break_turn", 3, 0.0, CFG, heading0=0.0)
    just_outside = CFG.break_trigger_m + 0.001
    v = ev.desired_velocity(0.0, (30.0, 0.0, 20.0),
                            (30.0 - just_outside, 0.0, 20.0))
    assert v == pytest.approx((CFG.speed, 0.0, 0.0), abs=1e-12)
    assert ev._break_started is None


def test_break_turn_commits_across_the_line_of_sight_inside_the_trigger():
    for seed in (0, 1, 2, 3):
        ev = make_evader("break_turn", seed, 0.0, CFG, heading0=0.0)
        sign = ev._break_sign
        # Chaser dead astern at 15 m: away = +x, so the break is along +/- y.
        v = ev.desired_velocity(0.0, (30.0, 0.0, 20.0), (15.0, 0.0, 20.0))
        assert v == pytest.approx((0.0, CFG.speed * sign, 0.0), abs=1e-12)
        assert ev._break_started == 0.0
        # Exactly at the trigger range it also commits (the test is `<=`).
        ev2 = make_evader("break_turn", seed, 0.0, CFG, heading0=0.0)
        ev2.desired_velocity(0.0, (30.0, 0.0, 20.0),
                             (30.0 - CFG.break_trigger_m, 0.0, 20.0))
        assert ev2._break_started == 0.0


def test_break_turn_is_one_shot_and_resumes_the_original_heading():
    ev = make_evader("break_turn", 3, 0.0, CFG, heading0=0.0)
    close = (15.0, 0.0, 20.0)
    own = (30.0, 0.0, 20.0)
    ev.desired_velocity(4.0, own, close)
    assert ev._break_started == 4.0
    # Held for exactly break_hold_s...
    assert ev.desired_velocity(4.0 + CFG.break_hold_s, own, close)[1] != \
        pytest.approx(0.0, abs=1e-6)
    # ...then straight again, and it never re-arms however close the chaser gets.
    for t in (4.0 + CFG.break_hold_s + 1e-6, 20.0, 300.0):
        assert ev.desired_velocity(t, own, (29.0, 0.0, 20.0)) == \
            pytest.approx((CFG.speed, 0.0, 0.0), abs=1e-12)
    assert ev._break_started == 4.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_break_turn_actually_turns_the_airframe(seed):
    """Flown, not just commanded: the aircraft leaves its original track."""
    # heading0 must be pinned here too: leaving it None consumes an rng draw
    # first and lands on a different break sign than the flight below.
    expected_sign = make_evader("break_turn", seed, 0.0, CFG,
                                heading0=0.0)._break_sign

    # Chaser 18 m dead astern -- inside break_trigger_m, so it commits at t = 0.
    traj = _fly("break_turn", seed=seed, secs=CFG.break_hold_s,
                start=(30.0, 0.0, 20.0), chaser0=(12.0, 0.0, 20.0),
                chaser_speed=0.0, vel0=(CFG.speed, 0.0, 0.0))
    final = traj[-1]
    heading = math.degrees(math.atan2(final[1] - traj[-2][1],
                                      final[0] - traj[-2][0]))
    assert math.copysign(1.0, heading) == expected_sign
    assert abs(heading) > 90.0                     # turned through a right angle
    assert abs(final[1]) > 10.0                    # and is well off its old track
    assert final[1] == pytest.approx(15.4964 * expected_sign, abs=1e-3)
    assert final[0] == pytest.approx(30.5446, abs=1e-3)

    # Control: same seed, chaser 45 m away -- never triggers, dead straight.
    straight = _fly("break_turn", seed=seed, secs=CFG.break_hold_s,
                    start=(30.0, 0.0, 20.0), chaser0=(-15.0, 0.0, 20.0),
                    chaser_speed=0.0, vel0=(CFG.speed, 0.0, 0.0))
    assert max(abs(p[1]) for p in straight) == 0.0
    assert straight[-1][0] > final[0]


# -- jink -------------------------------------------------------------------


def test_jink_changes_heading_on_the_scheduled_random_interval():
    ev = make_evader("jink", 5, 0.0, CFG, heading0=0.0)
    changes = []
    headings = [ev.heading]
    prev = ev.heading
    for i in range(400):                                    # 20 s at 50 ms
        t = i * DT
        ev.desired_velocity(t, (10.0, 0.0, 20.0), (0.0, 0.0, 20.0))
        if ev.heading != prev:
            changes.append(t)
            headings.append(ev.heading)
            prev = ev.heading

    assert changes[0] == 0.0                                # jinks immediately
    # 20 s of intervals drawn from [0.8, 2.0] -> between 10 and 25 of them.
    assert 10 <= len(changes) <= 26
    gaps = [b - a for a, b in zip(changes, changes[1:])]
    assert gaps, "expected more than one jink in 20 s"
    for g in gaps:
        assert CFG.jink_min_s - DT < g < CFG.jink_max_s + DT
    for a, b in zip(headings, headings[1:]):
        assert abs(b - a) <= 2.2 + 1e-12


def test_jink_holds_a_constant_heading_between_changes():
    ev = make_evader("jink", 5, 0.0, CFG, heading0=0.0)
    own, chaser = (10.0, 0.0, 20.0), (0.0, 0.0, 20.0)
    ev.desired_velocity(0.0, own, chaser)
    held = ev.heading
    want = (CFG.speed * math.cos(held), CFG.speed * math.sin(held), 0.0)
    for i in range(1, 15):                                  # 0.7 s < jink_min_s
        v = ev.desired_velocity(i * DT, own, chaser)
        assert v == pytest.approx(want, abs=1e-12)
    assert ev.heading == held


# -- command() --------------------------------------------------------------


def test_command_is_a_world_frame_velocity_labelled_with_its_policy():
    ev = make_evader("flee", 0, 0.0, CFG, heading0=0.0)
    frame = Airframe(xyz=(30.0, 40.0, 20.0), yaw=0.0, ground_z=0.0)
    cmd = ev.command(0.0, frame, (0.0, 0.0, 20.0))
    assert isinstance(cmd, BodyCommand)
    assert cmd.frame == "world"
    assert cmd.source == "evader:flee"
    want = ev.desired_velocity(0.0, frame.xyz, (0.0, 0.0, 20.0))
    assert (cmd.vx, cmd.vy, cmd.vz) == pytest.approx(want, abs=1e-12)


def test_command_yaw_rate_takes_the_short_way_round():
    # Wanted heading +3.0 rad, current -3.0 rad: the naive difference is 6.0 rad
    # the long way; the short way is 6.0 - 2*pi = -0.283 rad.
    ev = make_evader("straight", 0, 0.0, CFG, heading0=3.0)
    frame = Airframe(xyz=(10.0, 0.0, 20.0), yaw=-3.0, ground_z=0.0)
    cmd = ev.command(0.0, frame, (0.0, 0.0, 20.0))
    assert cmd.yaw_rate == pytest.approx(2.0 * (6.0 - 2.0 * math.pi), abs=1e-12)
    assert cmd.yaw_rate < 0.0


def test_command_puts_the_nose_on_the_velocity_vector():
    ev = make_evader("straight", 0, 0.0, CFG, heading0=math.radians(40.0))
    for yaw in (0.0, math.radians(40.0), -1.2):
        frame = Airframe(xyz=(10.0, 0.0, 20.0), yaw=yaw, ground_z=0.0)
        cmd = ev.command(0.0, frame, (0.0, 0.0, 20.0))
        err = math.atan2(cmd.vy, cmd.vx) - yaw
        err = (err + math.pi) % (2.0 * math.pi) - math.pi
        assert cmd.yaw_rate == pytest.approx(2.0 * err, abs=1e-12)
    # Aligned already -> no yaw demand at all.
    frame = Airframe(xyz=(10.0, 0.0, 20.0), yaw=math.radians(40.0), ground_z=0.0)
    assert ev.command(0.0, frame, (0.0, 0.0, 20.0)).yaw_rate == \
        pytest.approx(0.0, abs=1e-12)


def test_command_holds_yaw_when_there_is_no_horizontal_velocity():
    ev = Evader("straight", EvaderConfig(speed=0.0), seed=0, heading0=0.0)
    frame = Airframe(xyz=(10.0, 0.0, 20.0), yaw=0.7, ground_z=0.0)
    cmd = ev.command(0.0, frame, (0.0, 0.0, 20.0))
    assert (cmd.vx, cmd.vy, cmd.vz) == (0.0, 0.0, 0.0)
    assert cmd.yaw_rate == 0.0


# -- containment, closed loop -----------------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
def test_policy_stays_inside_the_arena(policy):
    """No policy escapes ``arena_radius_m``, chased for 90 s from three seeds."""
    for seed in (0, 3, 5):
        traj = _fly(policy, seed=seed, secs=90.0)
        worst = max(_radius(p) for p in traj)
        assert worst <= CFG.arena_radius_m, (
            f"{policy}/seed {seed} reached {worst:.2f} m against a "
            f"{CFG.arena_radius_m} m arena")
        # Liveness: it really flew rather than parking somewhere legal. Measured
        # as *motion*, not ground covered -- `evasive` deliberately crosses back
        # and forth across the line of sight instead of running anywhere, so it
        # travels less far while manoeuvring harder than anything else here.
        assert _path_length(traj) > 200.0


@pytest.mark.parametrize("policy", POLICIES)
def test_policy_stays_inside_the_altitude_band(policy):
    """REGRESSION: no policy leaves the band, and ``climb_flee`` least of all.

    The additive guard let ``climb_flee`` reach 68 m against this 45 m ceiling.
    """
    for seed in (0, 3, 5):
        traj = _fly(policy, seed=seed, secs=90.0, start=(30.0, 0.0, 20.0))
        zs = [p[2] for p in traj]
        assert min(zs) >= LOW - 1e-9, f"{policy}/seed {seed} sank to {min(zs):.2f} m"
        assert max(zs) <= HIGH + CEILING_SLOP_M, (
            f"{policy}/seed {seed} climbed to {max(zs):.2f} m against a "
            f"{HIGH} m ceiling")


def test_climb_flee_is_pinned_to_the_ceiling_not_merely_slowed():
    """REGRESSION, the exact case that broke: 3.5 m/s climb vs a 45 m ceiling.

    A bounded additive push-back would leave a net +0.5 m/s here, i.e. about
    +30 m over the 60 s below and ~68 m in a full episode. The replacement guard
    caps the overshoot at the vertical stopping distance and settles *on* the
    ceiling.
    """
    traj = _fly("climb_flee", seed=1, secs=60.0, start=(30.0, 0.0, 20.0))
    zs = [p[2] for p in traj]
    assert max(zs) <= HIGH + CEILING_SLOP_M
    # The invariant is a *bound*: the ceiling is overshot by the vertical
    # stopping distance and no more. The exact value moves with the
    # acceleration limit, so pinning it would break on any retune; the
    # bound is what the additive-guard bug violated by ~23 m.
    lim = evader_limits(CFG)
    over = lim.max_speed_z ** 2 / (2.0 * lim.max_accel_z) + lim.max_speed_z * DT
    assert HIGH <= max(zs) <= HIGH + over + 0.1
    assert zs[-1] == pytest.approx(HIGH, abs=0.05)       # settles, does not drift
    # It did climb -- the guard is capping an active climb, not a dead policy.
    assert max(zs) - zs[0] > 25.0


def test_climb_flee_holds_the_ceiling_for_a_long_run_at_high_ground():
    """The band is relative to ``ground_z``, and it still holds after 3 minutes."""
    traj = _fly("climb_flee", seed=2, secs=180.0, ground_z=100.0,
                start=(30.0, 0.0, 120.0), chaser0=(0.0, 0.0, 120.0))
    zs = [p[2] for p in traj]
    assert max(zs) <= 100.0 + HIGH + CEILING_SLOP_M
    assert min(zs) >= 100.0 + LOW - 1e-9
    assert zs[-1] == pytest.approx(100.0 + HIGH, abs=0.05)


def test_evader_climbs_back_into_the_band_from_below_without_overshooting():
    traj = _fly("straight", seed=0, secs=60.0, start=(20.0, 0.0, 4.0))
    zs = [p[2] for p in traj]
    assert zs[0] > 4.0                       # starts climbing on the first tick
    assert max(zs) <= LOW + 1e-6             # approaches the floor from below
    assert zs[-1] == pytest.approx(LOW, abs=1e-6)


def test_evader_descends_back_into_the_band_from_above():
    traj = _fly("climb_flee", seed=0, secs=60.0, start=(20.0, 0.0, 60.0),
                chaser0=(0.0, 0.0, 60.0))
    zs = [p[2] for p in traj]
    assert max(zs) <= 60.0                   # never climbs above where it began
    assert min(zs) >= HIGH - 1e-6            # approaches the ceiling from above
    assert zs[-1] == pytest.approx(HIGH, abs=1e-6)


def test_arena_guard_is_what_stops_a_flee_and_it_stops_it_short_of_the_wall():
    traj = _fly("flee", seed=0, secs=90.0)
    radii = [_radius(p) for p in traj]
    # It really does run to the edge region (the guard is exercised)...
    assert max(radii) > CFG.arena_radius_m - CFG.arena_margin_m
    # ...and is turned back with room to spare rather than bouncing off a wall.
    assert max(radii) <= CFG.arena_radius_m
    assert radii[-1] < CFG.arena_radius_m


def test_arena_containment_holds_about_an_offset_centre():
    centre = (250.0, -120.0)
    traj = _fly("flee", seed=0, secs=90.0, centre_xy=centre,
                start=(centre[0] + 30.0, centre[1], 20.0),
                chaser0=(centre[0], centre[1], 20.0))
    assert max(_radius(p, centre) for p in traj) <= CFG.arena_radius_m


def test_containment_holds_for_a_tighter_arena_and_a_faster_climber():
    """Containment is a property of the guards, not of one tuned config."""
    cfg = EvaderConfig(speed=14.0, climb_mps=6.0, arena_radius_m=60.0,
                       arena_margin_m=30.0, altitude_band=(10.0, 25.0))
    for policy in POLICIES:
        for seed in (0, 2, 5):
            traj = _fly(policy, seed=seed, cfg=cfg, secs=90.0,
                        start=(15.0, 0.0, 15.0), chaser0=(0.0, 0.0, 15.0),
                        chaser_speed=13.0)
            assert max(_radius(p) for p in traj) <= cfg.arena_radius_m
            zs = [p[2] for p in traj]
            assert min(zs) >= cfg.altitude_band[0] - 1e-9
            # Overshoot is still just the vertical stopping distance,
            # 6^2 / (2 * 5) = 3.6 m, plus a tick.
            assert max(zs) <= cfg.altitude_band[1] + 4.0


def test_arena_radius_is_not_honoured_when_the_margin_is_under_the_turn_radius():
    """SUSPECTED BUG: ``arena_radius_m`` is only kept for a wide enough margin.

    ``arena_margin_m`` is the distance the guard has to turn the aircraft
    around, so it has to be at least the minimum turn radius ``v^2 / a``. The
    defaults are safe (8.1 m of turn radius against a 25 m margin), but nothing
    checks the relationship, and a config that violates it leaves the arena
    silently: at 14 m/s the turn radius is 19.6 m, so a 12 m margin puts the
    evader 4 m outside a 40 m arena.

    Asserted as the code behaves today: this test documents the precondition,
    it does not endorse it.
    """
    cfg = EvaderConfig(speed=14.0, climb_mps=6.0, arena_radius_m=40.0,
                       arena_margin_m=12.0, altitude_band=(10.0, 25.0))
    turn_radius = cfg.speed ** 2 / evader_limits(cfg).max_accel_xy
    assert turn_radius > cfg.arena_margin_m               # the precondition fails

    traj = _fly("climb_flee", seed=2, cfg=cfg, secs=90.0,
                start=(15.0, 0.0, 15.0), chaser0=(0.0, 0.0, 15.0),
                chaser_speed=13.0)
    worst = max(_radius(p) for p in traj)
    assert worst > cfg.arena_radius_m
    # Bounded by the turn radius, not pinned to a digit: the exact overshoot
    # moves whenever the airframe limits are retuned, and what this test
    # documents is the *precondition*, not one number produced by it.
    assert worst <= cfg.arena_radius_m + turn_radius
    # The altitude band is unaffected -- that guard is a replacement, not a turn.
    assert max(p[2] for p in traj) <= cfg.altitude_band[1] + 4.0


# -- determinism ------------------------------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
def test_same_seed_gives_a_bit_identical_trajectory(policy):
    a = _fly(policy, seed=7, secs=30.0)
    b = _fly(policy, seed=7, secs=30.0)
    assert a == b


@pytest.mark.parametrize("policy", POLICIES)
def test_seeding_survives_interleaved_evaders(policy):
    """A second evader drawing from its own rng must not perturb the first."""
    solo = _fly(policy, seed=7, secs=30.0)

    ev_a = make_evader(policy, 7, 0.0, CFG, heading0=0.0, centre_xy=(0.0, 0.0))
    ev_b = make_evader("jink", 99, 0.0, CFG, heading0=0.0, centre_xy=(0.0, 0.0))
    frame = Airframe(xyz=(30.0, 0.0, 20.0), yaw=0.0, ground_z=0.0)
    frame.limits = evader_limits(CFG)
    chaser = [0.0, 0.0, 20.0]
    traj = []
    t = 0.0
    for _ in range(int(round(30.0 / DT))):
        ev_b.desired_velocity(t, (5.0, 5.0, 20.0), (0.0, 0.0, 20.0))
        frame.step(ev_a.command(t, frame, tuple(chaser)), DT)
        traj.append(frame.xyz)
        sep = math.dist(frame.xyz, chaser) or 1.0
        for k in range(3):
            chaser[k] += 8.0 * DT * (frame.xyz[k] - chaser[k]) / sep
        t += DT
    assert traj == solo


def test_different_seeds_diverge_for_jink():
    """``jink`` is the policy whose randomness is the whole point."""
    a = _fly("jink", seed=1, secs=30.0)
    c = _fly("jink", seed=2, secs=30.0)
    assert a != c
    assert max(math.dist(p, q) for p, q in zip(a, c)) > 20.0


def test_different_seeds_diverge_through_the_random_initial_heading():
    a = _fly("straight", seed=1, secs=30.0, heading0=None)
    c = _fly("straight", seed=2, secs=30.0, heading0=None)
    assert max(math.dist(p, q) for p, q in zip(a, c)) > 20.0


def test_seed_selects_the_break_and_orbit_turn_directions():
    signs = {(make_evader("break_turn", s, 0.0, CFG)._break_sign,
              make_evader("orbit", s, 0.0, CFG)._orbit_sign) for s in range(16)}
    assert len(signs) > 1, "the seed must pick a turn direction, not a constant"
    for b, o in signs:
        assert b in (1.0, -1.0) and o in (1.0, -1.0)
