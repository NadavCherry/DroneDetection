"""Tests for :mod:`pursuit.dynamics`.

The airframe exists so that guidance is tested against something that has to
obey physics, which only means anything if the limits are actually enforced --
every tick, in every frame, and without quietly rotating the command on the way
through. That is what is checked here:

* the four rate limits (speed, acceleration, yaw rate, yaw acceleration) hold
  over long runs of adversarial commands, not just on the first tick;
* :func:`~pursuit.dynamics._limit_norm` scales, it does not clip -- a diagonal
  command must not accelerate ``sqrt(2)`` times harder than an axis one and a
  saturated command must come out pointing where it went in;
* ``frame="body"`` and ``frame="world"`` describe the same motion when the body
  command is the world one expressed in the *current* heading, and the body
  command is resolved through the pre-rotation yaw (the failure the
  :class:`~pursuit.dynamics.BodyCommand` docstring is about);
* the ground floor clamps.

Everything is deterministic: the "random" command streams come from a seeded
``random.Random``, so a failure here is reproducible.
"""
from __future__ import annotations

import math
import random

import pytest

from pursuit.dynamics import (
    Airframe,
    BodyCommand,
    Limits,
    _clamp,
    _limit_norm,
    chaser_limits,
)
from pursuit.geometry import body_to_world, wrap_pi

DT = 0.05  # 20 Hz, the control rate the loop actually runs at
TOL = 1e-12


def _airframe(**kw) -> Airframe:
    """An airframe far from either altitude bound, so the band never confuses a
    limit test. The floor and the ceiling are both real and both clamp velocity;
    a test about acceleration or yaw wants neither of them in play."""
    kw.setdefault("xyz", (0.0, 0.0, 500.0))
    kw.setdefault("ground_z", -10_000.0)
    kw.setdefault("limits", Limits(max_agl=1e9))
    return Airframe(**kw)


def _heading(af: Airframe) -> float:
    return math.atan2(af.vel[1], af.vel[0])


# --------------------------------------------------------------------------
# helpers: _clamp / _limit_norm
# --------------------------------------------------------------------------

def test_clamp_bounds_and_passthrough():
    assert _clamp(0.5, -1.0, 1.0) == 0.5
    assert _clamp(-3.0, -1.0, 1.0) == -1.0
    assert _clamp(3.0, -1.0, 1.0) == 1.0
    assert _clamp(-1.0, -1.0, 1.0) == -1.0
    assert _clamp(1.0, -1.0, 1.0) == 1.0
    assert isinstance(_clamp(2, -1.0, 5.0), float)


def test_limit_norm_passes_through_inside_the_limit():
    assert _limit_norm(3.0, 4.0, 5.0) == (3.0, 4.0)
    assert _limit_norm(3.0, 4.0, 5.5) == (3.0, 4.0)
    assert _limit_norm(0.0, 0.0, 0.0) == (0.0, 0.0)


def test_limit_norm_scales_to_the_limit_and_keeps_direction():
    x, y = _limit_norm(3.0, 4.0, 2.5)
    assert math.hypot(x, y) == pytest.approx(2.5, abs=TOL)
    # direction preserved exactly: (3, 4) -> (1.5, 2.0)
    assert x == pytest.approx(1.5, abs=TOL)
    assert y == pytest.approx(2.0, abs=TOL)
    assert math.atan2(y, x) == pytest.approx(math.atan2(4.0, 3.0), abs=TOL)


def test_limit_norm_is_isotropic_not_componentwise():
    """A diagonal and an axis command clamp to the *same* magnitude."""
    lim = 1.0
    axis = _limit_norm(100.0, 0.0, lim)
    diag = _limit_norm(100.0, 100.0, lim)
    assert math.hypot(*axis) == pytest.approx(lim, abs=TOL)
    assert math.hypot(*diag) == pytest.approx(lim, abs=TOL)
    # componentwise clipping would give hypot(1, 1) = sqrt(2) here
    assert math.hypot(*diag) < math.sqrt(2.0) * lim - 0.4


@pytest.mark.parametrize("x,y", [(10.0, 1.0), (-7.0, 3.5), (0.25, -9.0), (-2.0, -2.0)])
def test_limit_norm_never_rotates_the_command(x, y):
    for lim in (0.1, 1.0, 3.0):
        lx, ly = _limit_norm(x, y, lim)
        assert math.atan2(ly, lx) == pytest.approx(math.atan2(y, x), abs=1e-14)
        assert math.hypot(lx, ly) <= lim + TOL


def test_limit_norm_leaves_a_degenerate_vector_alone():
    assert _limit_norm(0.0, 0.0, 1.0) == (0.0, 0.0)
    # below the 1e-12 guard the vector is returned untouched rather than
    # divided by a near-zero norm
    assert _limit_norm(1e-13, 0.0, 1e-30) == (1e-13, 0.0)


# --------------------------------------------------------------------------
# BodyCommand / Limits / chaser_limits
# --------------------------------------------------------------------------

def test_body_command_defaults_to_the_body_frame():
    c = BodyCommand()
    assert c.as_tuple() == (0.0, 0.0, 0.0, 0.0)
    assert c.frame == "body"
    assert c.source == ""
    assert BodyCommand(1.0, 2.0, 3.0, 4.0).as_tuple() == (1.0, 2.0, 3.0, 4.0)


def test_chaser_limits_scales_speed_and_accel_only():
    base = Limits()
    fast = chaser_limits(1.5, base)
    assert fast.max_speed_xy == pytest.approx(1.5 * base.max_speed_xy, abs=TOL)
    assert fast.max_speed_z == pytest.approx(1.5 * base.max_speed_z, abs=TOL)
    assert fast.max_accel_xy == pytest.approx(1.5 * base.max_accel_xy, abs=TOL)
    assert fast.max_accel_z == pytest.approx(1.5 * base.max_accel_z, abs=TOL)
    # the attitude limits and the floor are deliberately not scaled
    assert fast.max_yaw_rate == base.max_yaw_rate
    assert fast.max_yaw_accel == base.max_yaw_accel
    assert fast.min_agl == base.min_agl
    assert chaser_limits(1.0, base) == base
    assert base.max_speed_xy == 14.0  # base untouched


# --------------------------------------------------------------------------
# integration bookkeeping
# --------------------------------------------------------------------------

def test_post_init_coerces_state_to_float():
    af = Airframe(xyz=(1, 2, 3), yaw=0, vel=(4, 5, 6))
    assert af.xyz == (1.0, 2.0, 3.0)
    assert af.vel == (4.0, 5.0, 6.0)
    assert all(isinstance(v, float) for v in af.xyz + af.vel)
    assert isinstance(af.yaw, float)


@pytest.mark.parametrize("dt", [0.0, -0.05, -1.0])
def test_nonpositive_dt_is_a_no_op(dt):
    af = _airframe(yaw=0.3, vel=(1.0, 2.0, 0.5), yaw_rate=0.4)
    before = (af.xyz, af.yaw, af.vel, af.yaw_rate)
    af.step(BodyCommand(vx=10.0, vy=-4.0, vz=3.0, yaw_rate=2.0, frame="world"), dt)
    assert (af.xyz, af.yaw, af.vel, af.yaw_rate) == before


def test_position_integrates_the_post_update_velocity():
    """Semi-implicit Euler: dx == v_new * dt, exactly, when the floor is far."""
    af = _airframe()
    rng = random.Random(11)
    for _ in range(200):
        p0 = af.xyz
        af.step(BodyCommand(vx=rng.uniform(-30, 30), vy=rng.uniform(-30, 30),
                            vz=rng.uniform(-8, 8), yaw_rate=rng.uniform(-4, 4),
                            frame="world"), DT)
        for k in range(3):
            assert af.xyz[k] - p0[k] == pytest.approx(af.vel[k] * DT, abs=1e-12)


def test_speed_properties_match_the_velocity():
    af = _airframe(vel=(3.0, 4.0, 12.0))
    assert af.speed_xy == pytest.approx(5.0, abs=TOL)
    assert af.speed == pytest.approx(13.0, abs=TOL)


def test_pose_and_snapshot_round_for_the_wire():
    af = _airframe(xyz=(1.234567891, -2.0, 30.5), yaw=0.1234567891,
                   vel=(1.111111, 0.0, -0.5), yaw_rate=0.987654321)
    pose = af.pose()
    assert pose["xyz"] == [1.23457, -2.0, 30.5]
    assert pose["yaw"] == 0.123457
    snap = af.snapshot()
    assert snap["xyz"] == [1.2346, -2.0, 30.5]
    assert snap["yaw"] == 0.12346
    assert snap["vel"] == [1.1111, 0.0, -0.5]
    assert snap["yaw_rate"] == 0.9877
    assert snap["speed"] == pytest.approx(round(af.speed, 3), abs=TOL)


# --------------------------------------------------------------------------
# acceleration limits
# --------------------------------------------------------------------------

def test_first_tick_from_rest_takes_exactly_one_accel_step():
    af = _airframe()
    af.step(BodyCommand(vx=1000.0, vy=0.0, frame="world"), DT)
    assert af.vel[0] == pytest.approx(Limits().max_accel_xy * DT, abs=TOL)
    assert af.vel[1] == 0.0
    assert af.speed_xy == pytest.approx(Limits().max_accel_xy * DT, abs=TOL)


def test_diagonal_command_does_not_accelerate_faster_than_an_axis_one():
    """The whole point of ``_limit_norm``, at the airframe level."""
    axis = _airframe()
    diag = _airframe()
    for _ in range(6):  # still accel-limited, nowhere near the speed ceiling
        axis.step(BodyCommand(vx=1000.0, vy=0.0, frame="world"), DT)
        diag.step(BodyCommand(vx=1000.0, vy=1000.0, frame="world"), DT)
        # same speed gained, whatever direction it was asked for
        assert diag.speed_xy == pytest.approx(axis.speed_xy, abs=1e-12)
        # and the diagonal stayed a diagonal
        assert diag.vel[0] == pytest.approx(diag.vel[1], abs=1e-15)
    assert axis.speed_xy == pytest.approx(6 * Limits().max_accel_xy * DT, abs=1e-12)
    # componentwise clipping would have made the diagonal sqrt(2) faster
    assert diag.speed_xy < math.sqrt(2.0) * axis.speed_xy - 1.0


def test_saturated_command_is_not_rotated():
    """An off-axis command that exceeds the speed limit keeps its direction.

    Componentwise clipping of ``(1000, 100)`` would come out at 45 degrees
    instead of the 5.7 degrees that were asked for.
    """
    af = _airframe()
    want = math.atan2(100.0, 1000.0)
    for _ in range(400):
        af.step(BodyCommand(vx=1000.0, vy=100.0, frame="world"), DT)
        assert _heading(af) == pytest.approx(want, abs=1e-12)
    assert af.speed_xy == pytest.approx(Limits().max_speed_xy, abs=1e-9)


def test_accel_limits_hold_over_a_long_adversarial_run():
    """Sign-flipping, over-limit commands every tick; nothing may exceed a*dt."""
    lim = Limits()
    af = _airframe()
    rng = random.Random(2024)
    worst_xy = worst_z = 0.0
    for k in range(1500):
        cmd = BodyCommand(vx=rng.uniform(-60, 60), vy=rng.uniform(-60, 60),
                          vz=rng.uniform(-30, 30), yaw_rate=rng.uniform(-9, 9),
                          frame="world" if k % 2 else "body")
        v0 = af.vel
        af.step(cmd, DT)
        d_xy = math.hypot(af.vel[0] - v0[0], af.vel[1] - v0[1])
        d_z = abs(af.vel[2] - v0[2])
        assert d_xy <= lim.max_accel_xy * DT + 1e-12
        assert d_z <= lim.max_accel_z * DT + 1e-12
        worst_xy, worst_z = max(worst_xy, d_xy), max(worst_z, d_z)
    # the run must actually have pushed the limits, or it proved nothing
    assert worst_xy == pytest.approx(lim.max_accel_xy * DT, rel=1e-9)
    assert worst_z == pytest.approx(lim.max_accel_z * DT, rel=1e-9)


def test_accel_limit_holds_with_scaled_chaser_limits():
    lim = chaser_limits(1.6)
    af = _airframe(limits=lim)
    rng = random.Random(5)
    for _ in range(400):
        v0 = af.vel
        af.step(BodyCommand(vx=rng.uniform(-80, 80), vy=rng.uniform(-80, 80),
                            vz=rng.uniform(-40, 40), frame="world"), DT)
        assert math.hypot(af.vel[0] - v0[0], af.vel[1] - v0[1]) <= lim.max_accel_xy * DT + 1e-12
        assert abs(af.vel[2] - v0[2]) <= lim.max_accel_z * DT + 1e-12
        assert af.speed_xy <= lim.max_speed_xy + 1e-12


def test_reversal_takes_the_full_accel_time():
    """No teleporting: 14 -> -14 m/s cannot happen faster than 2V/a seconds."""
    lim = Limits()
    af = _airframe()
    for _ in range(200):
        af.step(BodyCommand(vx=1000.0, frame="world"), DT)
    assert af.vel[0] == pytest.approx(lim.max_speed_xy, abs=1e-9)
    ticks = 0
    while af.vel[0] > -lim.max_speed_xy + 1e-6 and ticks < 500:
        af.step(BodyCommand(vx=-1000.0, frame="world"), DT)
        ticks += 1
    expected = (2.0 * lim.max_speed_xy) / (lim.max_accel_xy * DT)
    assert ticks >= math.floor(expected)
    assert ticks <= math.ceil(expected) + 1


# --------------------------------------------------------------------------
# speed limits
# --------------------------------------------------------------------------

def test_horizontal_speed_saturates_at_the_limit_and_never_exceeds_it():
    lim = Limits()
    af = _airframe()
    for _ in range(600):
        af.step(BodyCommand(vx=1000.0, vy=-450.0, frame="world"), DT)
        assert af.speed_xy <= lim.max_speed_xy + 1e-12
    assert af.speed_xy == pytest.approx(lim.max_speed_xy, abs=1e-9)


def test_vertical_speed_ramps_at_the_accel_limit_and_saturates():
    lim = Limits()
    af = _airframe()
    for k in range(1, 6):
        af.step(BodyCommand(vz=1000.0, frame="world"), DT)
        assert af.vel[2] == pytest.approx(k * lim.max_accel_z * DT, abs=1e-12)
    for _ in range(200):
        af.step(BodyCommand(vz=1000.0, frame="world"), DT)
        assert af.vel[2] <= lim.max_speed_z + 1e-12
    assert af.vel[2] == pytest.approx(lim.max_speed_z, abs=1e-12)

    for _ in range(400):
        af.step(BodyCommand(vz=-1000.0, frame="world"), DT)
        assert af.vel[2] >= -lim.max_speed_z - 1e-12
    assert af.vel[2] == pytest.approx(-lim.max_speed_z, abs=1e-12)


def test_horizontal_and_vertical_limits_are_independent():
    """``max_speed_xy`` bounds the horizontal plane only; z has its own ceiling."""
    lim = Limits()
    af = _airframe()
    for _ in range(600):
        af.step(BodyCommand(vx=1000.0, vz=1000.0, frame="world"), DT)
    assert af.speed_xy == pytest.approx(lim.max_speed_xy, abs=1e-9)
    assert af.vel[2] == pytest.approx(lim.max_speed_z, abs=1e-12)
    assert af.speed > lim.max_speed_xy  # total speed is deliberately not capped


def test_sustained_turn_radius_is_the_kinematic_minimum():
    """v^2/a, the number the module docstring claims no guidance law can beat."""
    lim = Limits()
    af = _airframe()
    for _ in range(300):
        af.step(BodyCommand(vx=1000.0, frame="world"), DT)
    headings = []
    for _ in range(400):  # command a hard left turn at full speed, forever
        vx, vy = af.vel[0], af.vel[1]
        n = math.hypot(vx, vy)
        c, s = math.cos(0.2), math.sin(0.2)
        ux, uy = (vx * c - vy * s) / n, (vx * s + vy * c) / n
        af.step(BodyCommand(vx=ux * lim.max_speed_xy, vy=uy * lim.max_speed_xy,
                            frame="world"), DT)
        headings.append(_heading(af))
    tail = [wrap_pi(headings[i + 1] - headings[i]) for i in range(200, len(headings) - 1)]
    omega = (sum(tail) / len(tail)) / DT
    radius = af.speed_xy / omega
    assert radius == pytest.approx(af.speed_xy ** 2 / lim.max_accel_xy, rel=1e-3)
    assert radius >= 0.95 * lim.max_speed_xy ** 2 / lim.max_accel_xy


# --------------------------------------------------------------------------
# yaw rate and yaw acceleration
# --------------------------------------------------------------------------

def test_yaw_rate_ramps_at_the_yaw_accel_limit_then_saturates():
    lim = Limits()
    af = _airframe()
    step = lim.max_yaw_accel * DT          # 0.5 rad/s per tick
    n_ramp = int(round(lim.max_yaw_rate / step))  # 5 ticks to 2.5 rad/s
    for k in range(1, n_ramp + 1):
        prev_yaw = af.yaw
        af.step(BodyCommand(yaw_rate=99.0, frame="world"), DT)
        assert af.yaw_rate == pytest.approx(k * step, abs=1e-12)
        assert wrap_pi(af.yaw - prev_yaw) == pytest.approx(af.yaw_rate * DT, abs=1e-12)
    for _ in range(50):
        af.step(BodyCommand(yaw_rate=99.0, frame="world"), DT)
        assert af.yaw_rate == pytest.approx(lim.max_yaw_rate, abs=1e-12)


def test_yaw_rate_reversal_is_yaw_accel_limited():
    lim = Limits()
    af = _airframe()
    for _ in range(50):
        af.step(BodyCommand(yaw_rate=99.0, frame="world"), DT)
    assert af.yaw_rate == pytest.approx(lim.max_yaw_rate, abs=1e-12)
    ticks = 0
    while af.yaw_rate > -lim.max_yaw_rate + 1e-12 and ticks < 100:
        prev = af.yaw_rate
        af.step(BodyCommand(yaw_rate=-99.0, frame="world"), DT)
        assert prev - af.yaw_rate <= lim.max_yaw_accel * DT + 1e-12
        ticks += 1
    assert ticks == int(round(2 * lim.max_yaw_rate / (lim.max_yaw_accel * DT)))


def test_yaw_limits_hold_over_a_long_adversarial_run():
    lim = Limits()
    af = _airframe()
    rng = random.Random(99)
    worst_rate = worst_accel = 0.0
    for _ in range(1500):
        r0 = af.yaw_rate
        y0 = af.yaw
        af.step(BodyCommand(yaw_rate=rng.uniform(-20, 20), frame="world"), DT)
        assert abs(af.yaw_rate) <= lim.max_yaw_rate + 1e-12
        assert abs(af.yaw_rate - r0) <= lim.max_yaw_accel * DT + 1e-12
        assert abs(wrap_pi(af.yaw - y0)) <= lim.max_yaw_rate * DT + 1e-12
        assert -math.pi < af.yaw <= math.pi
        worst_rate = max(worst_rate, abs(af.yaw_rate))
        worst_accel = max(worst_accel, abs(af.yaw_rate - r0))
    assert worst_rate == pytest.approx(lim.max_yaw_rate, rel=1e-9)
    assert worst_accel == pytest.approx(lim.max_yaw_accel * DT, rel=1e-9)


def test_yaw_stays_wrapped_through_many_full_turns():
    af = _airframe(yaw=3.0)
    total = 0.0
    for _ in range(400):  # ~20 s at 2.5 rad/s is nearly 8 full turns
        prev = af.yaw
        af.step(BodyCommand(yaw_rate=99.0, frame="world"), DT)
        total += wrap_pi(af.yaw - prev)
        assert -math.pi < af.yaw <= math.pi
    assert total > 7 * 2 * math.pi  # it really did keep turning
    assert af.yaw == pytest.approx(wrap_pi(3.0 + total), abs=1e-9)


def test_yaw_rate_command_is_a_rate_not_a_torque_on_translation():
    """Yawing on the spot never moves the aircraft."""
    af = _airframe()
    p0 = af.xyz
    for _ in range(100):
        af.step(BodyCommand(yaw_rate=2.5, frame="world"), DT)
    assert af.xyz == p0
    assert af.vel == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# frames
# --------------------------------------------------------------------------

def test_body_and_world_frames_agree_when_rotated_by_the_current_yaw():
    """The equivalence the whole ``frame`` field exists to make checkable."""
    body_af = Airframe(xyz=(1.0, -2.0, 40.0), yaw=0.7)
    world_af = Airframe(xyz=(1.0, -2.0, 40.0), yaw=0.7)
    for k in range(300):
        bc = BodyCommand(vx=18.0 * math.sin(0.11 * k), vy=12.0 * math.cos(0.07 * k),
                         vz=4.0 * math.sin(0.03 * k), yaw_rate=3.0 * math.sin(0.05 * k),
                         frame="body")
        # rotate by the yaw the aircraft has RIGHT NOW, before the tick turns it
        wx, wy, wz = body_to_world(body_af.yaw, bc.vx, bc.vy, bc.vz)
        wc = BodyCommand(vx=wx, vy=wy, vz=wz, yaw_rate=bc.yaw_rate, frame="world")
        body_af.step(bc, DT)
        world_af.step(wc, DT)
        for a, b in zip(body_af.xyz, world_af.xyz):
            assert a == pytest.approx(b, abs=1e-12)
        for a, b in zip(body_af.vel, world_af.vel):
            assert a == pytest.approx(b, abs=1e-12)
        assert body_af.yaw == pytest.approx(world_af.yaw, abs=1e-12)
        assert body_af.yaw_rate == pytest.approx(world_af.yaw_rate, abs=1e-12)
    # and the run was not a trivial one
    assert body_af.speed_xy > 5.0
    assert abs(wrap_pi(body_af.yaw - 0.7)) > 0.1


def test_body_command_is_resolved_through_the_pre_rotation_yaw():
    """Using the post-rotation yaw would rotate the command by yaw_rate*dt."""
    yaw0 = 0.4
    dt = 0.1
    af = _airframe(yaw=yaw0)
    af.step(BodyCommand(vx=10.0, vy=0.0, yaw_rate=99.0, frame="body"), dt)
    yaw_after = af.yaw
    assert wrap_pi(yaw_after - yaw0) == pytest.approx(0.1, abs=1e-12)  # a real turn
    # velocity points along the OLD heading, not the new one
    assert _heading(af) == pytest.approx(yaw0, abs=1e-12)
    assert abs(wrap_pi(_heading(af) - yaw_after)) > 0.09


def test_body_command_tracks_the_heading_across_ticks():
    """After the aircraft has turned, the same body command points elsewhere."""
    af = _airframe(yaw=0.0)
    af.step(BodyCommand(yaw_rate=99.0, frame="world"), DT)  # turn, do not translate
    yaw1 = af.yaw
    assert yaw1 > 0.02
    assert af.vel == (0.0, 0.0, 0.0)
    af.step(BodyCommand(vx=10.0, yaw_rate=99.0, frame="body"), DT)
    # accelerating from rest, so the velocity is the command's direction: the
    # heading held at the START of this tick, not the one it ends with
    assert _heading(af) == pytest.approx(yaw1, abs=1e-12)
    assert af.yaw > yaw1 + 0.02
    assert abs(wrap_pi(_heading(af) - af.yaw)) > 0.02


def test_world_command_is_independent_of_heading():
    a = _airframe(yaw=0.0)
    b = _airframe(yaw=2.3)
    for _ in range(120):
        cmd = BodyCommand(vx=9.0, vy=-4.0, vz=1.5, yaw_rate=1.0, frame="world")
        a.step(cmd, DT)
        b.step(cmd, DT)
    assert a.xyz == b.xyz
    assert a.vel == b.vel
    assert a.speed_xy > 5.0
    # only the headings differ, by exactly their initial offset
    assert wrap_pi(b.yaw - a.yaw) == pytest.approx(2.3, abs=1e-9)


def test_unknown_frame_string_falls_back_to_body():
    """The frame check is ``cmd.frame == "world"``; anything else is body.

    Not a bug, but worth pinning: a typo'd frame is silently treated as body
    rather than rejected, so this is the behaviour callers actually get.
    """
    yaw = 1.1
    typo = _airframe(yaw=yaw)
    body = _airframe(yaw=yaw)
    typo.step(BodyCommand(vx=10.0, frame="World"), DT)
    body.step(BodyCommand(vx=10.0, frame="body"), DT)
    assert typo.vel == body.vel
    assert _heading(typo) == pytest.approx(yaw, abs=1e-12)


# --------------------------------------------------------------------------
# ground floor
# --------------------------------------------------------------------------

def test_descent_is_clamped_at_min_agl():
    lim = Limits()
    af = Airframe(xyz=(0.0, 0.0, 6.0), ground_z=0.0, limits=lim)
    for _ in range(200):
        af.step(BodyCommand(vx=10.0, vz=-1000.0, frame="world"), DT)
        assert af.xyz[2] >= lim.min_agl - 1e-12
    assert af.xyz[2] == pytest.approx(lim.min_agl, abs=TOL)
    # sitting on the floor, the downward velocity is zeroed each tick ...
    assert af.vel[2] == 0.0
    # ... while horizontal flight is unaffected
    assert af.vel[0] == pytest.approx(10.0, abs=1e-9)
    assert af.xyz[0] > 90.0


def test_floor_follows_ground_z():
    lim = Limits(min_agl=3.0)
    af = Airframe(xyz=(0.0, 0.0, 40.0), ground_z=25.0, limits=lim)
    for _ in range(300):
        af.step(BodyCommand(vz=-1000.0, frame="world"), DT)
        assert af.xyz[2] >= 25.0 + 3.0 - 1e-12
    assert af.xyz[2] == pytest.approx(28.0, abs=TOL)
    assert af.vel[2] == 0.0


def test_floor_clamp_preempts_the_vertical_accel_limit():
    """Hitting the floor stops the descent instantly, by design.

    The clamp is a hard positional constraint ("clamped rather than allowed to
    fly into terrain"), so it is the one place where the vertical velocity may
    change by more than ``max_accel_z * dt`` in a tick.
    """
    lim = Limits()
    af = Airframe(xyz=(0.0, 0.0, 12.0), ground_z=0.0, limits=lim)
    for _ in range(100):
        af.step(BodyCommand(vz=-1000.0, frame="world"), DT)
        if af.xyz[2] <= lim.min_agl + 1e-12:
            break
    assert af.xyz[2] == pytest.approx(lim.min_agl, abs=TOL)
    assert af.vel[2] == 0.0


def test_climbing_off_the_floor_works_immediately():
    lim = Limits()
    af = Airframe(xyz=(0.0, 0.0, lim.min_agl), ground_z=0.0, limits=lim)
    af.step(BodyCommand(vz=-1000.0, frame="world"), DT)
    assert af.xyz[2] == pytest.approx(lim.min_agl, abs=TOL)
    for k in range(1, 6):
        af.step(BodyCommand(vz=1000.0, frame="world"), DT)
        assert af.vel[2] == pytest.approx(k * lim.max_accel_z * DT, abs=1e-12)
    assert af.xyz[2] > lim.min_agl


def test_spawning_below_the_floor_is_corrected_on_the_first_tick():
    """A sub-floor spawn is teleported up to ``ground_z + min_agl``.

    ``__post_init__`` does not enforce the floor, so the correction happens in
    the first ``step`` and it is a jump, not a climb -- pinned here because the
    module otherwise goes out of its way to avoid position discontinuities.
    """
    lim = Limits()
    af = Airframe(xyz=(0.0, 0.0, 1.0), ground_z=10.0, limits=lim)
    assert af.xyz[2] == 1.0  # construction does not clamp
    af.step(BodyCommand(frame="world"), DT)
    assert af.xyz[2] == pytest.approx(12.0, abs=TOL)
    assert af.vel[2] == 0.0


def test_all_limits_hold_together_over_a_long_mixed_run():
    """One run, every invariant at once, floor included."""
    lim = Limits()
    af = Airframe(xyz=(0.0, 0.0, 30.0), yaw=-1.2, ground_z=0.0, limits=lim)
    rng = random.Random(4242)
    for k in range(2000):
        v0, r0 = af.vel, af.yaw_rate
        af.step(BodyCommand(vx=rng.uniform(-50, 50), vy=rng.uniform(-50, 50),
                            vz=rng.uniform(-25, 25), yaw_rate=rng.uniform(-12, 12),
                            frame="world" if k % 3 else "body"), DT)
        assert af.speed_xy <= lim.max_speed_xy + 1e-12
        assert abs(af.vel[2]) <= lim.max_speed_z + 1e-12
        assert math.hypot(af.vel[0] - v0[0], af.vel[1] - v0[1]) <= lim.max_accel_xy * DT + 1e-12
        assert abs(af.yaw_rate) <= lim.max_yaw_rate + 1e-12
        assert abs(af.yaw_rate - r0) <= lim.max_yaw_accel * DT + 1e-12
        assert -math.pi < af.yaw <= math.pi
        assert af.xyz[2] >= lim.min_agl - 1e-12
        # the vertical accel limit holds except where the floor clamps
        if af.xyz[2] > lim.min_agl + 1e-9:
            assert abs(af.vel[2] - v0[2]) <= lim.max_accel_z * DT + 1e-12
    assert af.speed_xy > 0.0


# ---------------------------------------------------------------- the ceiling

def test_the_ceiling_clamps_altitude_and_kills_upward_velocity():
    """The chaser has a service ceiling, not just a floor.

    Missing this was a real defect and a slow one to find, because it needs a
    long episode to bite: the search climbs deliberately (the camera is blind
    above 15.5 degrees, so altitude is the only way to look up), and over a
    45 s search an unbounded 3 m/s climb puts the aircraft 135 m up, above
    everything, looking down at terrain -- where it locked onto ground clutter
    and never recovered.
    """
    lim = Limits(max_agl=60.0, min_agl=2.0, max_speed_z=5.0, max_accel_z=100.0)
    af = Airframe(xyz=(0.0, 0.0, 10.0), ground_z=0.0, limits=lim)
    for _ in range(600):                      # 30 s of climbing flat out
        af.step(BodyCommand(0.0, 0.0, 5.0, 0.0, frame="world"), DT)
    assert af.xyz[2] == pytest.approx(60.0, abs=1e-9)
    assert af.vel[2] <= 0.0                   # the climb is arrested, not stored
    # ...and it can still come back down.
    af.step(BodyCommand(0.0, 0.0, -5.0, 0.0, frame="world"), DT)
    assert af.xyz[2] < 60.0


def test_the_ceiling_offsets_with_the_scene_ground():
    lim = Limits(max_agl=40.0, max_accel_z=100.0)
    af = Airframe(xyz=(0.0, 0.0, 30.0), ground_z=25.0, limits=lim)
    for _ in range(400):
        af.step(BodyCommand(0.0, 0.0, 5.0, 0.0, frame="world"), DT)
    assert af.xyz[2] == pytest.approx(65.0, abs=1e-9)     # 25 + 40


def test_the_band_leaves_ordinary_flight_untouched():
    """Between the bounds nothing is clamped -- the band must not tax normal flight."""
    lim = Limits(min_agl=2.0, max_agl=60.0, max_accel_z=100.0)
    af = Airframe(xyz=(0.0, 0.0, 30.0), ground_z=0.0, limits=lim)
    for _ in range(40):
        af.step(BodyCommand(0.0, 0.0, 2.0, 0.0, frame="world"), DT)
    assert af.xyz[2] == pytest.approx(30.0 + 2.0 * 40 * DT, abs=1e-9)
    assert af.vel[2] == pytest.approx(2.0, abs=1e-9)
