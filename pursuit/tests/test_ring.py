"""The camera ring: does it really see everywhere, and is it really one target?

Two failure modes are specific to a ring and both are invisible in a summary
statistic, which is why they are pinned here rather than watched for in a video.

A **hole** is a bearing no camera covers. Nothing raises; a fraction of
engagements simply fail for no reason anyone can reconstruct afterwards.

A **split** is the opposite: the 6 degree seam overlap means a target really is
detected twice, and a system that treats those as two objects spends the
engagement alternating between them. Both are geometry, so both can be tested
exactly instead of empirically.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pursuit.city import (_corners, build_city_suite, city_geometry,
                          defended, defended_radius, facade_aim,
                          load_buildings, nearest_target, station_and_set)
from pursuit.dynamics import Limits
from pursuit.geometry import (Intrinsics, angle_between, body_bearing,
                              body_to_pixel, offaxis_scale, pixel_to_body,
                              range_from_span, yaw_homography)
from pursuit.guidance import GuidanceConfig, PursuitGuidance
from pursuit.ring import (Ring, RingCamera, RingDetection, RingTracker,
                          RingTrackerConfig, default_ring, fuse_detections)
from pursuit.sandbox import RING_INTRINSICS
from simulators.pegasus.camera import (RING_HFOV_DEG, RING_RESOLUTION,
                                       ring_coverage_deg, ring_intrinsics,
                                       ring_mounts)

RING = default_ring(RING_INTRINSICS)


def _dirs(n=720, el_deg=0.0):
    el = math.radians(el_deg)
    for i in range(n):
        a = 2.0 * math.pi * i / n
        yield (math.cos(el) * math.cos(a), math.cos(el) * math.sin(a), math.sin(el))


class TestTheRingAgreesWithTheSimulator:
    """The fast loop and the renderer must describe the same aircraft.

    They are separate code -- one has to run inside a container with no numpy
    of ours and no weights, the other has to run without Isaac -- so the ring is
    defined twice. Two definitions that drift apart turn every sandbox-versus-
    Isaac difference into an unanswerable question, which is the exact failure
    the shared ``place_engagement`` was introduced to prevent elsewhere.
    """

    def test_intrinsics_match(self):
        sim = ring_intrinsics()
        for f in ("width", "height", "fx", "fy", "cx", "cy"):
            assert getattr(sim, f) == pytest.approx(getattr(RING_INTRINSICS, f),
                                                    rel=1e-9), f

    def test_mount_order_and_angles_match(self):
        sim = ring_mounts(4)
        assert [m.name for m in sim] == [c.name for c in RING.cameras]
        for m, c in zip(sim, RING.cameras):
            assert m.yaw == pytest.approx(c.mount_yaw, abs=1e-12)

    def test_the_declared_overlap_is_the_real_one(self):
        covered, overlap = ring_coverage_deg(ring_intrinsics(), 4)
        assert covered == pytest.approx(4 * RING_HFOV_DEG)
        assert overlap == pytest.approx(RING_HFOV_DEG - 90.0)
        assert overlap > 0.0, "four cameras that do not overlap leave four holes"


class TestCoverage:
    def test_no_bearing_is_unseen(self):
        for los in _dirs(720):
            assert RING.seeing(los), (
                f"blind at azimuth {math.degrees(math.atan2(los[1], los[0])):.1f}")

    def test_every_bearing_has_exactly_one_owner(self):
        for los in _dirs(720):
            owner = RING.owner(los)
            assert owner is not None
            assert owner.sees(los)

    def test_the_seams_overlap_by_the_declared_amount(self):
        two = [los for los in _dirs(3600) if len(RING.seeing(los)) > 1]
        # Four seams, each the declared width, as a fraction of the circle.
        _covered, overlap = ring_coverage_deg(RING_INTRINSICS, 4)
        assert len(two) / 3600 == pytest.approx(4 * overlap / 360.0, abs=0.01)

    def test_coverage_survives_elevation_within_the_vertical_field(self):
        # The horizontal field of a pinhole narrows off the horizontal plane;
        # a ring sized only on the equator develops holes above and below it.
        for el in (-15.0, -8.0, 8.0, 15.0):
            for los in _dirs(360, el_deg=el):
                assert RING.seeing(los), f"blind at elevation {el}"

    def test_a_direction_above_the_ring_still_has_an_owner(self):
        """The ring covers the circle, not the sphere, and a coast can leave it.

        A tracker coasting through a dropout predicts a direction, and nothing
        stops that prediction drifting above the 20.9 degree vertical field.
        "Which camera would be looking at this" still has an answer there, and
        the version that answered None instead produced a *valid* estimate with
        no pixel coordinates -- which crashed the telemetry writer three
        episodes into the first live run.
        """
        for el in (25.0, 40.0, 70.0):
            e = math.radians(el)
            for a in (0.0, 1.1, 2.7, -2.0):
                los = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a),
                       math.sin(e))
                assert RING.owner(los) is not None
                assert not RING.seeing(los), "this is meant to be outside them"

    def test_the_vertical_field_is_what_it_claims(self):
        assert RING_INTRINSICS.vfov_deg == pytest.approx(41.79, abs=0.05)
        assert RING_RESOLUTION == (RING_INTRINSICS.width, RING_INTRINSICS.height)


class TestBearings:
    def test_pixel_and_body_are_exact_inverses(self):
        cam = RING.cameras[1]
        for u in (5.0, 400.0, 1024.0, 1700.0, 2040.0):
            for v in (5.0, 200.0, 352.0, 700.0):
                los = cam.to_body(u, v)
                back = cam.to_pixel(los)
                assert back is not None
                assert back[0] == pytest.approx(u, abs=1e-7)
                assert back[1] == pytest.approx(v, abs=1e-7)

    def test_a_mount_rotation_is_not_an_azimuth_addition(self):
        """The reason :func:`pixel_to_body` rotates a vector and not an angle.

        Tangent-plane angles are what a pinhole measures and they are not
        additive across a mount rotation once the elevation is non-zero. Pinned
        because the cheap version looks right, passes on the horizon, and is
        wrong by degrees exactly where a target that is climbing sits.
        """
        cam = RING.cameras[1]           # 90 degrees round
        u, v = 1700.0, 120.0            # well off-axis, well above centre
        az_cam = -math.atan2(u - cam.intr.cx, cam.intr.fx)
        el_cam = -math.atan2(v - cam.intr.cy, cam.intr.fy)
        naive = (math.cos(el_cam) * math.cos(az_cam + cam.mount_yaw),
                 math.cos(el_cam) * math.sin(az_cam + cam.mount_yaw),
                 math.sin(el_cam))
        exact = cam.to_body(u, v)
        assert math.degrees(angle_between(naive, exact)) > 1.0

    def test_body_bearing_keeps_meaning_behind_the_aircraft(self):
        for los in _dirs(360):
            az, _el = body_bearing(los)
            assert -math.pi <= az <= math.pi
        behind = RING.owner((-1.0, 0.02, 0.0))
        assert behind is not None and behind.name == "aft"


class TestOffAxisRange:
    """A pinhole stretches by sec^2 off-axis; a range that ignores it is short.

    Measured on the rig before it was fixed: a target at a fixed 40 m rendered
    8 px on the boresight and 15 px at 43 degrees off it. Believing the second
    number puts the target at 21 m.
    """

    def test_scale_is_one_on_axis_and_secant_squared_at_the_edge(self):
        i = RING_INTRINSICS
        assert offaxis_scale(i, i.cx, i.cy) == pytest.approx(1.0)
        edge = offaxis_scale(i, 0.0, i.cy)
        assert edge == pytest.approx(1.0 / math.cos(math.radians(48.0)) ** 2,
                                     rel=1e-6)
        assert edge > 2.0

    def test_range_is_recovered_from_a_stretched_span(self):
        i = RING_INTRINSICS
        true_r = 40.0
        for u in (i.cx, 300.0, 1900.0):
            span = i.fx * 0.47 / true_r * offaxis_scale(i, u, i.cy)
            got = range_from_span(i, span, 0.47, u, i.cy)
            assert got == pytest.approx(true_r, rel=1e-9)

    def test_ignoring_it_is_wrong_by_the_amount_claimed(self):
        i = RING_INTRINSICS
        span = i.fx * 0.47 / 40.0 * offaxis_scale(i, 100.0, i.cy)
        assert range_from_span(i, span, 0.47) < 0.55 * 40.0


class TestFusion:
    def _det(self, cam, u, v, score=0.8, kind="appearance", span=10.0):
        return RingDetection(los=cam.to_body(u, v), score=score, camera=cam.name,
                             u=u, v=v, span_px=span, kind=kind,
                             span_rad=span / cam.intr.fx)

    def test_one_drone_in_a_seam_is_one_detection(self):
        # A direction inside the fwd/left overlap, so both cameras really see it.
        los = None
        for cand in _dirs(3600):
            if len(RING.seeing(cand)) > 1:
                los = cand
                break
        assert los is not None
        dets = []
        for cam in RING.seeing(los):
            uv = cam.to_pixel(los)
            dets.append(self._det(cam, uv[0], uv[1]))
        assert len(dets) == 2
        assert len(fuse_detections(dets, RING)) == 1

    def test_two_real_objects_are_not_merged(self):
        cam = RING.cameras[0]
        a = self._det(cam, 400.0, 352.0)
        b = self._det(cam, 1600.0, 352.0)
        assert len(fuse_detections([a, b], RING)) == 2

    def test_an_appearance_box_survives_a_motion_blob(self):
        """Only appearance carries a span, so the merge must not drop it."""
        cam = RING.cameras[0]
        app = self._det(cam, 1024.0, 352.0, score=0.6)
        blob = RingDetection(los=cam.to_body(1026.0, 353.0), score=0.9,
                             camera=cam.name, u=1026.0, v=353.0, span_px=None,
                             kind="motion", span_rad=14.0 / cam.intr.fx)
        out = fuse_detections([app, blob], RING)
        assert len(out) == 1
        assert out[0].kind == "appearance"
        assert out[0].span_px is not None
        assert out[0].score == pytest.approx(0.9)   # keeps the best evidence


class TestRingTracker:
    def _run(self, cfg=None, n=12, yaw_rate=0.0, az0=0.3, az_rate=0.0,
             kind="appearance", score=0.9, ego_static=True):
        tr = RingTracker(RING, cfg or RingTrackerConfig())
        dt, yaw = 0.05, 0.0
        for i in range(n):
            t = i * dt
            world_az = az0 + az_rate * t
            los = (math.cos(world_az - yaw), math.sin(world_az - yaw), 0.0)
            cam = RING.owner(los)
            uv = cam.to_pixel(los)
            det = RingDetection(
                los=los, score=score, camera=cam.name, u=uv[0], v=uv[1],
                # A real motion blob carries no span, and a real appearance box
                # always does -- the fake has to keep that difference or the
                # test cannot see it.
                span_px=(12.0 if kind == "appearance" else None),
                kind=kind, span_rad=12.0 / cam.intr.fx)
            tr.step([det], t, yaw, ego_static=ego_static)
            yaw += yaw_rate * dt
        return tr

    def test_a_target_that_flies_confirms(self):
        """Sixty frames, because a lock is not granted until flight is proved.

        Three seconds of watching before anything gets a real track is the
        deliberate cost of not locking onto buildings; see _CandidatePool.
        """
        tr = self._run(n=60, az_rate=0.05)
        assert tr.confirmed and tr.misses == 0
        est = tr.estimate()
        assert est.valid and est.source == "detector"
        assert est.los_body is not None

    def test_the_gate_survives_a_saturated_yaw(self):
        """The ego-yaw term, which is the point of tracking in body axes.

        At 2.5 rad/s the whole image sweeps 0.125 rad between frames -- twice
        the base gate. Without subtracting the known heading change, a
        stationary target falls outside the gate every single frame while the
        aircraft is turning, which is most of an engagement.
        """
        tr = self._run(yaw_rate=2.5, n=60, az_rate=0.05)
        assert tr.confirmed
        # It seeds only once flight is proved, so what is pinned is that the
        # gate holds every frame *after* that, not from frame one.
        assert tr.hits >= 8 and tr.misses == 0

    def test_a_seam_crossing_is_one_track(self):
        """The target walks from one camera into the next; the lock must not."""
        tr = RingTracker(RING, RingTrackerConfig())
        dt = 0.05
        cams = set()
        for i in range(40):
            t = i * dt
            az = math.radians(40.0) + math.radians(0.6) * i   # sweeps past 48 deg
            los = (math.cos(az), math.sin(az), 0.0)
            cam = RING.owner(los)
            cams.add(cam.name)
            uv = cam.to_pixel(los)
            # Flying: a 12 deg/s sweep is a close-in engagement, and while the
            # aircraft translates the candidate pool cannot separate a fixed
            # object from a moving one anyway, so seeding falls back to
            # corroboration. That is the path this test is about.
            tr.step([RingDetection(los=los, score=0.9, camera=cam.name,
                                   u=uv[0], v=uv[1], span_px=12.0,
                                   kind="appearance",
                                   span_rad=12.0 / cam.intr.fx)], t, 0.0,
                    ego_static=False)
        assert len(cams) >= 2, "the sweep never actually changed camera"
        assert tr.confirmed and tr.hits >= 38 and tr.misses == 0

    def test_a_motion_track_earns_confidence_by_persisting_and_moving(self):
        cfg = RingTrackerConfig()
        # Crossing at 0.05 rad/s: a real inbound target's bearing rate.
        moving = self._run(n=70, kind="motion", score=0.5, az_rate=0.05)
        assert moving.travel_rad > cfg.motion_min_travel_rad
        assert moving.score > 0.62, "a moving contact never earns the right to steer"
        # ...and a contact seen three times is worth nothing at all.
        assert self._run(n=3, kind="motion", score=0.5, az_rate=0.05).score < 0.62

    def test_a_persistent_but_motionless_contact_never_steers(self):
        """The failure that lost the first live engagement.

        A renderer artefact produces a gated detection on *every* frame -- it is
        more persistent than a real target, not less -- so a confidence built on
        persistence alone climbs straight to its ceiling and the aircraft flies
        at a flickering roof edge. From a station-keeping interceptor a fixed
        object has a bearing rate of exactly zero, and that is the whole
        discriminator.
        """
        cfg = RingTrackerConfig()
        # A lone motionless blob *is* granted a track -- with one thing in the
        # sky there is nothing to disambiguate -- but it is capped below the
        # promotion floor and never steers.
        stuck = self._run(n=120, kind="motion", score=0.9, az_rate=0.0)
        assert stuck.confirmed, "it should still be tracked -- just not trusted"
        assert stuck.travel_rad < cfg.motion_min_travel_rad
        assert stuck.score <= cfg.motion_unproven_score
        assert stuck.estimate().score < 0.62

        # And among *several* motionless contacts -- which is what a city looks
        # like -- none of them is granted a track at all, so the real drone is
        # not locked out of a single-target tracker by the first artefact.
        tr = RingTracker(RING, cfg)
        for i in range(120):
            dets = []
            for az in (0.30, -0.80, 2.10):
                los = (math.cos(az), math.sin(az), 0.0)
                cam = RING.owner(los)
                uv = cam.to_pixel(los)
                dets.append(RingDetection(los=los, score=0.9, camera=cam.name,
                                          u=uv[0], v=uv[1], span_px=None,
                                          kind="motion", span_rad=0.01))
            tr.step(dets, i * 0.05, 0.0)
        assert not tr.confirmed, "a motionless contact captured the tracker"

    def test_jitter_is_not_travel(self):
        """Net displacement, not path length.

        An artefact wandering a pixel a frame covers 1.1 mrad per step, and
        twenty frames of that sums to more than the 10 mrad bar without the
        thing having gone anywhere.
        """
        import random as _r
        rnd = _r.Random(4)
        tr = RingTracker(RING, RingTrackerConfig())
        for i in range(30):
            az = 0.3 + rnd.uniform(-0.0015, 0.0015)
            el = rnd.uniform(-0.0015, 0.0015)
            los = (math.cos(az) * math.cos(el), math.sin(az) * math.cos(el),
                   math.sin(el))
            cam = RING.owner(los)
            uv = cam.to_pixel(los)
            tr.step([RingDetection(los=los, score=0.9, camera=cam.name,
                                   u=uv[0], v=uv[1], span_px=None,
                                   kind="motion", span_rad=0.01)], i * 0.05, 0.0)
        assert tr.travel_rad < RingTrackerConfig().motion_min_travel_rad
        assert tr.score <= RingTrackerConfig().motion_unproven_score

    def test_the_aircrafts_own_yaw_is_not_counted_as_travel(self):
        """Otherwise a slewing aircraft certifies every rooftop it passes."""
        tr = self._run(n=25, kind="motion", score=0.9, az_rate=0.0, yaw_rate=1.5)
        assert tr.travel_rad < RingTrackerConfig().motion_min_travel_rad

    def test_travel_is_only_measured_from_a_stationary_observer(self):
        """Parallax is not evidence, and while flying it is all there is.

        A fixed object seen from a *moving* aircraft sweeps across the sky just
        as a flying one does -- at 100 m and 18 m/s, 180 mrad a second, which is
        eighteen times the bar. Accumulating travel while translating therefore
        certifies every building in the town, which is precisely what happened:
        the promotion arrived four seconds in, after the aircraft had already
        started moving on the unpromoted contact.
        """
        cfg = RingTrackerConfig()
        tr = RingTracker(RING, cfg)
        for i in range(70):
            az = 0.3 + 0.05 * i * 0.05        # genuinely crossing
            los = (math.cos(az), math.sin(az), 0.0)
            cam = RING.owner(los)
            uv = cam.to_pixel(los)
            tr.step([RingDetection(los=los, score=0.9, camera=cam.name,
                                   u=uv[0], v=uv[1], span_px=None,
                                   kind="motion", span_rad=0.01)],
                    i * 0.05, 0.0, ego_static=False)
        assert tr.confirmed
        assert tr.travel_rad == 0.0, "travel accrued while the observer moved"
        assert tr.score <= cfg.motion_unproven_score

    def test_a_confident_appearance_box_is_not_enough_on_its_own(self):
        """The sim-trained model returns 0.85 on Rivermark rooftops.

        So a confident box is not evidence of an aircraft, and the travel gate
        has to apply to it too -- otherwise the cheap wide-recall stage is held
        to a standard the expensive precise one is exempt from, which is the
        wrong way round.
        """
        cfg = RingTrackerConfig()
        tr = RingTracker(RING, cfg)
        # Two motionless contacts, which is what a city looks like: with more
        # than one thing in the sky nothing is granted a track until it has been
        # watched flying, however confident the box.
        for i in range(120):
            dets = []
            for az in (0.30, -0.80):
                los = (math.cos(az), math.sin(az), 0.0)
                cam = RING.owner(los)
                uv = cam.to_pixel(los)
                dets.append(RingDetection(los=los, score=0.9, camera=cam.name,
                                          u=uv[0], v=uv[1], span_px=10.0,
                                          kind="appearance", span_rad=0.01))
            tr.step(dets, i * 0.05, 0.0)
        assert not tr.confirmed, "a confident box on a motionless thing took the track"
        assert not tr.estimate().valid

    def test_only_appearance_sets_the_pixel_span(self):
        """A dilated blob's width is the morphology kernel, not the drone."""
        tr = self._run(n=10, kind="motion", score=0.6)
        assert tr.span_px is None
        assert tr.estimate().span_px is None

    def test_a_lone_detection_does_not_start_a_track(self):
        tr = RingTracker(RING, RingTrackerConfig())
        cam = RING.cameras[0]
        uv = (1024.0, 352.0)
        tr.step([RingDetection(los=cam.to_body(*uv), score=0.9, camera=cam.name,
                               u=uv[0], v=uv[1], span_px=10.0,
                               kind="appearance", span_rad=0.01)], 0.0, 0.0)
        assert not tr.confirmed


class TestOmnidirectionalGuidance:
    def _g(self, omni: bool):
        return PursuitGuidance(RING_INTRINSICS, Limits(), 0.47,
                               GuidanceConfig(omnidirectional=omni))

    def test_the_search_stops_slewing(self):
        cmd = self._g(True)._search_command()
        assert cmd.yaw_rate == 0.0
        assert cmd.source == "search:watch"
        assert self._g(False)._search_command().source.startswith("search:")

    def test_the_single_camera_search_still_slews(self):
        g = self._g(False)
        g._search_hold_s = 99.0
        assert abs(g._search_command().yaw_rate) > 0.5

    def test_closing_speed_is_no_longer_gated_by_boresight(self):
        far = math.radians(60.0)
        assert self._g(True)._speed_gate(far) == 1.0
        assert self._g(False)._speed_gate(far) < 0.5

    def test_acquire_does_not_fly_at_an_unproven_contact(self):
        """The failure that lost every engagement of the first live city run.

        ACQUIRE flies the closure exactly like PURSUE, which is right for a
        single forward camera -- moving toward a candidate is how that sensor
        finds out what it is. With a ring it is self-defeating: the moment the
        aircraft moves it loses the background model that sees a 3 px drone at
        140 m, *and* it loses the only test that separates a building from an
        aircraft, because a fixed object's bearing is constant only from a fixed
        observer. The seeker then talks itself into a rooftop and destroys the
        evidence that would have refuted it.
        """
        from pursuit.guidance import ACQUIRE
        from pursuit.perception import TargetEstimate

        def fly(omni: bool):
            g = self._g(omni)
            # A contact the detector keeps reporting but which has not earned
            # promotion: exactly the state a false lock lives in.
            for i in range(6):
                m = TargetEstimate(valid=True, u=1024.0, v=352.0, span_px=8.0,
                                   az=0.0, el=0.0, score=0.5, source="detector",
                                   los_body=(1.0, 0.0, 0.0))
                g._promoted = False
                st = g.step(i * 0.05, 0.05, (0.0, 0.0, 30.0), 0.0,
                            (0.0, 0.0, 0.0), m)
            assert st.mode == ACQUIRE and not g.confirmed, st.mode
            return math.hypot(st.command.vx, st.command.vy)

        assert fly(True) < 0.1, "a ring flew at an unpromoted contact"
        assert fly(False) > 1.0, "the single-camera behaviour was changed"

    def test_a_confirmed_lock_does_fly(self):
        """...and the hold must not become a system that never launches."""
        from pursuit.guidance import PURSUE
        g = self._g(True)
        g.mode = PURSUE
        g.confirmed = True
        g.los.s = (1.0, 0.0, 0.0)
        g.los.t_last = 0.0
        g.los.n = 9
        st = g.step(0.5, 0.05, (0.0, 0.0, 30.0), 0.0, (0.0, 0.0, 0.0), None)
        assert math.hypot(st.command.vx, st.command.vy) > 1.0

    def test_watching_holds_the_altitude_it_started_at(self):
        g = self._g(True)
        g._chaser_z = 35.0
        assert g._search_command().vz == pytest.approx(0.0)
        g._chaser_z = 30.0                      # drifted down 5 m
        assert g._search_command().vz > 0.0


class TestMotionDetector:
    """Synthetic frames, because the properties that matter are not subtle.

    A drone crossing a static scene must be found; a static scene must be
    quiet; and -- the one that cost a measurement run -- a target that lingers
    must *not* suppress itself, which is a bug that needs a hundred frames of
    dwell to appear and is therefore invisible to every other test here.
    """

    def _scene(self, seed=0):
        rng = np.random.default_rng(seed)
        # A textured background with a little per-frame sensor noise, plus a
        # bright band standing in for a skyline edge.
        base = (rng.random((704, 2048, 3)) * 30 + 90).astype(np.uint8)
        base[:200] = np.clip(base[:200].astype(int) + 90, 0, 255).astype(np.uint8)
        return base

    def _frames(self, base, uv=None, size=4, noise=1.5, seed=1):
        rng = np.random.default_rng(seed)
        f = np.clip(base.astype(np.float32)
                    + rng.normal(0.0, noise, base.shape), 0, 255).astype(np.uint8)
        out = {}
        for c in RING.cameras:
            g = f.copy()
            if uv is not None and c.name == "fwd":
                u, v = int(uv[0]), int(uv[1])
                g[v:v + size, u:u + size] = 25          # a dark quadrotor
            out[c.name] = g
        return out

    def test_it_finds_a_small_mover_and_ignores_a_static_scene(self):
        from pursuit.ring import MotionConfig, RingMotionDetector
        cfg = MotionConfig()
        md = RingMotionDetector(RING, cfg)
        base = self._scene()
        quiet = 0
        for i in range(cfg.bg_warmup + 12):                  # warm up, empty sky
            quiet += len(md.detect(self._frames(base, seed=i), 0.0, 0.0))
        assert quiet <= 6, f"{quiet} false blobs on a static scene"
        found = 0
        for i in range(10):                                  # now fly one through
            dets = md.detect(self._frames(base, uv=(600 + 9 * i, 420), size=4,
                                          seed=100 + i), 0.0, 0.0)
            found += any(d.camera == "fwd" for d in dets)
        assert found >= 7, f"found the mover on only {found}/10 frames"

    def test_a_lingering_target_does_not_suppress_itself(self):
        """The measured regression: detection got *worse* the longer it dwelt.

        With the per-pixel noise estimate updated on foreground pixels too, a
        target that holds still feeds its own contrast into its own threshold
        and disappears. Sixty frames of dwell is enough to see it.
        """
        from pursuit.ring import MotionConfig, RingMotionDetector
        cfg = MotionConfig()
        md = RingMotionDetector(RING, cfg)
        base = self._scene(seed=3)
        for i in range(cfg.bg_warmup + 8):
            md.detect(self._frames(base, seed=i), 0.0, 0.0)
        early = late = 0
        for i in range(70):
            # Creeping one pixel every ten frames: an inbound target on a
            # near-constant bearing, which is what this mission is made of.
            dets = md.detect(self._frames(base, uv=(700 + i // 10, 400), size=5,
                                          seed=500 + i), 0.0, 0.0)
            hit = any(d.camera == "fwd" for d in dets)
            if i < 15:
                early += hit
            elif i >= 55:
                late += hit
        assert early >= 10, f"not detected even at first ({early}/15)"
        assert late >= 10, (
            f"detected {early}/15 at first but only {late}/15 after 55 frames "
            f"of dwell -- the target is inflating its own threshold")

    def test_uncorrelated_noise_is_not_confirmed_and_a_target_is(self):
        """The one property that separates a drone from a renderer's noise.

        An empty Rivermark sky returns tens of motion blobs a frame, and
        raising the chronic-pixel suppression barely touched them -- so they are
        not stuck edges, they are *uncorrelated* per-frame noise. A drone at
        160 m moves a fifth of a pixel a tick, so it lands in the same place
        twice running and noise does not. Confirmation is therefore nearly free
        for the target and nearly fatal for the noise.
        """
        from pursuit.ring import MotionConfig, RingMotionDetector
        cfg = MotionConfig()
        md = RingMotionDetector(RING, cfg)
        base = self._scene(seed=11)
        for i in range(cfg.bg_warmup + 10):
            md.detect(self._frames(base, seed=i), 0.0, 0.0)
        kept = conf = 0
        for i in range(30):
            dets = md.detect(self._frames(base, uv=(700 + i * 0.2, 400), size=4,
                                          seed=900 + i), 0.0, 0.0)
            on = [d for d in dets if d.camera == "fwd" and abs(d.u - (700 + i * 0.2)) < 12]
            kept += bool(on)
            conf += any(d.confirmed for d in on)
        assert kept >= 25, f"the target was only detected {kept}/30 times"
        assert conf >= kept - 2, (
            f"detected {kept}/30 but confirmed only {conf} -- confirmation is "
            f"supposed to be nearly free for a real contact")

    def test_only_a_confirmed_contact_may_open_a_candidate(self):
        """Unconfirmed detections still feed an existing track, though.

        A tag, not a filter: recall where the tracker's own gate is already
        doing the work, precision where a new track is being created out of
        nothing.
        """
        from pursuit.ring import RingDetection as RD
        from pursuit.ring import _CandidatePool
        pool = _CandidatePool(RingTrackerConfig())
        cam = RING.cameras[0]

        def d(az, ok):
            los = (math.cos(az), math.sin(az), 0.0)
            uv = cam.to_pixel(los)
            return RD(los=los, score=0.7, camera=cam.name, u=uv[0], v=uv[1],
                      span_px=None, kind="motion", span_rad=0.004, confirmed=ok)

        for i in range(20):
            pool.update([d(0.2 + 0.001 * i, False)], i * 0.05, 0.0)
        assert not pool.items, "unconfirmed noise opened candidates"
        for i in range(20):
            pool.update([d(0.2 + 0.001 * i, True)], i * 0.05, 0.0)
        assert len(pool.items) == 1
        # ...and an unconfirmed detection now updates the candidate it matches.
        before = pool.items[0].hits
        pool.update([d(0.2 + 0.02, False)], 1.05, 0.0)
        assert pool.items[0].hits == before + 1

    def test_it_costs_less_than_the_appearance_model_it_aims(self):
        """Four full-resolution cameras a tick has to stay affordable.

        The 120 ms budget describes a workstation with the cores to itself. It is not a
        property of the code alone, so on hardware that cannot host the measurement this
        skips rather than reporting a red herring: a 2-core shared SLURM allocation
        measured 136 ms and failed a suite whose actual subject was an unrelated change
        to the evaluator. A performance guard that fires on machine load is not guarding
        performance, it is guarding nothing and costing a debugging hour each time.

        Set SPECKLOCK_TICK_BUDGET_MS to measure deliberately on other hardware.
        """
        import os
        import time as _t

        budget = float(os.environ.get("SPECKLOCK_TICK_BUDGET_MS", "120"))
        if "SPECKLOCK_TICK_BUDGET_MS" not in os.environ:
            try:
                cores = len(os.sched_getaffinity(0))     # Linux only
            except AttributeError:
                cores = os.cpu_count() or 1
            if cores < 4:
                pytest.skip(f"{cores} cores available; this budget describes a "
                            "four-camera tick on a machine with dedicated cores")

        from pursuit.ring import MotionConfig, RingMotionDetector
        cfg = MotionConfig()
        md = RingMotionDetector(RING, cfg)
        base = self._scene(seed=7)
        frames = [self._frames(base, uv=(600 + 8 * i, 420), seed=200 + i)
                  for i in range(8)]
        for _ in range(4):
            for f in frames:
                md.detect(f, 0.0, 0.0)
        t0 = _t.perf_counter()
        for f in frames:
            md.detect(f, 0.0, 0.0)
        ms = 1000.0 * (_t.perf_counter() - t0) / len(frames)
        assert ms < budget, f"{ms:.0f} ms a tick for four cameras (budget {budget:.0f})"


class TestYawHomography:
    def test_it_moves_a_fixed_point_the_way_a_turn_does(self):
        """Turning left must move a fixed world point to larger u.

        The sign is the whole content of the function, and getting it backwards
        doubles the apparent motion instead of removing it -- which reads as a
        detector that suddenly cannot cope with manoeuvring.
        """
        import numpy as np
        i = RING_INTRINSICS
        h = yaw_homography(i, math.radians(2.0))
        p = np.array([i.cx, i.cy, 1.0])
        q = h @ p
        q = q / q[2]
        assert q[0] > i.cx
        assert q[0] - i.cx == pytest.approx(i.fx * math.tan(math.radians(2.0)),
                                            rel=1e-6)
        assert q[1] == pytest.approx(i.cy, abs=1e-6)

    def test_zero_rotation_is_the_identity(self):
        import numpy as np
        h = yaw_homography(RING_INTRINSICS, 0.0)
        assert np.allclose(h, np.eye(3))


class TestCitySuite:
    def _suite(self):
        return build_city_suite(
            __import__("pursuit.episode", fromlist=["ScenarioConfig"]).ScenarioConfig())

    def test_pinning_the_arrival_angle_changes_the_heading_and_nothing_else(self):
        """A pinned suite must differ from the sampled one in yaw alone.

        The point of ``nose_relative_deg`` is to demonstrate one geometry, so it
        is only evidence if every *other* quantity -- range, speed, target,
        altitude, seed -- is the same engagement. That holds because the random
        heading is still drawn before it is discarded; delete that draw as dead
        code and the two suites silently stop being comparable.
        """
        from pursuit.episode import ScenarioConfig
        base = ScenarioConfig()
        sampled = build_city_suite(base, n=8)
        pinned = build_city_suite(base, n=8, nose_relative_deg=180.0,
                                  label="astern")
        assert [s.name for s in pinned] == [f"astern-{i * 45:03d}" for i in range(8)]
        for a, b in zip(sampled, pinned):
            assert abs(abs(b.start_bearing_deg) - 180.0) < 1e-9
            assert a.chaser_yaw_deg != b.chaser_yaw_deg
            for f in ("seed", "start_range_m", "start_elevation_deg", "defend_xy",
                      "defend_height_m", "defend_label", "transit_speed",
                      "evader_speed", "chaser_offset_xy", "altitude_m"):
                assert getattr(a, f) == getattr(b, f), f

    def test_any_pinned_arrival_angle_is_reproduced_exactly(self):
        from pursuit.episode import ScenarioConfig
        for want in (0.0, 90.0, -90.0, 145.0):
            for sc in build_city_suite(ScenarioConfig(), n=8,
                                       nose_relative_deg=want, label="pin"):
                assert abs(sc.start_bearing_deg - want) < 1e-9

    def test_the_astern_suite_is_registered_and_faces_away(self):
        from pursuit.episode import ScenarioConfig
        from pursuit.sandbox import build_suite
        suite = build_suite("city-astern", ScenarioConfig())
        assert len(suite) == 8
        for sc in suite:
            assert abs(abs(sc.start_bearing_deg) - 180.0) < 1e-9

    def test_the_survey_found_real_structures(self):
        bs = load_buildings()
        assert len(bs) >= 5
        for b in bs:
            w, d = b["footprint_m"]
            assert 6.0 <= min(w, d) and max(w, d) <= 150.0
            assert b["height_agl_m"] >= 8.0
        assert len(defended()) >= 2

    def test_the_defended_set_is_inside_the_measured_envelope(self):
        """And the excluded ones are outside it, for a stated reason.

        This is the whole shape of the mission: one interceptor covers a bubble
        whose radius is 0.6x its detection range, and which structures fall
        inside that bubble is arithmetic, not a preference. A suite that
        defended a building beyond it would be scoring an impossibility as a
        guidance failure.
        """
        station, keep, rad = station_and_set()
        limit = defended_radius()
        assert keep, "nothing is defensible -- the envelope collapsed"
        assert rad <= limit + 1e-6
        for b in keep:
            for c in _corners(b):
                assert math.dist(c, station) <= limit + 1e-6
        excluded = [b for b in load_buildings() if b not in keep]
        for b in excluded:
            assert max(math.dist(c, station) for c in _corners(b)) > limit

    def test_the_envelope_scales_with_detection_range_at_the_stated_rate(self):
        # 0.6 metres of defended radius per metre of detection, at 1.5x speed.
        a = defended_radius(detect_m=100.0)
        b = defended_radius(detect_m=200.0)
        assert (b - a) / 100.0 == pytest.approx(0.6, abs=1e-9)

    def test_arrivals_cover_the_whole_circle(self):
        world = sorted(((sc.chaser_yaw_deg + sc.start_bearing_deg) % 360.0)
                       for sc in self._suite())
        gaps = [b - a for a, b in zip(world, world[1:])]
        gaps.append(360.0 - world[-1] + world[0])
        assert max(gaps) < 25.0, f"a {max(gaps):.0f} degree quarter is never attacked"

    def test_the_interceptor_is_not_quietly_pre_pointed(self):
        """Every scenario draws a random heading, and the arrival bearing is
        measured from it. Forget that and a 360 degree sensor scores exactly
        what a 76 degree one would."""
        offs = [abs(((sc.start_bearing_deg + 180) % 360) - 180)
                for sc in self._suite()]
        assert max(offs) > 100.0, "no intruder ever starts behind the aircraft"
        assert sum(1 for o in offs if o > 48.0) >= len(offs) // 2

    def test_every_scenario_attacks_a_surveyed_building(self):
        names = {b["name"] for b in defended()}
        for sc in self._suite():
            assert sc.defend_label in names
            assert sc.strike_commit, "an intruder that breaks off cannot hit a building"
            assert sc.reveal_range_m == 0.0
            assert sc.speed_advantage == pytest.approx(1.5)

    def test_the_aim_point_is_on_the_facade_facing_the_intruder(self):
        b = defended()[0]
        far = (b["xy_rel"][0] + 500.0, b["xy_rel"][1])
        aim = facade_aim(b, far)
        assert aim[0] == pytest.approx(b["xy_rel"][0] + b["footprint_m"][0] / 2.0)
        assert aim[1] == pytest.approx(b["xy_rel"][1])
        # ...and from the other side it is the other face.
        near = (b["xy_rel"][0] - 500.0, b["xy_rel"][1])
        assert facade_aim(b, near)[0] < b["xy_rel"][0]

    def test_nearest_target_picks_by_facade_not_by_centre(self):
        """On a set where the answer is unambiguous, so the rule is the test.

        Not run against Rivermark's own buildings, and the reason is worth
        recording: two of the plaza blocks are close enough that their
        axis-aligned bounds *overlap*, so a point eight metres off one facade
        genuinely sits inside the neighbour. Any "obviously the nearest" probe
        on real data is testing the survey's geometry rather than the selection
        rule.
        """
        near = {"name": "near", "xy_rel": [0.0, 0.0], "footprint_m": [10.0, 10.0]}
        wide = {"name": "wide", "xy_rel": [60.0, 0.0], "footprint_m": [90.0, 10.0]}
        # By centre, `near` is closer to a probe at x=20; by facade, `wide` is,
        # because its wall reaches x=15.
        assert nearest_target([near, wide], (20.0, 0.0))["name"] == "wide"
        assert nearest_target([near, wide], (-40.0, 0.0))["name"] == "near"

    def test_each_scenario_attacks_the_building_nearest_its_own_start(self):
        bs = defended()
        for sc in self._suite():
            ang = math.radians(sc.chaser_yaw_deg + sc.start_bearing_deg)
            st = sc.chaser_offset_xy
            start = (st[0] + sc.start_range_m * math.cos(ang),
                     st[1] + sc.start_range_m * math.sin(ang))
            want = nearest_target(bs, start)
            assert sc.defend_label == want["name"]
            aim = facade_aim(want, start)
            assert sc.defend_xy[0] == pytest.approx(aim[0])
            assert sc.defend_xy[1] == pytest.approx(aim[1])

    def test_winnability_is_stated_rather_than_discovered(self):
        for sc in self._suite():
            g = city_geometry(sc)
            assert g["needed_acquire_m"] > g["asset_range_m"]
            assert g["time_to_strike_s"] > 0.0
