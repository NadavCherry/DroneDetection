"""The mission this system is for: hold over a city and stop a strike.

Every earlier suite is an *experiment about the interceptor*. The intruder's
business is with the chaser -- it flies at it, past it, or away from it -- which
quietly makes the chaser the centre of the world and gives it as long as it
likes to work. Nothing in those runs can be lost by being slow.

This one is the engagement:

* the interceptor rises over the middle of Rivermark and holds there, watching
  all four quarters at once (:mod:`pursuit.ring`);
* an intruder arrives from the horizon on a bearing drawn from the **whole
  circle**, at 150-190 m, where it is under 3 pixels across;
* it is not interested in the interceptor. It has picked the structure nearest
  its own approach and it is going to fly into it;
* the interceptor is 1.5x faster and has to reach it first.

So there are two ways to lose and they are not the same. Missing is one. The
other is arriving late, which the earlier suites could not even express, and
which is reported here as its own outcome -- ``target_struck`` -- alongside the
seconds that were left over on the intercepts that did land
(``strike_margin_s``). A defence that saves a building with 0.2 s to spare and
one that saves it with 8 s to spare are not the same defence.

## Where the buildings are

Measured off the loaded stage, not chosen off a compass. ``rivermark_plaza``
has real structures at real bearings, and an earlier defence suite that placed
its "buildings" by angle and radius put several of them in the middle of a
road. :mod:`simulators.pegasus.scripts.find_buildings` walks the USD stage and
writes ``simulators/pegasus/rivermark_buildings.json``; this module reads it.

Positions are stored **relative to the scene origin** so the same numbers mean
the same place in the renderer, whose usable origin is (60, 60), and in the
headless sandbox, whose origin is (0, 0).

## What the intruder aims at

The nearest point on the building's *footprint*, not its centre. A Rivermark
plaza block is 53 x 62 m, so "within 10 m of the centre" is a strike declared
when the intruder is still 20 m outside the wall -- which hands the interceptor
a defeat it did not suffer and, worse, makes the far buildings unwinnable for a
reason that is pure bookkeeping. Aiming at the facade and scoring a strike at
6 m is the same statement about the world with the arithmetic done right.

## Which engagements are winnable, and why that is stated in advance

A head-on intruder is stopped only if it is detected far enough out. With the
interceptor 1.5x faster, both flying straight, and the building ``d`` metres
away on the inbound line, the interceptor arrives first only when acquisition
happens beyond ``d (1 + 1/1.5) = 1.67 d`` -- so a building 45 m out needs
detection at 75 m, and one at 71 m needs 118 m. At the ring's resolution those
are 4.5 px and 2.9 px respectively.

That is a *prediction*, written down before the runs, and the point of writing
it down is that the failures then mean something: an engagement lost against a
far building on a head-on bearing is the acquisition envelope doing exactly what
this arithmetic says, while one lost against a near building on an oblique
bearing is a bug.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .episode import ScenarioConfig

BUILDINGS_JSON = (Path(__file__).resolve().parents[1]
                  / "simulators/pegasus/rivermark_buildings.json")

STATION_ALTITUDE_M = 30.0
"""Height above ground the interceptor holds while watching.

High enough to see over the plaza blocks it is standing among -- the nearest is
13.9 m and the interceptor is directly above part of its footprint -- and low
enough that the slant range to a building 71 m away is barely longer than the
ground range, so altitude is not quietly spending the detection budget.
"""

STRIKE_RADIUS_M = 6.0
AIM_HEIGHT_FRACTION = 0.75
"""Where on the facade the intruder aims: three quarters of the way up, so the
aim point is unambiguously *on the building* rather than on its roofline or in
the street at its foot."""


def load_buildings(path: Optional[Path] = None) -> List[dict]:
    """Read the surveyed structure list."""
    p = Path(path) if path else BUILDINGS_JSON
    return json.loads(p.read_text(encoding="utf-8"))["buildings"]


RELIABLE_DETECT_M = 140.0
"""Range at which the ring detects an inbound intruder on most frames.

**Measured, on this scene, with this camera** -- ``pursuit.tools.ring_detect_range``
against a 12 m/s intruder, 50 percent of frames or better. Not a specification
and not a hope: change the resolution, the scene or the detector and this
changes with them, which is why the defended radius below is derived from it
rather than written down separately.
"""

REACTION_S = 1.2
"""Seconds between the intruder becoming detectable and the interceptor making
good its full speed: the corroborated seed, the promotion that licenses steering,
and the second it takes an aircraft at 21 m/s^2 to reach 18 m/s from a hover."""


def defended_radius(detect_m: float = RELIABLE_DETECT_M,
                    intruder_mps: float = 12.0,
                    advantage: float = 1.5) -> float:
    """How far from its station one interceptor can actually defend, metres.

    The whole mission in one line. Both aircraft on the same line, the
    interceptor starting from a hover the moment the intruder becomes
    detectable at ``R``: they meet at ``R * v_c / (v_c + v_i)``, and the
    interceptor wins only if that is beyond the structure. With a 1.5x speed
    advantage that is ``0.6 R``, less whatever the reaction costs.

    Two things follow, and the second is the one worth internalising. Detection
    range is not one figure of merit among several -- it converts to defended
    radius at 0.6 metres per metre, which is a steeper exchange rate than
    anything else in the system offers. And a *faster* interceptor helps less
    than it looks: going from 1.5x to 2x moves the coefficient from 0.60 to
    0.67, while a third more detection range moves the radius by a third.
    """
    v_c = float(intruder_mps) * float(advantage)
    return (float(detect_m) * v_c / (v_c + float(intruder_mps))
            - float(intruder_mps) * REACTION_S)


def _corners(b: dict) -> List[Tuple[float, float]]:
    cx, cy = b["xy_rel"]
    hw, hd = b["footprint_m"][0] / 2.0, b["footprint_m"][1] / 2.0
    return [(cx + sx * hw, cy + sy * hd)
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]


def enclosing_circle(points: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], float]:
    """Smallest circle containing every point: ``((x, y), radius)``.

    Exact, by enumeration over the pairs and triples that can define it. There
    are twenty-odd corners here, so the cubic cost is microseconds and the
    alternative -- a heuristic that is usually right -- would put the
    interceptor's station somewhere slightly wrong for reasons nobody would ever
    trace.
    """
    pts = [(float(x), float(y)) for x, y in points]
    if not pts:
        return ((0.0, 0.0), 0.0)

    def circ(c, r):
        return all(math.dist(p, c) <= r + 1e-9 for p in pts)

    best = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            c = ((pts[i][0] + pts[j][0]) / 2.0, (pts[i][1] + pts[j][1]) / 2.0)
            r = math.dist(pts[i], c)
            if circ(c, r) and (best is None or r < best[1]):
                best = (c, r)
    if best is not None:
        return best
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            for k in range(j + 1, len(pts)):
                c = _circumcentre(pts[i], pts[j], pts[k])
                if c is None:
                    continue
                r = math.dist(pts[i], c)
                if circ(c, r) and (best is None or r < best[1]):
                    best = (c, r)
    return best or ((0.0, 0.0), 0.0)


def _circumcentre(a, b, c):
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-9:
        return None
    ax, ay = a
    bx, by = b
    cx, cy = c
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    return (ux, uy)


def station_and_set(buildings: Optional[Sequence[dict]] = None,
                    radius_m: Optional[float] = None
                    ) -> Tuple[Tuple[float, float], List[dict], float]:
    """Where to hold, and what that station can actually protect.

    Returns ``(station_xy_rel, buildings, enclosing_radius)``.

    The defended set is not chosen by taste. It is the **largest** set of
    structures whose every corner fits inside one circle of the measured
    defended radius, and the station is that circle's centre -- standing
    anywhere else makes the worst building worse. Enumerated rather than grown
    greedily, because nearest-first is not optimal here and it is worth being
    exact about: taking Rivermark's closest block first commits the circle to
    the north-west and then excludes two blocks that a slightly different centre
    would have covered.

    The result is a statement about the *system*, not about Rivermark: one
    interceptor, 1.5x faster, seeing 140 m, covers a 70 m bubble. Whatever ends
    up outside it is outside it, and a suite that pretended otherwise would be
    measuring an impossibility and reporting it as a failure of the guidance
    law.
    """
    bs = [b for b in (buildings if buildings is not None else load_buildings())
          if b["range_m"] <= 120.0]
    limit = defended_radius() if radius_m is None else float(radius_m)
    best: Tuple[int, float, Tuple[float, float], List[dict]] = (0, 0.0, (0.0, 0.0), [])
    for mask in range(1, 1 << len(bs)):
        subset = [bs[i] for i in range(len(bs)) if mask >> i & 1]
        if len(subset) < best[0]:
            continue
        c, r = enclosing_circle([p for b in subset for p in _corners(b)])
        if r > limit:
            continue
        if (len(subset), -r) > (best[0], -best[1]):
            best = (len(subset), r, c, subset)
    if not best[3]:
        one = min(bs, key=lambda b: max(math.hypot(*p) for p in _corners(b)))
        c, r = enclosing_circle(_corners(one))
        return c, [one], r
    return best[2], sorted(best[3], key=lambda b: b["range_m"]), best[1]


def defended(buildings: Optional[Sequence[dict]] = None,
             max_range_m: Optional[float] = None) -> List[dict]:
    """The structures the interceptor is responsible for.

    Derived from the measured detection range by default (see
    :func:`station_and_set`); ``max_range_m`` overrides it with a plain
    centre-distance cut, which is what the ``city-wide`` suite uses to measure
    what happens outside the envelope.
    """
    bs = buildings if buildings is not None else load_buildings()
    if max_range_m is not None:
        return [b for b in bs if b["range_m"] <= max_range_m]
    return station_and_set(bs)[1]


def facade_aim(building: dict, from_xy: Sequence[float]) -> Tuple[float, float]:
    """The point on the building's footprint nearest ``from_xy``.

    Clamped to the axis-aligned footprint, which is what the survey measures.
    An intruder that flew at the centre would have to pass through half the
    building to get there, and a strike radius large enough to cover that is
    large enough to cover the street outside.
    """
    cx, cy = building["xy_rel"]
    hw, hd = building["footprint_m"][0] / 2.0, building["footprint_m"][1] / 2.0
    return (min(max(float(from_xy[0]), cx - hw), cx + hw),
            min(max(float(from_xy[1]), cy - hd), cy + hd))


def nearest_target(buildings: Sequence[dict],
                   start_xy: Sequence[float]) -> dict:
    """The structure an intruder arriving at ``start_xy`` would go for.

    Nearest to *the intruder*, not to the station. That is what makes the
    arrival bearing matter: a drone coming in from the north-west is a threat to
    the north-west buildings, and the interceptor's problem is a different one
    in every quarter of the circle.
    """
    return min(buildings,
               key=lambda b: math.dist(facade_aim(b, start_xy), tuple(start_xy)))


def build_city_suite(base: ScenarioConfig, n: int = 24, seed: int = 900,
                     hard: bool = False,
                     buildings: Optional[Sequence[dict]] = None,
                     nose_relative_deg: Optional[float] = None,
                     label: str = "city") -> List[ScenarioConfig]:
    """``n`` arrivals spread evenly around the compass, one scenario each.

    Evenly spaced *and* jittered: the spacing guarantees the whole circle is
    covered, including the quarters with no building in them, and the jitter
    stops every engagement being flown at a bearing that happens to be a
    multiple of 15 degrees. The interceptor's own heading is drawn at random
    too, which is the cheapest possible check that a ring works regardless of
    which way the airframe is facing -- the exact property the single camera did
    not have.

    Args:
        base: Defaults to build on (speed advantage, hit radius, ...).
        n: How many arrival bearings.
        seed: Draws every per-scenario quantity. A named scenario is the same
            engagement every time it is run.
        hard: Faster intruders that jink harder on the way in, from further
            out. The same station and the same structures -- a "hard" variant
            that also moved the buildings outside the envelope would be
            measuring the envelope again rather than the defence, and the
            envelope is already arithmetic (:func:`defended_radius`).
        buildings: Override the defended set (tests use this).
        nose_relative_deg: Pin every arrival to this angle off the airframe's
            own nose instead of drawing the heading at random. The random draw
            is the right default because it *samples* the property a ring has
            and a single camera does not; pinning it to 180 instead
            *demonstrates* one corner of that property -- an intruder arriving
            dead astern, which the nose camera could not have seen at all.
        label: Scenario-name prefix, so a pinned suite does not collide with
            the sampled one in the same results directory.
    """
    if buildings is not None:
        bs, station = list(buildings), (0.0, 0.0)
    else:
        station, bs, _rad = station_and_set()
    if not bs:
        raise ValueError("no defended structures -- is the building survey present?")
    rng = random.Random(seed)
    out: List[ScenarioConfig] = []
    for i in range(n):
        bearing = (360.0 / n) * i
        jittered = bearing + rng.uniform(-4.0, 4.0)
        r0 = rng.uniform(155.0, 190.0) * (1.12 if hard else 1.0)
        chaser_yaw = rng.uniform(-180.0, 180.0)
        if nose_relative_deg is not None:
            # Drawn and then discarded on purpose: the draw keeps the random
            # stream -- and therefore every other scenario quantity -- identical
            # to the sampled suite, so the two differ in heading and in nothing
            # else.
            chaser_yaw = jittered - float(nose_relative_deg)
        ang = math.radians(jittered)
        # Bearings are measured from the station, not from the scene origin --
        # the station is where the interceptor is and where the arrival circle
        # is centred, and the two are the same point only by coincidence.
        start = (station[0] + r0 * math.cos(ang), station[1] + r0 * math.sin(ang))
        target = nearest_target(bs, start)
        aim = facade_aim(target, start)
        speed = rng.uniform(11.0, 13.0) + (2.5 if hard else 0.0)
        aim_z = AIM_HEIGHT_FRACTION * target["height_agl_m"]
        # The intruder cruises in **above** the interceptor and dives onto its
        # target. That is how a strike is flown, and it is also the difference
        # between an engagement and a walkover: measured on the first live runs,
        # an intruder coming in at 24 m against an interceptor holding at 30 m
        # is seen *downward*, against roofs and roads, and the motion detector
        # first found it at 121 m instead of the 160 m it manages against sky.
        # Sixty-two metres, and the number is set by an angle rather than by
        # taste. What matters is whether the target is above the *horizon* from
        # the interceptor's camera, because above it the background is sky and
        # below it the background is a city. At 150 m an intruder cruising at
        # 42 m sits 3.8 degrees up -- which is above the horizon line and still
        # against the distant hills, and the motion detector found it only
        # sporadically. At 62 m it sits 9.5 degrees up, clean sky, where the
        # same detector reaches 140 m at a detection rate of 0.88.
        #
        # It descends onto the facade from there, so the geometry decays through
        # the run: sky for the acquisition, clutter for the terminal seconds --
        # which is the right way round, because that is where the appearance
        # model takes over.
        cruise_agl = 62.0 + 5.0 * (i % 3)
        start_elev = math.degrees(math.atan2(cruise_agl - STATION_ALTITUDE_M, r0))
        out.append(replace(
            base,
            name=f"{label}-{int(round(bearing)):03d}" + ("-hard" if hard else ""),
            entry=f"{int(round(bearing)):03d}deg",
            policy="weave",          # only used if something reveals it
            seed=seed + i,
            ingress=True,
            chaser_offset_xy=station,
            chaser_yaw_deg=chaser_yaw,
            altitude_m=STATION_ALTITUDE_M,
            start_range_m=r0,
            # start_bearing_deg is measured from the airframe's own nose, so a
            # world arrival bearing has to have the (random) heading taken back
            # out of it. Skip that and the interceptor is silently always
            # pointing at the intruder -- which would make a 360 degree sensor
            # look exactly as good as a 76 degree one.
            start_bearing_deg=((jittered - chaser_yaw + 180.0) % 360.0) - 180.0,
            start_elevation_deg=start_elev,
            defend_xy=(aim[0], aim[1]),
            defend_height_m=aim_z,
            defend_label=target["name"],
            strike_radius_m=STRIKE_RADIUS_M,
            strike_commit=True,
            strike_evade=(0.45 if hard else 0.25),
            transit_speed=speed,
            evader_speed=speed,
            speed_advantage=1.5,
            reveal_range_m=0.0,
            # The interceptor has been on station; let its cameras have seen
            # the city before the intruder appears. See ScenarioConfig.
            calibrate_frames=120,
            # Every one of these engagements is over inside fifteen seconds --
            # the intruder's whole run is at most 132 m at 11-13 m/s, and it
            # either arrives or is stopped. A longer wall costs nothing when the
            # loop is arithmetic and fifteen minutes a scenario when it is a
            # renderer.
            max_seconds=30.0,
        ))
    return out


def city_guidance(base=None, top_speed: float = 18.0):
    """Guidance sized for this airframe and this sensor.

    Three of :class:`~pursuit.guidance.GuidanceConfig`'s defaults are absolute
    speeds tuned against a 14.4 m/s interceptor (an evader at 9 m/s and a 1.6x
    advantage). Carried over unchanged to an 18 m/s aircraft they throttle it to
    14 -- a fifth of the closing speed given away silently, which costs nothing
    in a suite where the interceptor has all day and costs the building in one
    where it does not.

    ``approach_speed`` goes to zero, which means *the airframe's own limit* and
    is therefore right per scenario rather than right on average; the other two
    scale with the fastest engagement in the suite. ``omnidirectional`` comes
    with them because a ring and a step-and-stare search are not compatible.

    Args:
        base: Configuration to start from.
        top_speed: The fastest interceptor in the suite, m/s.
    """
    from dataclasses import replace as _replace

    from .guidance import GuidanceConfig

    v = float(top_speed)
    return _replace(
        base or GuidanceConfig(),
        omnidirectional=True,
        approach_speed=0.0,
        terminal_speed=v * 1.1,
        # The same fraction of the speed budget the tuned configuration gave to
        # across-LOS correction (9.0 of 14.4), not the same absolute number.
        max_lateral_speed=0.625 * v,
    )


def city_top_speed(scenarios: Sequence[ScenarioConfig]) -> float:
    """Fastest interceptor across a suite, for the speeds that must be absolute."""
    return max((sc.evader_speed * sc.speed_advantage for sc in scenarios),
               default=18.0)


def map_overlay(sc: ScenarioConfig, origin_xy: Sequence[float]) -> Tuple[list, Optional[tuple]]:
    """Structures and aim point in **world** coordinates, for the top-down map.

    The scenarios carry origin-relative positions because that is what makes
    them portable between the renderer and the sandbox; a map drawn over real
    flight paths needs world ones. Converting in one place keeps the two
    conventions from meeting anywhere else.
    """
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    try:
        bs = load_buildings()
    except (OSError, ValueError):
        return [], None
    world = [{"name": b["name"],
              "xy": [b["xy_rel"][0] + ox, b["xy_rel"][1] + oy],
              "footprint_m": b["footprint_m"], "defended": b["defended"]}
             for b in bs]
    aim = None if sc.defend_xy is None else (sc.defend_xy[0] + ox,
                                             sc.defend_xy[1] + oy)
    return world, aim


def city_geometry(sc: ScenarioConfig, origin_xy=(0.0, 0.0),
                  ground_z: float = 0.0) -> Dict[str, float]:
    """The arithmetic that says whether a scenario is winnable, per the docstring.

    Reported next to the result rather than kept in someone's head: an
    engagement lost with ``needed_acquire_m`` above what the sensor can do is
    the acquisition envelope, and one lost with it comfortably inside is a bug.
    """
    if sc.defend_xy is None:
        return {}
    # Everything is measured from the *station*, which is where the interceptor
    # is; the scene origin is only the coordinate frame the survey happens to
    # be written in.
    st = tuple(float(v) for v in sc.chaser_offset_xy)
    d_asset = math.dist(tuple(sc.defend_xy), st)
    # Distance the intruder still has to fly, from where it starts. The world
    # arrival bearing is the airframe's heading plus the scenario's, exactly as
    # place_engagement composes them.
    ang = math.radians(sc.chaser_yaw_deg + sc.start_bearing_deg)
    start = (st[0] + sc.start_range_m * math.cos(ang),
             st[1] + sc.start_range_m * math.sin(ang))
    d_run = math.dist(start, tuple(sc.defend_xy))
    v_i = sc.transit_speed or sc.evader_speed
    v_c = sc.evader_speed * sc.speed_advantage
    return {
        "asset_range_m": round(d_asset, 1),
        "intruder_run_m": round(d_run, 1),
        "obliquity_m": round(d_run - (sc.start_range_m - d_asset), 1),
        "time_to_strike_s": round(d_run / max(1e-6, v_i), 2),
        "time_to_asset_s": round(d_asset / max(1e-6, v_c), 2),
        # Head-on worst case: acquisition must beat d(1 + v_i/v_c).
        "needed_acquire_m": round(d_asset * (1.0 + v_i / v_c), 1),
    }
