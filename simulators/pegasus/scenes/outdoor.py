"""Outdoor environments for the Pegasus/Isaac Sim air-to-air rig.

The external PEGASUS platform only registers *indoor* scenes
(``robots/PEGASUS/adapters/scene.py``'s ``INDOOR_SCENES``) because everything it
was built for -- flying a drone around furniture to collect navigation data --
happens indoors. Air-to-air detection is the opposite problem: the target is
against sky, roofline and distant terrain, and none of that exists in an office.

Three outdoor options are registered here, in descending order of realism and
descending order of what they cost to load:

``rivermark``
    NVIDIA's own outdoor town (roads, pavement, houses, foliage, street
    furniture), from the Isaac 4.5 asset pack. Verified: ``metersPerUnit = 1.0``,
    ``upAxis = Z``, and it ships its own ``/World/Sky`` dome light with a 22.6 MB
    HDRI, so it needs no lighting added. It is also by far the heaviest thing
    here -- see :data:`SCENE_LOAD_BUDGET_S`.

``skydome``
    A ground plane under a photographic HDRI sky. No 3D scenery at all, which
    sounds like a downgrade and mostly is not: a drone at altitude is seen
    against sky far more often than against buildings, and this loads in
    seconds instead of minutes. It is the right scene for isolating "can we see
    a 20-pixel drone against bright sky" from "can we see it against clutter".

``rivermark_props``
    A middle path: the skydome, plus a hand-placed handful of Rivermark's *own*
    house/tree/lamp assets referenced individually. Gives real ground clutter
    under the flight path without paying for the whole town.

**Which to use depends on what you are testing**, which is why all three stay.
Detection against sky and detection against a rooftop are different problems and
a single scene cannot pose both.
"""
from __future__ import annotations

# The Isaac asset pack. Pinned to 4.5: it is the newest pack in the bucket
# (confirmed by listing Assets/Isaac/ -- there is no 5.x or 6.x), and Isaac Sim
# 6.0.1 reads it fine.
ISAAC_ASSETS = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/4.5/Isaac"
)

# The DomeLights/HDRI pack lives under a DIFFERENT top-level prefix than the
# Isaac asset pack -- it is Omniverse content, not Isaac content. This cannot be
# built by appending to ISAAC_ASSETS, which is why it is its own constant.
OMNI_ENVIRONMENTS = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Environments/2024_1"
)

RIVERMARK_USD = f"{ISAAC_ASSETS}/Environments/Outdoor/Rivermark/rivermark.usd"
RIVERMARK_CONTENT = (
    f"{ISAAC_ASSETS}/Environments/Outdoor/Rivermark/dsready_content/nv_content"
)

TERRAIN_FLAT = f"{ISAAC_ASSETS}/Environments/Terrains/flat_plane.usd"
TERRAIN_ROUGH = f"{ISAAC_ASSETS}/Environments/Terrains/rough_plane.usd"

SKY_HDRI = {
    "noon_grass": f"{OMNI_ENVIRONMENTS}/DomeLights/Clear/noon_grass.hdr",
    "lakeside": f"{OMNI_ENVIRONMENTS}/DomeLights/Clear/lakeside.hdr",
    "mealie_road": f"{OMNI_ENVIRONMENTS}/DomeLights/Clear/mealie_road.hdr",
    "sunrise": f"{OMNI_ENVIRONMENTS}/DomeLights/Clear/signal_hill_sunrise.hdr",
    "clear": f"{OMNI_ENVIRONMENTS}/DomeLights/Clear/syferfontein_18d_clear.hdr",
    "partly_cloudy": f"{OMNI_ENVIRONMENTS}/DomeLights/Cloudy/kloofendal_48d_partly_cloudy.hdr",
    "cloudy": f"{OMNI_ENVIRONMENTS}/DomeLights/Cloudy/table_mountain_1.hdr",
    "overcast": f"{OMNI_ENVIRONMENTS}/DomeLights/Cloudy/white_cliff_top.hdr",
    "evening": f"{OMNI_ENVIRONMENTS}/DomeLights/Evening/kloppenheim_02.hdr",
}
"""Photographic skies. The default is deliberately a *bright* one.

A drone against bright sky is the hardest exposure case and the most common real
one; picking an overcast HDRI because it looks nicer in a screenshot would make
the detector's job easier than reality does.
"""

# Rivermark's own props, referenced one at a time. These are the assets the town
# is built from, so the clutter is the same clutter -- it is only the count that
# is smaller.
RIVERMARK_PROPS = {
    "house_1": f"{RIVERMARK_CONTENT}/usa/country_assets/props_structures/kasa_house_01/kasa_house_01.usd",
    "house_2": f"{RIVERMARK_CONTENT}/usa/country_assets/props_structures/kasa_house_02/kasa_house_02.usd",
    "house_3": f"{RIVERMARK_CONTENT}/usa/country_assets/props_structures/kasa_house_03/kasa_house_03.usd",
    "bush": f"{RIVERMARK_CONTENT}/common_assets/props_vegetation/bush_gen_06/bush_gen_06.usd",
    "hedge": f"{RIVERMARK_CONTENT}/common_assets/props_vegetation/hedge_round_01/hedge_round_01.usd",
    "grass_clump": f"{RIVERMARK_CONTENT}/common_assets/props_vegetation/grass_clump_01/grass_clump_01.usd",
    "street_lamp": f"{RIVERMARK_CONTENT}/common_assets/props_poles/gen_street_lamp_01/gen_street_lamp_01.usd",
}

OUTDOOR_SCENES = ("rivermark", "skydome", "rivermark_props")

SCENE_LOAD_BUDGET_S = {
    # Measured, not guessed: the whole town is a payload plus 17 sublayers
    # totalling ~38 MB, plus a 22.6 MB HDRI, plus every asset those layers
    # reference, all streamed over HTTPS and then compiled by the RTX material
    # system. A first (cold-cache) load ran past 15 minutes without finishing.
    # Isaac Sim's own /root/.cache/ov is a persistent bind mount, so the second
    # load of the same scene is far cheaper than the first.
    "rivermark": 3000.0,
    "skydome": 300.0,
    "rivermark_props": 900.0,
}

# Where the ground actually is. Rivermark's drivable surfaces are NOT at z=0 --
# they sit 4.6..9.5 m up in world space. Spawning an aircraft at the usual
# z=0.15 puts it five metres UNDERNEATH the town, looking at the underside of
# the terrain, with the autopilot perfectly happy because it thinks it landed.
SCENE_GROUND_Z = {
    "rivermark": 5.2,
    "skydome": 0.0,
    "rivermark_props": 0.0,
}

# A spot with real scenery around and under it. Rivermark's usable footprint is
# x -353..+536, y -293..+454, so every origin-centred default in the PEGASUS
# harness is wrong for it.
SCENE_ORIGIN_XY = {
    "rivermark": (60.0, 60.0),
    "skydome": (0.0, 0.0),
    "rivermark_props": (0.0, 0.0),
}


def scene_ground_z(scene: str) -> float:
    """World-frame height of the ground the drones fly above, metres."""
    _check(scene)
    return SCENE_GROUND_Z[scene]


def scene_origin_xy(scene: str) -> tuple:
    """A point in the scene with real geometry around it, ``(x, y)``."""
    _check(scene)
    return SCENE_ORIGIN_XY[scene]


def _check(scene: str) -> None:
    if scene not in OUTDOOR_SCENES:
        raise KeyError(f"Unknown outdoor scene {scene!r}; choose from {list(OUTDOOR_SCENES)}")


def load_outdoor_scene(simulation_app, scene: str, sky: str = "clear",
                       prim_path: str = "/World/Scene", load_timeout_s: float = None,
                       progress=print) -> dict:
    """Reference an outdoor environment onto the current stage and wait for it.

    Waiting is the whole job. ``add_reference_to_stage`` returns as soon as the
    reference is *authored*, not when its content has arrived, and the PEGASUS
    harness's usual "tick the app 20 times and carry on" is tuned for one small
    indoor USD. Give Rivermark 20 ticks and you get a half-composed town: prim
    queries see nothing, bounding boxes are empty, and the first rendered frames
    are of a scene that is not there yet -- none of which raises. This polls
    ``is_stage_loading()`` instead, so the wait is as long as the scene needs
    and no longer.

    Args:
        simulation_app: The live ``SimulationApp``.
        scene: One of :data:`OUTDOOR_SCENES`.
        sky: A key of :data:`SKY_HDRI`. Ignored by ``rivermark``, which ships
            its own sky.
        prim_path: Stage path to reference the environment at.
        load_timeout_s: Give up waiting after this long. None uses
            :data:`SCENE_LOAD_BUDGET_S`.
        progress: Called with progress strings; loading a town takes minutes and
            silence is indistinguishable from a hang.

    Returns:
        A dict describing what was loaded (``scene``, ``ground_z``,
        ``load_seconds``, ``prims``, ``meshes``).

    Raises:
        KeyError: If ``scene`` is not a known outdoor scene.
        RuntimeError: If the stage came back empty -- see the note below.
    """
    import time

    _check(scene)
    budget = load_timeout_s if load_timeout_s is not None else SCENE_LOAD_BUDGET_S[scene]
    started = time.time()

    if scene == "rivermark":
        _reference(RIVERMARK_USD, prim_path)
    else:
        _reference(TERRAIN_FLAT, f"{prim_path}/Ground")
        add_sky_dome(SKY_HDRI[sky])
        if scene == "rivermark_props":
            _scatter_props(prim_path)

    _drain_loads(simulation_app, budget, started, progress)

    # An unreachable or mistyped USD URL does not raise -- add_reference_to_stage
    # authors the reference and composition quietly resolves to nothing, leaving
    # a valid but childless Xform. The failure then shows up as an all-black
    # recording an hour later. Assert the geometry is actually here instead.
    stats = _stage_stats(prim_path)
    if stats["meshes"] == 0:
        raise RuntimeError(
            f"scene {scene!r} composed to {stats['prims']} prims but ZERO meshes -- "
            f"the USD reference resolved to nothing. Check the asset URL is "
            f"reachable from inside the container."
        )

    elapsed = time.time() - started
    progress(f"scene {scene!r} loaded in {elapsed:.1f}s: "
             f"{stats['prims']} prims, {stats['meshes']} meshes, {stats['lights']} lights")
    return {
        "scene": scene,
        "ground_z": SCENE_GROUND_Z[scene],
        "load_seconds": elapsed,
        **stats,
    }


def _reference(usd_path: str, prim_path: str) -> None:
    from isaacsim.core.utils.stage import add_reference_to_stage

    add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)


def _drain_loads(simulation_app, budget_s: float, started: float, progress) -> None:
    """Tick the app until the stage stops loading, or the budget runs out."""
    import time

    from isaacsim.core.utils.stage import is_stage_loading

    ticks = 0
    last = 0.0
    while is_stage_loading():
        simulation_app.update()
        ticks += 1
        now = time.time()
        if now - last > 20.0:
            last = now
            progress(f"  ... still streaming, {now - started:.0f}s / {budget_s:.0f}s budget, "
                     f"{ticks} ticks")
        if now - started > budget_s:
            progress(f"  ... LOAD BUDGET EXHAUSTED after {budget_s:.0f}s -- continuing anyway, "
                     f"the scene may be incomplete")
            return
    # Hydra/material work continues briefly after is_stage_loading() goes false.
    for _ in range(60):
        simulation_app.update()


def _stage_stats(prim_path: str) -> dict:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    prims = meshes = lights = 0
    for p in stage.Traverse():
        prims += 1
        t = str(p.GetTypeName())
        if t == "Mesh":
            meshes += 1
        elif "Light" in t:
            lights += 1
    return {"prims": prims, "meshes": meshes, "lights": lights,
            "root_valid": bool(root.IsValid())}


def add_sky_dome(hdri_url: str, prim_path: str = "/World/SkyDome",
                 intensity: float = 1000.0) -> None:
    """Add a photographic sky as a dome light.

    Without this a ``Terrains/*.usd`` scene has no light source at all, and --
    the part that costs a day if you miss it -- that failure is *silent*. Depth
    comes from a separate AOV and does not care about lighting, so a lightless
    scene still writes a complete recording, with plausible poses and perfectly
    good depth, and an entirely black RGB stream.

    Args:
        hdri_url: A ``.hdr`` from :data:`SKY_HDRI`.
        prim_path: Stage path for the dome light.
        intensity: Dome light intensity.
    """
    import omni.usd
    from pxr import Gf, UsdLux

    stage = omni.usd.get_context().get_stage()
    dome = UsdLux.DomeLight.Define(stage, prim_path)
    dome.CreateIntensityAttr(intensity)
    dome.CreateTextureFileAttr(hdri_url)
    dome.CreateTextureFormatAttr("latlong")
    # No rotation op. Measured, not assumed: a 90-degree RotateX was tried first
    # on the reasoning that a Y-up HDRI needs turning onto a Z-up stage, and it
    # put the photographed *ground* across the top of the sky -- the horizon
    # ended up vertical. Isaac Sim already orients a latlong dome for the
    # stage's up-axis, so the correct rotation here is none.
    return dome


def _scatter_props(prim_path: str) -> None:
    """Place a handful of Rivermark's own assets as ground clutter.

    Hand-placed rather than randomised: this exists to put *known* structures
    under a *known* flight path, so that "the target crossed a roofline here"
    is a repeatable statement about a recording rather than a seed.
    """
    layout = [
        ("house_1", (18.0, 12.0, 0.0), 0.0),
        ("house_2", (-16.0, 14.0, 0.0), 180.0),
        ("house_3", (22.0, -14.0, 0.0), 90.0),
        ("house_1", (-20.0, -18.0, 0.0), 270.0),
        ("hedge", (6.0, 8.0, 0.0), 0.0),
        ("hedge", (-6.0, 8.0, 0.0), 0.0),
        ("bush", (10.0, -5.0, 0.0), 0.0),
        ("bush", (-11.0, -6.0, 0.0), 45.0),
        ("grass_clump", (3.0, -9.0, 0.0), 0.0),
        ("street_lamp", (12.0, 0.0, 0.0), 0.0),
        ("street_lamp", (-12.0, 0.0, 0.0), 180.0),
    ]
    from pxr import Gf, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for i, (key, (x, y, z), yaw_deg) in enumerate(layout):
        path = f"{prim_path}/Props/prop_{i:02d}_{key}"
        _reference(RIVERMARK_PROPS[key], path)
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        xform.AddRotateZOp().Set(yaw_deg)
