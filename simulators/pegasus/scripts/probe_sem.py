"""Dump the raw bounding_box_2d_tight payload in rivermark so the semantic
filtering can be written against what Isaac actually returns."""
import json, math, sys, time
from pathlib import Path
sys.path.insert(0, "/tmp/dev/dronedet")

from simulators.pegasus.camera import load_intrinsics
intr = load_intrinsics(Path("/tmp/dev/platform/robots/PEGASUS/config"),
                       resolution=(1440, 840))
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from isaacsim.core.api import World
from simulators.pegasus.drone import KinematicDrone
from simulators.pegasus.scenes.outdoor import load_outdoor_scene, scene_ground_z, scene_origin_xy

world = World(physics_dt=0.05, rendering_dt=0.05, stage_units_in_meters=1.0)
load_outdoor_scene(app, "rivermark", progress=lambda m: print(m, flush=True))
gz = scene_ground_z("rivermark"); ox, oy = scene_origin_xy("rivermark")
root = Path("/tmp/dev/PegasusSimulator/extensions/pegasus.simulator")
ch = KinematicDrone(root, "/World/chaser", intr, position=(ox, oy, gz + 20), yaw=0.0)
tg = KinematicDrone(root, "/World/target", intr, position=(ox + 25, oy, gz + 20),
                    yaw=math.pi, with_camera=False)
import omni.replicator.core as rep
with rep.get.prims(path_pattern="/World/target"):
    rep.modify.semantics([("class", "drone")])
ch.camera.add_bounding_box_2d_tight_to_frame()
world.reset()
for _ in range(140):
    world.step(render=True)

out = {}
# Two poses: one looking at open sky, one deliberately looking down at the town.
for name, (cz, yaw, pitchless_target_dz) in {
        "sky": (gz + 30.0, 0.0, 0.0),
        "town": (gz + 12.0, 0.9, -3.0)}.items():
    ch.set_pose((ox, oy, cz), yaw)
    tg.set_pose((ox + 25 * math.cos(yaw), oy + 25 * math.sin(yaw),
                 cz + pitchless_target_dz), math.pi)
    for _ in range(6):
        world.step(render=True)
    f = ch.camera.get_current_frame()
    raw = f.get("bounding_box_2d_tight")
    rec = {"payload_type": str(type(raw))}
    if isinstance(raw, dict):
        rec["keys"] = sorted(str(k) for k in raw.keys())
        info = raw.get("info")
        rec["info_type"] = str(type(info))
        if isinstance(info, dict):
            rec["info_keys"] = sorted(str(k) for k in info.keys())
            rec["idToLabels"] = {str(k): str(v) for k, v in
                                 (info.get("idToLabels") or {}).items()}
        data = raw.get("data")
    else:
        data = raw
    rows = []
    try:
        for r in list(data)[:12]:
            rows.append([float(x) for x in list(r)[:6]])
    except Exception as e:
        rec["rows_error"] = f"{type(e).__name__}: {e}"
    rec["n_rows"] = 0 if data is None else len(data)
    rec["rows"] = rows
    out[name] = rec
    print(name, json.dumps(rec)[:1200], flush=True)

Path("/tmp/dev/pursuit/probe_sem.json").write_text(json.dumps(out, indent=1))
print("PROBE_SEM_OK", flush=True)
app.close()
