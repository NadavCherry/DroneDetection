# Pegasus — outdoor air-to-air rig (Isaac Sim)

Two drones in a photorealistic outdoor town. One hovers and watches; the other
sweeps a straight line in front of it. The output is a side-by-side recording of
**both aircraft's own onboard cameras**, with per-frame ground truth for where
the target actually was.

That was stage one of the intercept pipeline: *see the other drone*. The same
scenes, camera and airframe now also serve the stage that closes on it —
`scripts/pursuit_server.py` holds the simulator open as a long-lived server and
steps it one control tick at a time on request, so the guidance loop in
[`pursuit/`](../../pursuit/README.md) can fly against rendered pixels from a
process that has never imported Isaac. `--cameras ring` swaps the single nose
camera for four wide ones at 90° spacing.

```
simulators/pegasus/
  scenes/outdoor.py            outdoor environments + HDRI skies (no Isaac import at module level)
  camera.py                    the platform calibration, the ring fit, and projecting a world point
  drone.py                     KinematicDrone: the Iris airframe + its onboard camera
  control/trajectories.py      Hover / LineSweep / Orbit -- position as a function of time
  recording/split_recorder.py  two camera streams -> one side-by-side MP4 + per-frame JPEGs
  pursuit_proto.py             the wire protocol, and the ONLY module both processes run
                               -- so it stays dependency-free ("Two processes, one socket")
  rivermark_buildings.json     the surveyed buildings the city-defence mission defends
  scripts/run_two_drone.py     the recording entry point (runs INSIDE the container)
  scripts/pursuit_server.py    the closed-loop entry point: one episode per connection,
                               nose camera or --cameras ring (runs INSIDE the container)
  scripts/find_buildings.py    surveys Rivermark's meshes -> rivermark_buildings.json
  scripts/probe_gt.py          checks a projected ground-truth pixel against the render
  scripts/probe_sem.py         semantic/instance-segmentation probe of a loaded scene
  scripts/run_in_container.sh  host-side launcher: sync in, run, copy out
  scripts/annotate_recording.py  host-side: ground-truth boxes + magnified insets, H.264
```

## Prerequisites — what is, and is not, in this repository

**The simulator is not.** Everything under `scripts/` that runs *inside* the
container needs two things this repository does not contain and cannot ship:

- **An Isaac Sim 6.0.1 container** running under the name `isaac-sim`, with a
  directory bind-mounted from the host — `/tmp/dev` inside by default,
  overridable with `ISAAC_DEV_ROOT` — because the closed loop meets the host
  brain on a Unix socket inside it. NVIDIA distributes the image, and the USD
  assets Rivermark streams on first load come from their asset servers.
- **The PEGASUS platform**, a separate internal robotics codebase, which owns
  the Iris asset, the camera calibration YAML, and the Isaac Sim 6.0.1 / PX4
  compatibility work this rig is built on. It is not public and there is no
  acquisition path from here.

So an outside reader cannot run `run_in_container.sh`, `run_two_drone.py` or
`pursuit_server.py`, and this section exists to say so before the Quick Start
below wastes anyone's afternoon. Three things do still work without any of it:

- the fast pursuit loop, `.venv/bin/python -m pursuit.sandbox --suite city --ring`,
  which flies the same guidance law over the same ring geometry with arithmetic
  instead of a renderer — self-contained, and it finishes in seconds;
- `camera.py`, whose ring geometry the ordinary `pytest` run exercises directly —
  `pursuit/tests/test_ring.py` imports `ring_intrinsics`, `ring_mounts` and
  `ring_coverage_deg` from it, so the seam arithmetic below is checked on a
  machine with no simulator at all. (`control/trajectories.py` and
  `pursuit_proto.py` import nothing outside the standard library either, but
  neither is under test: the first is only imported by `run_two_drone.py`
  inside the container, the second only by host-side tools that need a
  simulator on the other end of the socket.)
- the ring's own calibration, which is worth stating separately because it is
  the surprising part: `ring_intrinsics()` derives square-pixel, centred
  intrinsics from a chosen HFOV instead of reading the platform YAML — a ring
  watching a whole horizon should not inherit the downward tilt of a different
  sensor — so the **city-defence mission needs no external calibration at all**.
  `camera_pegasus_iris_720x420.yaml`, and therefore `PEGASUS_CONFIG_DIR`, is
  read only on the nose-camera path.

## Quick start

Given the container and the platform config above:

```bash
docker start isaac-sim          # if it is not already up
export PEGASUS_CONFIG_DIR=/path/to/robots/PEGASUS/config

simulators/pegasus/scripts/run_in_container.sh \
    --scene rivermark --seconds 30 --altitude 20 --standoff 15

.venv/bin/python simulators/pegasus/scripts/annotate_recording.py \
    simulators/pegasus/recordings/run_YYYYmmdd_HHMMSS
```

What comes out:

```
recordings/<run>/
  split_view.mp4     observer left, target right (mp4v -- see "no ffmpeg" below)
  annotated.mp4      the readable one: ground-truth boxes, x3 insets, H.264
  left/000000.jpg    observer's onboard camera, one file per frame
  right/000000.jpg   target's onboard camera
  frames.json        per frame: both positions, range, target's pixel position and span
  meta.json          scene, intrinsics, trajectories, how many frames had the target in view
```

`frames.json` is the point of the whole rig. Every frame carries where the
target *actually* was in the observer's image and how many pixels across it was,
so a detection can be scored against the range it happened at without anyone
having to estimate the range first.

`recordings/` is gitignored — hundreds of megabytes of MP4 and per-frame JPEG per
run — so nothing under it is in the repository and the layout above is what the
recorder writes, not something you can go and look at.

## The outdoor scene

`rivermark` is NVIDIA's own outdoor town from the Isaac 4.5 asset pack — roads,
a shopping plaza, apartment blocks, parking, street furniture, distant hills. It
composes at `metersPerUnit = 1.0`, `upAxis = Z`, and ships its own `/World/Sky`
dome light with a 22.6 MB HDRI, so it needs no lighting added. Measured on this
machine: **56,672 prims, 12,125 meshes, 669 lights, ~6.1 GB of the 8 GB GPU.**

Two other scenes are registered. `skydome` is a ground plane under a photographic
HDRI and loads in seconds — the right scene for isolating "can we see a small
drone against bright sky" from "can we see it against clutter".
`rivermark_props` is the skydome plus a handful of Rivermark's own house/hedge/
lamp assets, for ground clutter without the whole town.

### Things that cost time to find out

- **The first load of Rivermark takes far longer than the second.** Streaming a
  payload, 17 sublayers (~38 MB), the HDRI and every referenced asset over HTTPS
  ran past 15 minutes cold. Isaac Sim's `/root/.cache/ov` is a persistent bind
  mount, so the *second* load of the same scene took **25 seconds**. A first run
  that looks hung is usually not.
- **Wait on `is_stage_loading()`, never on a tick count.** The PEGASUS harness's
  `STAGE_SETTLE_STEPS = 20` is tuned for one small indoor USD. Give Rivermark 20
  ticks and you get a half-composed town — prim queries see nothing, bounding
  boxes come back empty, the first frames are of a scene that is not there yet,
  and none of it raises.
- **Rivermark's ground is not at z=0.** Its drivable surfaces sit at
  z = +4.6..+9.5 m and its footprint is x −353..+536, y −293..+454. Every
  origin-centred default in the PEGASUS harness is wrong here, including
  `add_collision_ground()`'s invisible plane at z=0 — spawn at the usual
  `z=0.15` and the aircraft rests five metres *underneath* the town.
  `SCENE_GROUND_Z` and `SCENE_ORIGIN_XY` carry the corrected values.
- **A mistyped USD URL does not raise.** `add_reference_to_stage` authors the
  reference, composition quietly resolves to nothing, and you get a valid but
  childless Xform — which surfaces as an all-black recording an hour later.
  `load_outdoor_scene` asserts a non-zero mesh count instead.
- **A scene with no light still records perfectly.** Depth comes from a separate
  AOV and does not care about lighting, so a lightless stage writes a complete
  recording with plausible poses and good depth and a completely black RGB
  stream. Rivermark ships its own sky; anything built on `Terrains/*.usd` needs
  `add_sky_dome` and needs it checked before a long run.
- **The sky dome takes no rotation.** A 90° `RotateX` was tried first, on the
  reasoning that a Y-up HDRI needs turning onto a Z-up stage. It puts the
  photographed *ground* across the top of the sky. Isaac Sim already orients a
  latlong dome for the stage's up-axis; the correct rotation is none.

## Why the drones are kinematic

`KinematicDrone` references the Iris **mesh** and moves it by setting its pose.
It does not use Pegasus's `Multirotor`, which would bring a PX4 SITL autopilot,
a simulated sensor suite, four `World.add_physics_callback` registrations, and
the hand-rolled driver that exists because Isaac Sim 6.0.1 stops dispatching
those callbacks after ~2 calls.

All of that exists to make an aircraft fly *itself*. At this stage nothing is:
one drone hovers and the other slides along a line, both on trajectories written
down in advance. Skipping it removes the PX4 build, MAVLink ports, `/tmp/px4_lock`
files, arming, an EKF that can refuse to take off, and the callback workaround —
and it makes the trajectory *exactly* what was asked for rather than what a
position controller converged to. A recording made this way repeats to the pixel,
which matters when the recording is the input to a detector you are measuring.

The cost is that the drones have no dynamics: no banking, no rotor wash, motion
as smooth as the trajectory function. The pursuit half did need dynamics and did
not solve it here — `pursuit/dynamics.py` integrates a rate-limited rigid-body
model on the *host* and sends the resulting pose down the socket, so
`pursuit_server.py` still drives the same `KinematicDrone`. The remaining
replacement, if PX4-in-the-loop is ever wanted, is a `PegasusDrone` implementing
the same three methods (`set_pose`, `rgb`, `position`) — the recorder and the
trajectories do not need to know which one they have.

One cosmetic consequence: making every rigid body kinematic means PhysX logs
`cannot create a joint between static bodies` for the four rotor joints at
startup, and the propellers do not spin. Harmless, and the airframe still reads
unmistakably as a quadrotor at 30 m.

## The camera, and the bug worth knowing about

The nose camera's intrinsics are read from the PEGASUS platform's own
`camera_pegasus_iris_720x420.yaml` rather than restated here, so this rig renders
the same camera as the rest of the project's simulated data and a calibration
change cannot be applied in one place and missed in the other. That is the one
file the Prerequisites section above says you must supply, via
`PEGASUS_CONFIG_DIR`; the ring derives its own intrinsics and reads nothing. The
default render is **1440x840, an exact 2x** — a *uniform*
scale, so the field of view is unchanged and the target simply lands on twice as
many pixels. Avoid sizes of a different aspect ratio (1280x720 is 1.778 against
the camera's 1.714): `fx` and `fy` then scale by different factors, which is a
stretched camera, not the same one rendered larger.

**Mount the camera with `set_local_pose(..., camera_axes="world")`, not by
passing a quaternion to the `Camera` constructor.** The constructor's
`orientation` is a raw USD local rotation and applies no conversion; a USD camera
looks down its own −Z with +Y up. Getting this wrong renders a perfectly sharp,
perfectly exposed, upside-down world — which is exactly what the first run
produced.

Note the principal point is well above centre (`cy = 128.6` of 420): the real
camera looks down, that asymmetry is genuine, and two drones at the *same*
altitude therefore meet in the upper third of the frame, not the middle. When a
ground-truth overlay and the rendered target disagree by ~160 px vertically, this
is why.

## How big is the target?

`fx * 0.47 m / range`, at the 1440x840 default:

| range | target span |
|---|---|
| 10 m | 43 px |
| 15 m | 29 px |
| 20 m | 22 px |
| 30 m | 14 px |
| 50 m | 9 px |

`--standoff` is the difficulty knob. The script prints the resulting span at both
ends of the sweep before it records anything.

## No ffmpeg in the container

`nvcr.io/nvidia/isaac-sim:6.0.1` has no ffmpeg, so the in-sim writer is OpenCV's
`mp4v` — large files, and not what you want to share. `annotate_recording.py`
runs on the **host** and re-encodes to H.264 (via `imageio-ffmpeg`, already in
`requirements.txt`) while adding the ground-truth boxes and insets. A 30-second
recording goes from ~72 MB of `mp4v` to ~14 MB of H.264.

## Two processes, one socket

The recording rig is one process that runs to completion. The closed loop cannot
be, because its two halves have irreconcilable dependencies: Isaac Sim lives in a
container with no ultralytics, no scipy and none of this project's weights, and
the perception stack lives on the host and cannot import Isaac. They share one
thing:
`/tmp/dev` in the container is a directory on the host (`~/isaac_dev_root` by
default; `DRONEDET_SIM_ROOT` moves it), so a Unix socket in there is visible to
both, and `pursuit_proto.py` is **the only module both sides run** — which is
why it imports nothing outside the standard library and why it must stay that
way. Booting Isaac with a scene loaded costs 30–60 seconds and the brain is the
half that changes every few minutes, so the simulator is a long-lived server: one
boot serves an entire scenario matrix. Frames cross **raw**, not JPEG — the
target is 10–30 px across and keys on exactly the small local contrast a
quantiser discards first, so 3.6 MB a frame over a local socket is cheaper than
arguing about whether a miss was the algorithm or the codec.

Full protocol and message format in the `pursuit_proto.py` docstring; the loop
that drives it is in [`pursuit/README.md`](../../pursuit/README.md).

## Running the detector over a recording

This has since been done, and the answer was no: scored against this rig's own
ground truth on `run_two_drone.py` observer footage, EDGE-RT reaches recall 0.50
with 11 283 false positives, round-7 fusion 0.07, and the PC-MAX appearance
expert 0.00 — the table and the diagnosis are in
[`pursuit/README.md`](../../pursuit/README.md), under "The detector", which also
records that this was a one-off pass whose raw output was not archived, so the
numbers are a record of what was seen rather than something re-derivable from
`work/`. The reason is the thing worth knowing here:
**every strong detector in this repo is temporal.** The round-7 fusion model
takes 4 channels (B, G, R, ego-motion) where the motion channel is built from
frames t−3 and t−6; the PC-MAX / Edge models take three *stabilized greyscales*
at t−12/t−6/t, not colour. None of them can be fed one isolated frame — a cold
single-frame call returns nothing on a frame the warm detector is 0.83 confident
about.

That is why the recorder writes `left/*.jpg` as well as an MP4: the detector must
be driven over consecutive frames with monotonically increasing indices, and
decoding those back out of a compressed video adds an artefact to the exact
signal (small inter-frame differences) those models key on.

## Known limits

- The observer tracks the target perfectly, because it is told where the target
  is. That is right for stage one — it isolates *detection* from *pointing* —
  but it means the target sits near the principal point on every frame, which is
  an easier detection problem than a real search. `Orbit` and a fixed observer
  heading are the way to make it honest.
- `frames.json` records the target's projected centre and pixel span, not a
  segmented bounding box. Good enough to score a detection's centre against;
  not an IoU ground truth.
- Only `rivermark` has measured `SCENE_GROUND_Z` / `SCENE_ORIGIN_XY`. The other
  two are flat ground at the origin, which is true for them but not measured.
- Both drones are the same airframe. A detector trained on this alone would
  learn one silhouette.
