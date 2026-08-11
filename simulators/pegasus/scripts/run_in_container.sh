#!/usr/bin/env bash
# Run a two-drone air-to-air recording, from the HOST.
#
# Syncs this repo and the nose camera's calibration into the running isaac-sim
# container, runs the sim under Isaac Sim's own Python, and copies the recording
# back out. Everything the sim needs lives in the container; everything you look
# at afterwards lands here.
#
# Two prerequisites are NOT part of this repository and cannot be shipped with
# it: the Isaac Sim 6.0.1 container itself, and the camera calibration, which
# belongs to the external PEGASUS platform. Point PEGASUS_CONFIG_DIR at a copy
# of that config directory. Everything that needs neither -- the ring geometry,
# and the whole fast pursuit loop (.venv/bin/python -m pursuit.sandbox --suite
# city --ring) -- is described in ../README.md under "Prerequisites".
#
# Usage:
#   PEGASUS_CONFIG_DIR=/path/to/robots/PEGASUS/config \
#       run_in_container.sh [--scene rivermark] [--seconds 30] [--standoff 15] ...
#   run_in_container.sh --scout            # just render overview stills
#
# Any flag not listed below is passed straight through to run_two_drone.py.
set -euo pipefail

CONTAINER="${ISAAC_CONTAINER:-isaac-sim}"
DEV_ROOT="${ISAAC_DEV_ROOT:-/tmp/dev}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# No default: the calibration lives outside this repo, and a path baked in here
# would be one developer's machine and a silent failure on everyone else's.
if [[ -z "${PEGASUS_CONFIG_DIR:-}" ]]; then
    echo "ERROR: PEGASUS_CONFIG_DIR is not set." >&2
    echo "  It must point at the PEGASUS platform's robots/PEGASUS/config" >&2
    echo "  directory, which holds camera_pegasus_iris_720x420.yaml. That" >&2
    echo "  platform is external to this repository -- see ../README.md," >&2
    echo "  'Prerequisites', for what an outside reader can run instead." >&2
    exit 2
fi
if [[ ! -d "$PEGASUS_CONFIG_DIR" ]]; then
    echo "ERROR: PEGASUS_CONFIG_DIR is not a directory: $PEGASUS_CONFIG_DIR" >&2
    exit 2
fi

RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_LOCAL="$REPO_ROOT/simulators/pegasus/recordings/$RUN_NAME"
OUT_REMOTE="$DEV_ROOT/air2air/$RUN_NAME"
CFG_REMOTE="$DEV_ROOT/platform/robots/PEGASUS/config"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "ERROR: container '$CONTAINER' is not running. Start it with:" >&2
    echo "  docker start $CONTAINER" >&2
    exit 1
fi

# A previous run still holding the GPU is the most common cause of an
# unexplained crash in the next one -- Kit's start-up is the heaviest moment of
# its life and two overlapping ones contend hard enough to kill the RTX shader
# compiler.
if docker exec "$CONTAINER" pgrep -f "run_two_drone.py" > /dev/null 2>&1; then
    echo "ERROR: a run is already in progress. Stop it with:" >&2
    echo "  docker exec $CONTAINER pkill -9 -f run_two_drone.py" >&2
    exit 1
fi

echo "Syncing into $CONTAINER ..."
docker exec "$CONTAINER" mkdir -p "$DEV_ROOT/dronedet" "$CFG_REMOTE" "$OUT_REMOTE"
docker cp "$REPO_ROOT/simulators" "$CONTAINER:$DEV_ROOT/dronedet/" > /dev/null
# Trailing '/.' copies the *contents*, so the config lands at a fixed path
# whatever the source directory happens to be called.
docker cp "$PEGASUS_CONFIG_DIR/." "$CONTAINER:$CFG_REMOTE" > /dev/null

LOG="$OUT_REMOTE.log"
echo "Running (log: docker exec $CONTAINER tail -f $LOG)"
set +e
# --pegasus-config goes first so a caller passing their own still wins.
docker exec "$CONTAINER" bash -c \
    "cd $DEV_ROOT/dronedet && /isaac-sim/python.sh \
     simulators/pegasus/scripts/run_two_drone.py \
     --pegasus-config '$CFG_REMOTE' \
     --out-dir '$OUT_REMOTE' $(printf '%q ' "$@") > '$LOG' 2>&1"
status=$?
set -e

docker exec "$CONTAINER" grep -aE "^\[ +[0-9.]+s\]|^DONE|Traceback" "$LOG" | tail -20 || true

if [[ $status -ne 0 ]]; then
    echo "Run FAILED (exit $status). Full log:" >&2
    echo "  docker exec $CONTAINER cat $LOG" >&2
    exit "$status"
fi

echo "Copying the recording out to $OUT_LOCAL ..."
mkdir -p "$(dirname "$OUT_LOCAL")"
docker cp "$CONTAINER:$OUT_REMOTE" "$OUT_LOCAL" > /dev/null

echo
echo "Recording: $OUT_LOCAL"
echo "Make it readable (ground-truth boxes + magnified insets, H.264):"
echo "  python simulators/pegasus/scripts/annotate_recording.py '$OUT_LOCAL'"
