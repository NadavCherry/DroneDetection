#!/usr/bin/env bash
# Copy working-tree files to the cluster checkout, safely.
#
#   ./cluster/push.sh dronedet/metrics.py tools/size_curve.py
#   ./cluster/push.sh --all-untracked
#
# Three things this exists to prevent, all of which happened:
#
# 1. CRLF. This repo is developed on Windows with core.autocrlf=true, so the working tree
#    holds CRLF while the repository stores LF. A plain scp carries the CRLF to Linux and
#    every line of the file reads as modified -- a 900-line file showed 922 changed lines
#    for a 6-line edit, which buried a real problem underneath it. Text files are
#    normalised to LF on arrival.
#
# 2. Staging through /tmp. This cluster gives each SSH session a PRIVATE /tmp, so a file
#    scp'd to /tmp in one connection does not exist to the next one. Two separate rounds
#    of "cannot stat" came from that. Files go straight to their destination path.
#
# 3. Copying across a divergence. The cluster checkout may be on a different commit than
#    this working tree. Overwriting dronedet/metrics.py from an older lineage silently
#    deleted average_precision_11pt, a function the summary generator depends on. This
#    refuses to run unless the local HEAD is an ancestor of, or equal to, the cluster's.

set -uo pipefail
HOST=${SPECKLOCK_HOST:-cherryn@slurm.bgu.ac.il}
REMOTE=${SPECKLOCK_REMOTE:-projects/SpeckLock}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=40"

cd "$(git rev-parse --show-toplevel)" || exit 1
local_head=$(git rev-parse HEAD)
remote_head=$($SSH "$HOST" "cd $REMOTE && git rev-parse HEAD" 2>/dev/null)

if [ -z "$remote_head" ]; then
    echo "cannot read the cluster's HEAD -- is $HOST reachable?" >&2
    exit 1
fi
if [ "$local_head" != "$remote_head" ]; then
    if git merge-base --is-ancestor "$remote_head" "$local_head" 2>/dev/null; then
        echo "note: local ($(git rev-parse --short HEAD)) is AHEAD of cluster " \
             "(${remote_head:0:7}); pushing files from the newer lineage."
    else
        echo "REFUSE: local HEAD $(git rev-parse --short HEAD) is not a descendant of" >&2
        echo "        the cluster's ${remote_head:0:7}. Copying would overwrite work" >&2
        echo "        that exists only on the cluster. Fast-forward first." >&2
        exit 1
    fi
fi

files=("$@")
if [ "${1:-}" = "--all-untracked" ]; then
    mapfile -t files < <(git ls-files --others --exclude-standard)
fi
[ "${#files[@]}" -gt 0 ] || { echo "nothing to push"; exit 0; }

for f in "${files[@]}"; do
    [ -f "$f" ] || { echo "  skip (missing): $f"; continue; }
    $SSH "$HOST" "mkdir -p $REMOTE/$(dirname "$f")" || exit 1
    scp -q -o BatchMode=yes -o ConnectTimeout=40 "$f" "$HOST:$REMOTE/$f" || exit 1
    case "$f" in
        *.py|*.md|*.sh|*.sbatch|*.txt|*.yaml|*.yml|*.json|*.cfg|*.toml)
            $SSH "$HOST" "sed -i 's/\r\$//' $REMOTE/$f" ;;
    esac
    echo "  pushed $f"
done

echo
$SSH "$HOST" "cd $REMOTE && git diff --stat | tail -5"
