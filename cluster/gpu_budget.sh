#!/bin/bash
# How many GPUs am I holding right now, and how many may I still ask for?
#
# The account limit is 7 GPUs concurrently across ALL jobs, including jobs launched from
# other sessions that this one cannot see. So the number must be measured, never assumed,
# and measured immediately before each submission.
#
#   ./cluster/gpu_budget.sh          # print the budget
#   ./cluster/gpu_budget.sh 3        # also exit non-zero unless 3 more GPUs are free
#
# Counting method matters. `squeue -o %b` prints TRES_PER_NODE, which reports N/A for
# array tasks and silently undercounts; `squeue -O AllocTRES` is rejected outright by this
# SLURM build. `scontrol show job` reports gres/gpu= per job and is the only form that
# agreed with reality when three array tasks were running on one node.

set -uo pipefail
LIMIT=${SPECKLOCK_GPU_LIMIT:-7}
WANT=${1:-0}

total=0
while read -r j; do
    [ -n "$j" ] || continue
    n=$(scontrol show job "$j" 2>/dev/null | grep -oE 'gres/gpu=[0-9]+' | head -1 |
        grep -oE '[0-9]+$')
    total=$(( total + ${n:-0} ))
done < <(squeue -u "$USER" -h -t R,PD,CG -o '%A')

free=$(( LIMIT - total ))
[ "$free" -lt 0 ] && free=0
echo "GPUs held (running+pending): $total / $LIMIT     still available: $free"

if [ "$WANT" -gt 0 ]; then
    if [ "$WANT" -gt "$free" ]; then
        echo "REFUSE: asked for $WANT, only $free within the limit."
        echo "Queueing and waiting is fine; exceeding the limit is not."
        exit 1
    fi
    echo "OK to request $WANT."
fi
