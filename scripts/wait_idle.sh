#!/usr/bin/env bash
# Idle guard for measurement chains. Source it, then call `wait_idle` before a
# measured session.
#
# The 1-minute load average alone does not separate a session from the work that
# ran immediately before it: on C9g a session admitted at 1-minute load 0.39 with
# the 5-minute average still at 3.10 measured 0.54% off a clean baseline whose own
# session-to-session spread is 0.1%, and the work it had not been separated from
# was our own grid run, finished 27 s earlier
# (`optimizations/fused_moe_sve/results/numa_interference_c9g_20260922.md`).
# So both averages must settle, and the first measured session of a chain also
# waits out a fixed cooldown.
#
#   source scripts/wait_idle.sh
#   wait_idle              # defaults below
#   wait_idle 2 4 60 1800  # one-minute max, five-minute max, cooldown s, timeout s
#
# Exits nonzero when the machine does not settle within the timeout, so a chain
# can record the failure instead of measuring through it.

wait_idle() {
    local max_one="${1:-2}" max_five="${2:-4}" cooldown="${3:-60}" timeout="${4:-1800}"
    local deadline=$(( $(date +%s) + timeout ))
    local settled_since=0 now one five

    while :; do
        now=$(date +%s)
        read -r one five _ < /proc/loadavg
        # Integer comparison: the thresholds are coarse and bash has no floats.
        if [ "$(printf '%.0f' "${one}")" -le "${max_one}" ] &&
           [ "$(printf '%.0f' "${five}")" -le "${max_five}" ]; then
            [ "${settled_since}" -eq 0 ] && settled_since="${now}"
            if [ $(( now - settled_since )) -ge "${cooldown}" ]; then
                echo "[$(date -Is)] idle: load ${one} ${five}, settled ${cooldown}s"
                return 0
            fi
        else
            settled_since=0
        fi
        if [ "${now}" -ge "${deadline}" ]; then
            echo "[$(date -Is)] idle guard timed out: load $(cat /proc/loadavg)" >&2
            return 1
        fi
        sleep 15
    done
}
