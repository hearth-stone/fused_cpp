#!/usr/bin/env bash
# run.sh — build + test + bench, one shot.
set -e
cd "$(dirname "$0")"
./build.sh
echo
./microkernel_min test
echo
./microkernel_min bench --iters=500000 --warmup=20000
