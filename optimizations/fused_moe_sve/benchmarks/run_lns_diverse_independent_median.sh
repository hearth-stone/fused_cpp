#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhangxu/codex/fused_cpp}"
OUT="${OUT:-$ROOT/tmp/moe_lns_diverse_independent_median_20260904}"
PY="${PY:-$ROOT/.venv/bin/python}"
CALIBRATION="bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json"
ROUTE="bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request016_case017_zh2048-018.pt"
PAIRWISE="cpu_moe_schedule_optimization/planners/profiles/arm_codex_80c_pairwise_ordering_gated_v8_20260904.json"
EXPECTED_CAL="7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3"
EXPECTED_ROUTE="afabc7a1c9ffbabf4a844cf6842b1001d43a13df6fde0e37a5ee1eae58649431"
EXPECTED_PAIRWISE="a1b89cec87cc6c9cacab4477965bd060b8f41ba7eac227ef055552609268e54a"
EXPECTED_EXT="dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f"
EXPECTED_POLICY="6c07e7b9fbf78412a103e3265d9dabbed224c1b9e3e248334e2ab1fb6ab08c97"

cd "$ROOT"
mkdir -p "$OUT"
export PYTHONPATH=.:src
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=FALSE
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

sha() {
  sha256sum "$1" | awk '{print $1}'
}

fail() {
  echo "FAILED: $*" | tee "$OUT/FAILED"
  exit 1
}

echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "pwd=$(pwd)"
echo "page_policy_env=$(env | grep -E 'HUGE|THP|PAGE|FUSED_CPP_MOE' | sort | tr '\n' ' ')"

"$PY" - <<'PY' || fail "unit tests"
import hashlib
import subprocess
import sys
from pathlib import Path

from fused_cpp import _moe_C

ext = hashlib.sha256(Path(_moe_C.__file__).read_bytes()).hexdigest()
print("extension", Path(_moe_C.__file__), ext)
if ext != "dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f":
    raise SystemExit(f"unexpected extension sha256: {ext}")
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_moe_lns_diverse_shortlist.py",
        "tests/test_moe_partial_order_vnd_runner.py",
        "tests/test_moe_partial_order_hardware_frontier.py",
    ],
    check=False,
)
if result.returncode != 0:
    raise SystemExit(result.returncode)
PY

[[ "$(sha "$CALIBRATION")" == "$EXPECTED_CAL" ]] || fail "calibration hash mismatch"
[[ "$(sha "$ROUTE")" == "$EXPECTED_ROUTE" ]] || fail "route hash mismatch"
[[ "$(sha "$PAIRWISE")" == "$EXPECTED_PAIRWISE" ]] || fail "pairwise hash mismatch"

echo "STEP model"
numactl --physcpubind=240-319 --membind=3 "$PY" \
  optimizations/fused_moe_sve/benchmarks/bench_template_lns_layer.py \
  --initial-controls \
  --route-file "$ROUTE" \
  --route-layer 4 \
  --analytic-calibration "$CALIBRATION" \
  --pairwise-calibration "$PAIRWISE" \
  --critical-experts 32 \
  --neighbors-per-operator 50 \
  --shortlist-budget 16 \
  --audit-budget 32 \
  --lns-destroy-sizes 4,8,16 \
  --lns-repair-beam-widths 16,32,64 \
  --lns-templates-per-block 4 \
  --restarts-per-parent 1 \
  --seed 20261010 \
  --output "$OUT/median_lns_diverse_model.json" \
  || fail "model generation"

MODEL_SHA="$(sha "$OUT/median_lns_diverse_model.json")"
echo "model_sha256=$MODEL_SHA"
"$PY" - <<PY || fail "model policy hash"
import json
from pathlib import Path
payload = json.loads(Path("$OUT/median_lns_diverse_model.json").read_text())
policy = payload["method"]["lns_shortlist_policy_sha256"]
print("policy", policy)
if policy != "$EXPECTED_POLICY":
    raise SystemExit(f"unexpected policy sha256: {policy}")
if payload["method"]["partial_order_decision_mode"] != "diagnostic_only":
    raise SystemExit("partial-order decision mode is not diagnostic_only")
if int(payload["method"]["restarts_per_parent"]) != 1:
    raise SystemExit("restarts_per_parent is not 1")
print("starts", payload["summary"])
PY

echo "STEP frontier"
"$PY" optimizations/fused_moe_sve/benchmarks/build_lns_diverse_hardware_frontier.py \
  --lns-artifact "$OUT/median_lns_diverse_model.json" \
  --output "$OUT/median_lns_diverse_frontier.json" \
  || fail "frontier build"

echo "STEP session1"
numactl --physcpubind=240-319 --membind=3 "$PY" \
  optimizations/fused_moe_sve/benchmarks/bench_partial_order_hardware_frontier.py \
  --frontier "$OUT/median_lns_diverse_frontier.json" \
  --route-file "$ROUTE" \
  --route-layer 4 \
  --warmup 5 \
  --runs 31 \
  --weight-copies 4 \
  --seed 20261011 \
  --output "$OUT/median_lns_diverse_session1.json" \
  || fail "session 1"

echo "STEP session2"
numactl --physcpubind=240-319 --membind=3 "$PY" \
  optimizations/fused_moe_sve/benchmarks/bench_partial_order_hardware_frontier.py \
  --frontier "$OUT/median_lns_diverse_frontier.json" \
  --route-file "$ROUTE" \
  --route-layer 4 \
  --warmup 5 \
  --runs 31 \
  --weight-copies 4 \
  --seed 20261012 \
  --output "$OUT/median_lns_diverse_session2.json" \
  || fail "session 2"

echo "STEP analyze"
"$PY" optimizations/fused_moe_sve/benchmarks/analyze_lns_diverse_hardware_frontier.py \
  --frontier "$OUT/median_lns_diverse_frontier.json" \
  --model-artifact "$OUT/median_lns_diverse_model.json" \
  --session "$OUT/median_lns_diverse_session1.json" "$OUT/median_lns_diverse_session2.json" \
  --output "$OUT/median_lns_diverse_analysis.json" \
  || fail "analysis"

echo "DONE"
{
  echo "DONE"
  echo "model=$(sha "$OUT/median_lns_diverse_model.json")"
  echo "frontier=$(sha "$OUT/median_lns_diverse_frontier.json")"
  echo "session1=$(sha "$OUT/median_lns_diverse_session1.json")"
  echo "session2=$(sha "$OUT/median_lns_diverse_session2.json")"
  echo "analysis=$(sha "$OUT/median_lns_diverse_analysis.json")"
} | tee "$OUT/DONE"
