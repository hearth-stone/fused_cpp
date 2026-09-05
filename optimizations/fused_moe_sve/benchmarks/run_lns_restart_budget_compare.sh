#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhangxu/codex/fused_cpp}"
OUT="${OUT:-$ROOT/tmp/moe_lns_restart_budget_20260905}"
PY="${PY:-$ROOT/.venv/bin/python}"
CALIBRATION="bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json"
ROUTE="bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request016_case017_zh2048-018.pt"
PAIRWISE="cpu_moe_schedule_optimization/planners/profiles/arm_codex_80c_pairwise_ordering_gated_v8_20260904.json"
ELITE_SRC="${ELITE_SRC:-$ROOT/tmp/moe_template_lns_20260904/median_lns_frontier.json}"
ELITE_HASH="2ab43572d14e11060e4fc07b3aa7e0c1780553233dfda61abe74f08abc51e973"
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

[[ -f "$ELITE_SRC" ]] || fail "missing elite source $ELITE_SRC"
[[ "$(sha "$CALIBRATION")" == "$EXPECTED_CAL" ]] || fail "calibration hash mismatch"
[[ "$(sha "$ROUTE")" == "$EXPECTED_ROUTE" ]] || fail "route hash mismatch"
[[ "$(sha "$PAIRWISE")" == "$EXPECTED_PAIRWISE" ]] || fail "pairwise hash mismatch"

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
        "tests/test_moe_executable_plan_neighborhood.py",
        "tests/test_moe_lns_diverse_shortlist.py",
        "tests/test_moe_partial_order_vnd_runner.py",
        "tests/test_moe_partial_order_hardware_frontier.py",
    ],
    check=False,
)
if result.returncode != 0:
    raise SystemExit(result.returncode)
PY

echo "STEP extract elite"
"$PY" - <<PY || fail "elite extract"
import json
from pathlib import Path
src = Path("$ELITE_SRC")
out = Path("$OUT/known_median_elite.json")
key = "$ELITE_HASH"
frontier = json.loads(src.read_text())
plan = frontier["plans"][key]
out.write_text(
    json.dumps(
        {
            "state_hash": key,
            "canonical_state": plan["canonical_state"],
            "plan_v2_bridge": plan["plan_v2_bridge"],
            "source_frontier": str(src),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print("elite", key)
PY

run_model() {
  local name="$1"
  local restarts="$2"
  local neighbors="$3"
  echo "STEP model $name restarts=$restarts neighbors=$neighbors"
  numactl --physcpubind=240-319 --membind=3 "$PY" \
    optimizations/fused_moe_sve/benchmarks/bench_template_lns_layer.py \
    --initial-controls \
    --route-file "$ROUTE" \
    --route-layer 4 \
    --analytic-calibration "$CALIBRATION" \
    --pairwise-calibration "$PAIRWISE" \
    --critical-experts 32 \
    --neighbors-per-operator "$neighbors" \
    --shortlist-budget 16 \
    --audit-budget 32 \
    --lns-destroy-sizes 4,8,16 \
    --lns-repair-beam-widths 16,32,64 \
    --lns-templates-per-block 4 \
    --restarts-per-parent "$restarts" \
    --pool-restarts-by-parent \
    --stratified-outside-audit 16 \
    --stratified-seed 20261013 \
    --seed 20261010 \
    --output "$OUT/${name}_model.json" \
    || fail "model $name"
}

build_and_measure() {
  local name="$1"
  local session1_seed="$2"
  local session2_seed="$3"
  echo "STEP frontier $name"
  "$PY" optimizations/fused_moe_sve/benchmarks/build_lns_diverse_hardware_frontier.py \
    --lns-artifact "$OUT/${name}_model.json" \
    --known-elite-plan "$OUT/known_median_elite.json" \
    --no-include-audit \
    --include-stratified \
    --output "$OUT/${name}_frontier.json" \
    || fail "frontier $name"
  echo "STEP session1 $name"
  numactl --physcpubind=240-319 --membind=3 "$PY" \
    optimizations/fused_moe_sve/benchmarks/bench_partial_order_hardware_frontier.py \
    --frontier "$OUT/${name}_frontier.json" \
    --route-file "$ROUTE" \
    --route-layer 4 \
    --warmup 5 \
    --runs 31 \
    --weight-copies 4 \
    --seed "$session1_seed" \
    --output "$OUT/${name}_session1.json" \
    || fail "session1 $name"
  echo "STEP session2 $name"
  numactl --physcpubind=240-319 --membind=3 "$PY" \
    optimizations/fused_moe_sve/benchmarks/bench_partial_order_hardware_frontier.py \
    --frontier "$OUT/${name}_frontier.json" \
    --route-file "$ROUTE" \
    --route-layer 4 \
    --warmup 5 \
    --runs 31 \
    --weight-copies 4 \
    --seed "$session2_seed" \
    --output "$OUT/${name}_session2.json" \
    || fail "session2 $name"
}

run_model one_restart 1 50
build_and_measure one_restart 20261021 20261022
run_model two_restart 2 25
build_and_measure two_restart 20261023 20261024

echo "STEP compare"
"$PY" optimizations/fused_moe_sve/benchmarks/analyze_lns_restart_budget.py \
  --one-restart-frontier "$OUT/one_restart_frontier.json" \
  --one-restart-model "$OUT/one_restart_model.json" \
  --one-restart-session "$OUT/one_restart_session1.json" "$OUT/one_restart_session2.json" \
  --two-restart-frontier "$OUT/two_restart_frontier.json" \
  --two-restart-model "$OUT/two_restart_model.json" \
  --two-restart-session "$OUT/two_restart_session1.json" "$OUT/two_restart_session2.json" \
  --output "$OUT/restart_budget_analysis.json" \
  || fail "compare"

echo "DONE"
{
  echo "DONE"
  echo "elite=$(sha "$OUT/known_median_elite.json")"
  echo "one_model=$(sha "$OUT/one_restart_model.json")"
  echo "one_frontier=$(sha "$OUT/one_restart_frontier.json")"
  echo "two_model=$(sha "$OUT/two_restart_model.json")"
  echo "two_frontier=$(sha "$OUT/two_restart_frontier.json")"
  echo "analysis=$(sha "$OUT/restart_budget_analysis.json")"
} | tee "$OUT/DONE"
