#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhangxu/codex/fused_cpp}"
OUT="${OUT:-$ROOT/tmp/moe_lns_second_seed_20260905}"
PREV="${PREV:-$ROOT/tmp/moe_lns_restart_budget_20260905}"
PY="${PY:-$ROOT/.venv/bin/python}"
CALIBRATION="bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json"
ROUTE="bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request016_case017_zh2048-018.pt"
PAIRWISE="cpu_moe_schedule_optimization/planners/profiles/arm_codex_80c_pairwise_ordering_gated_v8_20260904.json"
ELITE_SRC="${ELITE_SRC:-$ROOT/tmp/moe_template_lns_20260904/median_lns_frontier.json}"
ELITE_HASH="2ab43572d14e11060e4fc07b3aa7e0c1780553233dfda61abe74f08abc51e973"
PREVIOUS_SELECTED="0418b88445c2f88488ca10a1eeae3aeb7b6080e806e241eacfed17ec10904bf6"
PROPOSAL_SEED=20261011
STRATIFIED_SEED=20261013
SESSION1_SEED=20261025
SESSION2_SEED=20261026
EXPECTED_CAL="7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3"
EXPECTED_ROUTE="afabc7a1c9ffbabf4a844cf6842b1001d43a13df6fde0e37a5ee1eae58649431"
EXPECTED_PAIRWISE="a1b89cec87cc6c9cacab4477965bd060b8f41ba7eac227ef055552609268e54a"
EXPECTED_EXT="dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f"
EXPECTED_POLICY="6c07e7b9fbf78412a103e3265d9dabbed224c1b9e3e248334e2ab1fb6ab08c97"
EXPECTED_PREV_MODEL="9b334b784e80c8376aa059084875a75d4f323c2a4c8f4192e60288f4590595b8"
EXPECTED_PREV_FRONTIER="18914c52d5c628ac44086db2653fc93628864193adea251f3a0078029cb424a7"
EXPECTED_PREV_ANALYSIS="027ea3302286290a874f910d2498862037e8530a2baf8b4f911f0068e598a16f"

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
[[ -f "$PREV/two_restart_model.json" ]] || fail "missing previous 2-restart model"
[[ -f "$PREV/two_restart_frontier.json" ]] || fail "missing previous 2-restart frontier"
[[ -f "$PREV/restart_budget_analysis.json" ]] || fail "missing previous restart analysis"
[[ "$(sha "$CALIBRATION")" == "$EXPECTED_CAL" ]] || fail "calibration hash mismatch"
[[ "$(sha "$ROUTE")" == "$EXPECTED_ROUTE" ]] || fail "route hash mismatch"
[[ "$(sha "$PAIRWISE")" == "$EXPECTED_PAIRWISE" ]] || fail "pairwise hash mismatch"
[[ "$(sha "$PREV/two_restart_model.json")" == "$EXPECTED_PREV_MODEL" ]] || fail "previous model hash mismatch"
[[ "$(sha "$PREV/two_restart_frontier.json")" == "$EXPECTED_PREV_FRONTIER" ]] || fail "previous frontier hash mismatch"
[[ "$(sha "$PREV/restart_budget_analysis.json")" == "$EXPECTED_PREV_ANALYSIS" ]] || fail "previous analysis hash mismatch"

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

echo "STEP extract previous selected"
"$PY" - <<PY || fail "previous selected extract"
import json
from pathlib import Path

analysis = json.loads(Path("$PREV/restart_budget_analysis.json").read_text())
frontier = json.loads(Path("$PREV/two_restart_frontier.json").read_text())
first = analysis["two_restart"]["session_1"]["selected_best_state_hash"]
second = analysis["two_restart"]["session_2"]["selected_best_state_hash"]
if first != second:
    raise SystemExit(f"previous selected-best differs across sessions: {first} vs {second}")
if first != "$PREVIOUS_SELECTED":
    raise SystemExit(f"unexpected previous selected-best: {first}")
plan = frontier["plans"][first]
out = Path("$OUT/previous_seed_selected.json")
out.write_text(
    json.dumps(
        {
            "state_hash": first,
            "canonical_state": plan["canonical_state"],
            "plan_v2_bridge": plan["plan_v2_bridge"],
            "source_frontier": "$PREV/two_restart_frontier.json",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print("previous_selected", first)
PY

echo "STEP model second_seed restarts=2 neighbors=25 seed=$PROPOSAL_SEED"
numactl --physcpubind=240-319 --membind=3 "$PY" \
  optimizations/fused_moe_sve/benchmarks/bench_template_lns_layer.py \
  --initial-controls \
  --route-file "$ROUTE" \
  --route-layer 4 \
  --analytic-calibration "$CALIBRATION" \
  --pairwise-calibration "$PAIRWISE" \
  --critical-experts 32 \
  --neighbors-per-operator 25 \
  --shortlist-budget 16 \
  --audit-budget 32 \
  --lns-destroy-sizes 4,8,16 \
  --lns-repair-beam-widths 16,32,64 \
  --lns-templates-per-block 4 \
  --restarts-per-parent 2 \
  --pool-restarts-by-parent \
  --stratified-outside-audit 16 \
  --stratified-seed "$STRATIFIED_SEED" \
  --seed "$PROPOSAL_SEED" \
  --output "$OUT/second_seed_model.json" \
  || fail "model second_seed"

echo "STEP frontier second_seed"
"$PY" optimizations/fused_moe_sve/benchmarks/build_lns_diverse_hardware_frontier.py \
  --lns-artifact "$OUT/second_seed_model.json" \
  --known-elite-plan "$OUT/known_median_elite.json" \
  --reference-plan "$OUT/previous_seed_selected.json" \
  --no-include-audit \
  --include-stratified \
  --output "$OUT/second_seed_frontier.json" \
  || fail "frontier second_seed"

echo "STEP session1 second_seed"
numactl --physcpubind=240-319 --membind=3 "$PY" \
  optimizations/fused_moe_sve/benchmarks/bench_partial_order_hardware_frontier.py \
  --frontier "$OUT/second_seed_frontier.json" \
  --route-file "$ROUTE" \
  --route-layer 4 \
  --warmup 5 \
  --runs 31 \
  --weight-copies 4 \
  --seed "$SESSION1_SEED" \
  --output "$OUT/second_seed_session1.json" \
  || fail "session1 second_seed"

echo "STEP session2 second_seed"
numactl --physcpubind=240-319 --membind=3 "$PY" \
  optimizations/fused_moe_sve/benchmarks/bench_partial_order_hardware_frontier.py \
  --frontier "$OUT/second_seed_frontier.json" \
  --route-file "$ROUTE" \
  --route-layer 4 \
  --warmup 5 \
  --runs 31 \
  --weight-copies 4 \
  --seed "$SESSION2_SEED" \
  --output "$OUT/second_seed_session2.json" \
  || fail "session2 second_seed"

echo "STEP compare"
"$PY" optimizations/fused_moe_sve/benchmarks/analyze_lns_second_proposal_seed.py \
  --frontier "$OUT/second_seed_frontier.json" \
  --model "$OUT/second_seed_model.json" \
  --session "$OUT/second_seed_session1.json" "$OUT/second_seed_session2.json" \
  --previous-frontier "$PREV/two_restart_frontier.json" \
  --previous-model "$PREV/two_restart_model.json" \
  --output "$OUT/second_seed_analysis.json" \
  || fail "compare"

echo "DONE"
{
  echo "DONE"
  echo "elite=$(sha "$OUT/known_median_elite.json")"
  echo "reference=$(sha "$OUT/previous_seed_selected.json")"
  echo "model=$(sha "$OUT/second_seed_model.json")"
  echo "frontier=$(sha "$OUT/second_seed_frontier.json")"
  echo "analysis=$(sha "$OUT/second_seed_analysis.json")"
} | tee "$OUT/DONE"
