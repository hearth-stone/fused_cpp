#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: $0 MODE [TAG] [K] [N]}"
tag="${2:-run}"
k="${3:-4096}"
n="${4:-1024}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "${script_dir}/../../.." && pwd)"
python="${PYTHON:-${repo}/.venv/bin/python}"
cpu="${CPU:-48}"
numa_node="${NUMA_NODE:-0}"
experts="${EXPERTS:-64}"
warmup="${WARMUP:-256}"
runs="${RUNS:-1024}"
attach_delay="${ATTACH_DELAY_SECONDS:-0.2}"
events="${EVENTS:-cycles:u,instructions:u,r70:u,l1d_cache_refill:u,l2d_cache_refill:u,stall_backend_mem:u}"
output_dir="${OUTPUT_DIR:-/tmp}"
output="${output_dir}/gemm_probe_k${k}_n${n}_${mode}_${tag}.out"
perf_output="${output_dir}/gemm_probe_k${k}_n${n}_${mode}_${tag}.perf"
benchmark_pid=""
perf_pid=""

mkdir -p "${output_dir}"

cleanup() {
  if [[ -n "${perf_pid}" ]] && kill -0 "${perf_pid}" 2>/dev/null; then
    kill -INT "${perf_pid}" 2>/dev/null || true
  fi
  if [[ -n "${benchmark_pid}" ]] && kill -0 "${benchmark_pid}" 2>/dev/null; then
    kill -KILL "${benchmark_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "${repo}"
OMP_NUM_THREADS=1 \
PYTHONPATH=src \
FUSED_CPP_MOE_SVE_JIT_EXACT_M=1 \
numactl --physcpubind="${cpu}" --membind="${numa_node}" \
"${python}" \
  optimizations/fused_moe_sve/benchmarks/bench_pure_w13_gemm_m1_m2.py \
  --k "${k}" \
  --n "${n}" \
  --rows 1 \
  --n-ranges 1 \
  --experts "${experts}" \
  --warmup "${warmup}" \
  --runs "${runs}" \
  --probe-mode "${mode}" \
  --profile-window >"${output}" 2>&1 &
benchmark_pid=$!

while true; do
  if [[ ! -r "/proc/${benchmark_pid}/status" ]]; then
    cat "${output}"
    exit 1
  fi
  state="$(awk '/^State:/ { print $2 }' "/proc/${benchmark_pid}/status")"
  if [[ "${state}" == "T" ]]; then
    break
  fi
  sleep 0.01
done

perf stat \
  -x, \
  -e "${events}" \
  -p "${benchmark_pid}" \
  -o "${perf_output}" &
perf_pid=$!
sleep "${attach_delay}"
initial_context_switches="$(
  awk '/^(voluntary|nonvoluntary)_ctxt_switches:/ { total += $2 } END { print total + 0 }' \
    "/proc/${benchmark_pid}/status"
)"
kill -CONT "${benchmark_pid}"

while true; do
  if [[ ! -r "/proc/${benchmark_pid}/status" ]]; then
    cat "${output}"
    exit 1
  fi
  read -r state context_switches < <(
    awk '
      /^State:/ { state = $2 }
      /^(voluntary|nonvoluntary)_ctxt_switches:/ { total += $2 }
      END { print state, total + 0 }
    ' "/proc/${benchmark_pid}/status"
  )
  if [[ "${state}" == "T" ]] && ((context_switches > initial_context_switches)); then
    break
  fi
  sleep 0.001
done

kill -INT "${perf_pid}"
wait "${perf_pid}" || true
perf_pid=""
kill -CONT "${benchmark_pid}"
wait "${benchmark_pid}"
benchmark_pid=""
trap - EXIT

printf 'TAG=%s MODE=%s K=%s N=%s\n' "${tag}" "${mode}" "${k}" "${n}"
grep -E '^[[:space:]]+1 1' "${output}"
cat "${perf_output}"
