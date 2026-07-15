#!/usr/bin/env bash

set -euo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUFF_VERSION="0.15.21"
readonly CLANG_FORMAT_VERSION="20.1.5"

cd "${ROOT_DIR}"

resolve_tool() {
  local tool="$1"

  if [[ -x "${ROOT_DIR}/.venv/bin/${tool}" ]]; then
    printf '%s\n' "${ROOT_DIR}/.venv/bin/${tool}"
  else
    command -v "${tool}" || true
  fi
}

require_version() {
  local tool_name="$1"
  local tool_path="$2"
  local expected="$3"
  local actual

  if [[ -z "${tool_path}" ]]; then
    echo "error: ${tool_name} is not installed; run: python -m pip install -r requirements-style.txt" >&2
    exit 2
  fi
  actual="$(${tool_path} --version)"
  if [[ "${actual}" != *"${expected}"* ]]; then
    echo "error: expected ${tool_name} ${expected}, got: ${actual}" >&2
    exit 2
  fi
}

readonly RUFF_BIN="$(resolve_tool ruff)"
readonly CLANG_FORMAT_BIN="$(resolve_tool clang-format)"

require_version ruff "${RUFF_BIN}" "${RUFF_VERSION}"
require_version clang-format "${CLANG_FORMAT_BIN}" "${CLANG_FORMAT_VERSION}"

"${RUFF_BIN}" check --fix .
"${RUFF_BIN}" format .

cpp_files=()
while IFS= read -r -d '' file; do
  cpp_files+=("${file}")
done < <(
  git ls-files -z -- \
    '*.c' '*.cc' '*.cpp' '*.cxx' \
    '*.h' '*.hh' '*.hpp' '*.hxx' \
    ':(exclude)refs/**'
)

if ((${#cpp_files[@]} > 0)); then
  "${CLANG_FORMAT_BIN}" -i "${cpp_files[@]}"
fi
