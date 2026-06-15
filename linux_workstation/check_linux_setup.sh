#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"

echo "Repository: ${REPO_ROOT}"
echo

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "Missing conda command: ${CONDA_BIN}" >&2
  echo "Set CONDA_BIN=/path/to/conda if needed." >&2
  exit 1
fi

echo "Conda: $(${CONDA_BIN} --version)"
echo

for env_name in fiji_env caiman suite2p; do
  if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${env_name}"; then
    echo "OK conda env: ${env_name}"
  else
    echo "Missing conda env: ${env_name}" >&2
  fi
done

echo
if [[ -n "${FIJI_BIN:-}" ]]; then
  if [[ -x "${FIJI_BIN}" || -f "${FIJI_BIN}" ]]; then
    echo "OK FIJI_BIN: ${FIJI_BIN}"
  else
    echo "FIJI_BIN is set but not found: ${FIJI_BIN}" >&2
  fi
elif [[ -n "${FIJI_PATH:-}" ]]; then
  if [[ -x "${FIJI_PATH}" || -f "${FIJI_PATH}" ]]; then
    echo "OK FIJI_PATH: ${FIJI_PATH}"
  else
    echo "FIJI_PATH is set but not found: ${FIJI_PATH}" >&2
  fi
else
  echo "FIJI_BIN/FIJI_PATH is not set. Set it before running step 01." >&2
fi

echo
cd "${REPO_ROOT}"
python3 current/run_pipeline.py --list-steps
