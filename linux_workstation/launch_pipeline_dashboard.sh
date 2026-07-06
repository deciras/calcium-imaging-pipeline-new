#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
  RESOLVED_PATH="$(readlink -f "${SCRIPT_PATH}" 2>/dev/null || true)"
  if [[ -n "${RESOLVED_PATH}" ]]; then
    SCRIPT_PATH="${RESOLVED_PATH}"
  fi
fi
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${CONDA_BIN:-}" ]]; then
  for candidate in \
    "${HOME}/anaconda3/bin/conda" \
    "${HOME}/miniconda3/bin/conda" \
    "/opt/conda/bin/conda" \
    "/usr/local/anaconda3/bin/conda"
  do
    if [[ -x "${candidate}" ]]; then
      CONDA_BIN="${candidate}"
      break
    fi
  done
fi
CONDA_BIN="${CONDA_BIN:-conda}"
DASHBOARD_ENV="${DASHBOARD_ENV:-dashboard_gui}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1 && [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Cannot find conda command: ${CONDA_BIN}" >&2
  echo "Set CONDA_BIN=/path/to/conda if needed." >&2
  exit 2
fi

if [[ -n "${DATA_ROOT}" ]]; then
  if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Data root does not exist: ${DATA_ROOT}" >&2
    exit 2
  fi
  export DATA_ROOT

  PIPELINE_TMPDIR="${PIPELINE_TMPDIR:-${DATA_ROOT%/}/.tmp}"
  PIPELINE_CACHE_DIR="${PIPELINE_CACHE_DIR:-${DATA_ROOT%/}/.cache}"
  mkdir -p "${PIPELINE_TMPDIR}" "${PIPELINE_CACHE_DIR}" "${PIPELINE_CACHE_DIR}/matplotlib"
  export TMPDIR="${TMPDIR:-${PIPELINE_TMPDIR}}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PIPELINE_CACHE_DIR}}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${PIPELINE_CACHE_DIR}/matplotlib}"
fi

cd "${REPO_ROOT}"
exec "${CONDA_BIN}" run -n "${DASHBOARD_ENV}" --no-capture-output python current/pipeline_dashboard_gui.py "$@"
