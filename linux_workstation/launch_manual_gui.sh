#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash linux_workstation/launch_manual_gui.sh DATA_ROOT [TRIAL_ID] [extra run_pipeline args]" >&2
  exit 2
fi

DATA_ROOT="$1"
shift
TRIAL_ID=""
if [[ $# -gt 0 && "${1}" != --* ]]; then
  TRIAL_ID="$1"
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
MOVIE_KIND="${MOVIE_KIND:-corrected}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Data root does not exist: ${DATA_ROOT}" >&2
  exit 2
fi

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "Cannot find conda command: ${CONDA_BIN}" >&2
  echo "Set CONDA_BIN=/path/to/conda if needed." >&2
  exit 2
fi

cd "${REPO_ROOT}"

PIPELINE_TMPDIR="${PIPELINE_TMPDIR:-${DATA_ROOT%/}/.tmp}"
PIPELINE_CACHE_DIR="${PIPELINE_CACHE_DIR:-${DATA_ROOT%/}/.cache}"
mkdir -p "${PIPELINE_TMPDIR}" "${PIPELINE_CACHE_DIR}" "${PIPELINE_CACHE_DIR}/matplotlib"
export TMPDIR="${TMPDIR:-${PIPELINE_TMPDIR}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PIPELINE_CACHE_DIR}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PIPELINE_CACHE_DIR}/matplotlib}"

ARGS=(
  current/run_pipeline.py
  --steps manual
  --data-root "${DATA_ROOT}"
  --conda-bin "${CONDA_BIN}"
  --movie-kind "${MOVIE_KIND}"
)

if [[ -n "${TRIAL_ID}" ]]; then
  ARGS+=(--trial-id "${TRIAL_ID}")
fi

python3 "${ARGS[@]}" "$@"
