#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash linux_workstation/run_00_05_suite2p.sh DATA_ROOT [extra run_pipeline args]" >&2
  echo "Example: bash linux_workstation/run_00_05_suite2p.sh /data/experiment --dry-run" >&2
  exit 2
fi

DATA_ROOT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Data root does not exist: ${DATA_ROOT}" >&2
  exit 2
fi

CONDA_BIN="${CONDA_BIN:-conda}"
ACTION="${ACTION:-skip}"
FIJI_MEMORY="${FIJI_MEMORY:-24G}"
PROJECTION_MODE="${PROJECTION_MODE:-auto}"
STIM_EXPORT_MODE="${STIM_EXPORT_MODE:-projected}"
METADATA_MODE="${METADATA_MODE:-update-missing}"
ANALOG_SOURCE="${ANALOG_SOURCE:-auto}"
RAW_Z_STRATEGY="${RAW_Z_STRATEGY:-planes}"
SIGMA_PX="${SIGMA_PX:-12}"
SUITE2P_THREADS="${SUITE2P_THREADS:-8}"
N_WORKERS="${N_WORKERS:-1}"
NUM_THREADS="${NUM_THREADS:-1}"
CELL_DIAMETER_UM="${CELL_DIAMETER_UM:-8}"
DIAMETER_SCALE="${DIAMETER_SCALE:-1.3}"
THRESHOLD_SCALING="${THRESHOLD_SCALING:-1.2}"
PREPARE_STIM_LOGS="${PREPARE_STIM_LOGS:-1}"
STIM_LOG_MODE="${STIM_LOG_MODE:-symlink}"

PIPELINE_DRY_RUN=0
for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    PIPELINE_DRY_RUN=1
  fi
done

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "Cannot find conda command: ${CONDA_BIN}" >&2
  echo "Set CONDA_BIN=/path/to/conda if needed." >&2
  exit 2
fi

if [[ -z "${FIJI_BIN:-}" && -z "${FIJI_PATH:-}" ]]; then
  echo "Warning: FIJI_BIN/FIJI_PATH is not set. Step 01 will use built-in candidate paths." >&2
fi

LOG_ROOT="${DATA_ROOT%/}/pipeline_logs"
mkdir -p "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/linux_00_05_suite2p_${STAMP}.log"

echo "Repository : ${REPO_ROOT}"
echo "Data root  : ${DATA_ROOT}"
echo "Action     : ${ACTION}"
echo "Log file   : ${LOG_FILE}"
echo

cd "${REPO_ROOT}"

if [[ "${PREPARE_STIM_LOGS}" != "0" ]]; then
  PREPARE_ARGS=("${DATA_ROOT}" --mode "${STIM_LOG_MODE}" --overwrite)
  if [[ "${PIPELINE_DRY_RUN}" == "0" ]]; then
    PREPARE_ARGS+=(--execute)
  fi
  python3 linux_workstation/prepare_stim_logs.py "${PREPARE_ARGS[@]}"
  echo
fi

python3 current/run_pipeline.py \
  --steps premanual \
  --data-root "${DATA_ROOT}" \
  --action "${ACTION}" \
  --conda-bin "${CONDA_BIN}" \
  --fiji-memory "${FIJI_MEMORY}" \
  --projection-mode "${PROJECTION_MODE}" \
  --stim-export-mode "${STIM_EXPORT_MODE}" \
  --metadata-mode "${METADATA_MODE}" \
  --analog-source "${ANALOG_SOURCE}" \
  --raw-z-strategy "${RAW_Z_STRATEGY}" \
  --sigma-px "${SIGMA_PX}" \
  --suite2p-threads "${SUITE2P_THREADS}" \
  --n-workers "${N_WORKERS}" \
  --num-threads "${NUM_THREADS}" \
  --cell-diameter-um "${CELL_DIAMETER_UM}" \
  --diameter-scale "${DIAMETER_SCALE}" \
  --threshold-scaling "${THRESHOLD_SCALING}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
