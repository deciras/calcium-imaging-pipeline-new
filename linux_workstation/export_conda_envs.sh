#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-conda}"
OUT_DIR="${1:-conda_env_exports_$(date +%Y%m%d_%H%M%S)}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "Cannot find conda command: ${CONDA_BIN}" >&2
  echo "Set CONDA_BIN=/path/to/conda if needed." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

"${CONDA_BIN}" info > "${OUT_DIR}/conda_info.txt"
"${CONDA_BIN}" env list > "${OUT_DIR}/conda_env_list.txt"

"${CONDA_BIN}" env list | awk '
  /^[[:space:]]*#/ {next}
  NF >= 1 {
    name=$1
    if (name == "*") {
      name=$2
    }
    if (name != "" && name != "base") {
      print name
    }
  }
' | sort -u > "${OUT_DIR}/env_names.txt"

while IFS= read -r env_name; do
  safe_name="$(printf '%s' "${env_name}" | tr '/ :' '___')"
  echo "Exporting ${env_name}"
  "${CONDA_BIN}" env export -n "${env_name}" > "${OUT_DIR}/${safe_name}.yml"
  "${CONDA_BIN}" list -n "${env_name}" > "${OUT_DIR}/${safe_name}_packages.txt"
done < "${OUT_DIR}/env_names.txt"

echo
echo "Wrote conda environment exports to: ${OUT_DIR}"
