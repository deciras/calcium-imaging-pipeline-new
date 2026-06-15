#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN="${CONDA_BIN:-conda}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_EXPORT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
OUT_DIR="${1:-${CALCIUM_EXPORT_ROOT:-${DEFAULT_EXPORT_ROOT}}/conda_env_exports_$(date +%Y%m%d_%H%M%S)}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "Cannot find conda command: ${CONDA_BIN}" >&2
  echo "Set CONDA_BIN=/path/to/conda if needed." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

"${CONDA_BIN}" info > "${OUT_DIR}/conda_info.txt"
"${CONDA_BIN}" env list > "${OUT_DIR}/conda_env_list.txt"
"${CONDA_BIN}" env list --json > "${OUT_DIR}/conda_env_list.json"

python3 - "${OUT_DIR}/conda_env_list.json" > "${OUT_DIR}/env_paths.tsv" <<'PY'
import json
import re
import sys
from pathlib import Path

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

seen = set()
for env_path in payload.get("envs", []):
    path = Path(env_path)
    if env_path in seen:
        continue
    seen.add(env_path)
    name = path.name or "env"
    if name == "envs" and path.parent.name:
        name = path.parent.name
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "env"
    print(f"{env_path}\t{safe}")
PY

while IFS=$'\t' read -r env_path safe_name; do
  echo "Exporting ${safe_name} (${env_path})"
  "${CONDA_BIN}" env export -p "${env_path}" > "${OUT_DIR}/${safe_name}.yml"
  "${CONDA_BIN}" list -p "${env_path}" > "${OUT_DIR}/${safe_name}_packages.txt"
done < "${OUT_DIR}/env_paths.tsv"

echo
echo "Wrote conda environment exports to: ${OUT_DIR}"
