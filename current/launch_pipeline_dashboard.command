#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${CONDA_BIN:-}" ]]; then
  for candidate in \
    "${HOME}/anaconda3/bin/conda" \
    "${HOME}/miniconda3/bin/conda" \
    "/opt/homebrew/anaconda3/bin/conda" \
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
DASHBOARD_ENV="${DASHBOARD_ENV:-caiman}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1 && [[ ! -x "${CONDA_BIN}" ]]; then
  echo "未找到 conda：${CONDA_BIN}"
  echo "请先设置 CONDA_BIN，或安装/恢复 conda。"
  read -r "?按回车键退出..."
  exit 2
fi

cd "${REPO_ROOT}"
echo "正在启动 calcium pipeline dashboard..."
echo "代码目录：${REPO_ROOT}"
echo "conda：${CONDA_BIN}"
echo "环境：${DASHBOARD_ENV}"

if ! "${CONDA_BIN}" run -n "${DASHBOARD_ENV}" --no-capture-output python current/pipeline_dashboard_gui.py "$@"; then
  echo
  echo "Dashboard 启动失败。"
  read -r "?按回车键退出..."
  exit 1
fi
