#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -lt 1 ]; then
  echo "usage: $0 LEROBOT_V3_DATASET [OUTPUT_DIR] [extra evaluate options...]" >&2
  exit 64
fi

dataset="$1"
shift
output="${1:-${project_dir}/outputs/tianji_arm_quality}"
if [ "$#" -gt 0 ]; then
  shift
fi

python_bin="${project_dir}/.venv/bin/python"
if [ ! -x "${python_bin}" ] || ! "${python_bin}" -c 'import numpy, pyarrow, scipy' >/dev/null 2>&1; then
  echo "首次运行：正在创建环境并安装/编译依赖…"
  "${project_dir}/setup.sh"
fi

exec "${python_bin}" "${project_dir}/scripts/evaluate_arm_quality.py" \
  --dataset "${dataset}" \
  --output "${output}" \
  "$@"
