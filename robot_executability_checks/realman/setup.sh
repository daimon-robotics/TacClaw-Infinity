#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip_index_url="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
python3 -m venv "$project_dir/.venv"
"$project_dir/.venv/bin/python" -m pip install --index-url "$pip_index_url" --upgrade pip
"$project_dir/.venv/bin/python" -m pip install --index-url "$pip_index_url" -r "$project_dir/requirements.txt"

if ! command -v cmake >/dev/null 2>&1; then
  echo "缺少 cmake。请安装 CMake 后重新运行 ./setup.sh。" >&2
  exit 1
fi
if ! command -v c++ >/dev/null 2>&1; then
  echo "缺少 C++ 编译器。Debian/Ubuntu 可安装 build-essential 后重新运行 ./setup.sh。" >&2
  exit 1
fi

echo "正在编译 Tianji Aik7 解析 IK 后端…"
cmake -S "$project_dir/native" -B "$project_dir/native/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$project_dir/native/build" --config Release

echo "安装完成。运行："
echo "  $project_dir/.venv/bin/python $project_dir/scripts/replay_lerobot.py --help"
