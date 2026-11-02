#!/usr/bin/env bash
# W1 一键检查 + 编译 + 运行（在阿里云 PAI-DSW 实例的终端里执行）
# 用法：bash run_w1.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. GPU / 驱动 =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[错误] 找不到 nvidia-smi：当前实例可能是 CPU 规格，请换 GPU 规格。" >&2; exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n1)
echo "GPU: ${GPU_NAME}"

echo "== 2. CUDA 编译器 =="
if ! command -v nvcc >/dev/null 2>&1; then
  for p in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
    [ -x "$p/nvcc" ] && export PATH="$p:$PATH" && break
  done
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[错误] 找不到 nvcc：所选镜像可能只带 CUDA runtime，没有完整 Toolkit（需自行核实），换带 devel/完整 CUDA 的官方镜像。" >&2; exit 1
fi
nvcc --version | tail -n2

echo "== 3. 编译 =="
mkdir -p build/bin
if command -v cmake >/dev/null 2>&1 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j >/dev/null; then
  echo "用 CMake 编译完成"
else
  echo "CMake 不可用或失败，改用 nvcc 直接编译"
  nvcc -O2 -I. -o build/bin/vector_add 01_vector_add/vector_add.cu
  nvcc -O2 -I. -o build/bin/matrix_add_2d 02_matrix_add_2d/matrix_add_2d.cu
fi

echo "== 4. 运行 =="
VA=$(./build/bin/vector_add); echo "$VA"
MA=$(./build/bin/matrix_add_2d); echo "$MA"

echo
echo "== 填进 README「运行记录」表 =="
echo "| $(date +%F) | $(echo "$GPU_NAME" | cut -d, -f1) | vector_add | 1<<20 | $(echo "$VA" | sed -n 's/.*time=\([0-9.]*\) ms.*/\1/p') | PAI-DSW |"
echo "| $(date +%F) | $(echo "$GPU_NAME" | cut -d, -f1) | matrix_add_2d | 1024x1024 | $(echo "$MA" | sed -n 's/.*time=\([0-9.]*\) ms.*/\1/p') | PAI-DSW |"
echo
echo "提醒：用完请回 PAI 控制台「停止」实例，关浏览器不会停止计费。"
