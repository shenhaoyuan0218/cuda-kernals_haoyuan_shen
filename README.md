# cuda-kernels

UNNC CS 大一 · 24 周 AI Infra 计划 · 阶段一 CUDA kernel 练习仓库。

硬件说明：MacBook 无 NVIDIA GPU，所有编译与运行都在云端完成（Colab 免费 GPU 或 AutoDL 按量实例）。

## W1 检查点

- [ ] 云端 `nvidia-smi` / `nvcc --version` 可用
- [ ] `01_vector_add` 跑通，理解 `blockIdx` / `threadIdx` 一维映射
- [ ] `02_matrix_add_2d` 跑通，理解二维 grid/block 映射
- [ ] 本 README 记下本次 GPU 型号与 kernel 耗时

## 云端快速跑

```bash
# 在 Colab 终端或 AutoDL 实例里
cd 01_vector_add
nvcc -O2 -o vector_add vector_add.cu
./vector_add

cd ../02_matrix_add_2d
nvcc -O2 -o matrix_add_2d matrix_add_2d.cu
./matrix_add_2d
```

Colab 提示：Runtime → Change runtime type → GPU；用 `!nvcc` 或终端面板编译。


## CMake（W2）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/vector_add
./build/bin/matrix_add_2d
```

## Colab 一键跑

打开 `colab/run_w1.ipynb`：Runtime → GPU，按顺序跑完单元格，把打印的 GPU 型号与耗时填回上方「运行记录」。

## 目录

| 路径 | 内容 |
|---|---|
| `CMakeLists.txt` | 统一构建（每个 kernel 一个 target） |
| `colab/run_w1.ipynb` | Colab 上编译运行 W1 两个 kernel |
| `common/cuda_check.h` | `CUDA_CHECK` 错误检查宏 |
| `01_vector_add/` | 向量加法 + `cudaEvent` 计时 |
| `02_matrix_add_2d/` | 二维矩阵加法 |

## 运行记录（自己填）

| 日期 | GPU | kernel | n / 规模 | 耗时 (ms) | 备注 |
|---|---|---|---|---|---|
| | | vector_add | 1<<20 | | |
| | | matrix_add_2d | 1024x1024 | | |

## 参考

- PMPP ch1–3（GPU 架构 / 异构计算 / 多维 grid）
- GPU MODE Lecture 2
