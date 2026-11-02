# cuda-kernels

UNNC CS 大一 · 24 周 AI Infra 计划 · 阶段一 CUDA kernel 练习仓库。

硬件说明：本机（MacBook）不装 CUDA、不编译不运行任何 kernel；所有编译与运行都在 **阿里云 PAI-DSW** 交互式实例的终端里完成。其他云 GPU 平台仅在 DSW 不可用时作为备选。

## W1 检查点

- [ ] DSW 实例里 `nvidia-smi` / `nvcc --version` 可用
- [ ] `01_vector_add` 跑通，理解 `blockIdx` / `threadIdx` 一维映射
- [ ] `02_matrix_add_2d` 跑通，理解二维 grid/block 映射
- [ ] 本 README 记下本次 GPU 型号与 kernel 耗时

## Run on Alibaba Cloud PAI-DSW

> 以下步骤依据阿里云帮助中心《创建DSW实例》《DSW计费说明》《免费试用领取、使用和释放》《DSW常见问题》（2026-11 查阅）。控制台界面、镜像名和价格会变，标「需自行核实」的以控制台为准。

### 1. 创建实例

1. （可选，先领后用）新用户可在阿里云试用中心先领 DSW 免费试用抵扣包，**必须在第一次创建 DSW 实例之前领取**；额度、有效期、可抵扣规格与地域以领取页为准（需自行核实）。
2. PAI 控制台 → 选地域 → 工作空间列表 → 进入工作空间 → 交互式建模（DSW）→ 新建实例。
3. 资源类型：公共资源（按量付费）；资源规格选一张 GPU 的规格，官方示例是 `ecs.gn7i-c8g1.2xlarge`（1×A10 24GB）。用免费试用时要在「免费试用」页签里选，规格不在抵扣范围内会直接扣费。
4. 镜像：官方镜像，搜 `cu124` 之类关键字，选带 PyTorch + CUDA 的 GPU 镜像（官方示例：`modelscope:1.31.0-pytorch2.8.0-gpu-py311-cu124-ubuntu22.04`）。**镜像里是否自带 `nvcc` / 完整 CUDA Toolkit、`cmake`，需自行核实**（下面第 3 步会检查）。
5. 存储：系统盘不要扩容（扩容部分停机也计费）；想长期保存代码/结果，配置 OSS/NAS 挂载（例如 `/mnt/data`），挂载会产生 OSS/NAS 费用（需自行核实价格）。
6. 状态变为「运行中」后点「打开」，在 JupyterLab 里新建 **Terminal**。

### 2. 拉代码

```bash
cd /mnt/workspace            # DSW 默认工作目录
git clone https://github.com/<你的用户名>/cuda-kernels.git
cd cuda-kernels
```

### 3. 环境检查

```bash
nvidia-smi                   # 看到 GPU 型号、驱动版本
nvcc --version               # 看到 CUDA 编译器版本；找不到就试 export PATH=/usr/local/cuda/bin:$PATH
cmake --version              # 没有的话可 pip install cmake，或直接用下面的 nvcc 命令
```

### 4. 编译 + 运行

一键：

```bash
bash run_w1.sh               # 检查 GPU/nvcc → 编译 → 运行两个 kernel → 打印 GPU 名和耗时，并生成可粘贴的表格行
```

手动（CMake）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/vector_add
./build/bin/matrix_add_2d
```

手动（不用 CMake）：

```bash
nvcc -O2 -I. -o vector_add 01_vector_add/vector_add.cu && ./vector_add
nvcc -O2 -I. -o matrix_add_2d 02_matrix_add_2d/matrix_add_2d.cu && ./matrix_add_2d
```

### 5. 记录 GPU 型号与耗时

`run_w1.sh` 最后会打印两行表格（日期、GPU、kernel、规模、耗时 ms）。把它们粘贴到下面「运行记录」表里替换空行；kernel 输出里的 `mismatches` 必须是 0。建议同一配置跑 3 次取中位数，第一次运行含 CUDA 初始化开销，可能偏慢。

### 6. 在 DSW 里提交并推送

```bash
git config --global user.name  "Haoyuan Shen"
git config --global user.email "<你的 GitHub 邮箱>"
git add README.md && git commit -m "W1: record DSW GPU timings"
```

二选一：

- **HTTPS + Personal Access Token**：GitHub → Settings → Developer settings → Fine-grained token，只授权本仓库 Contents 读写。`git push` 时用户名填 GitHub 用户名、密码填 token。不要把 token 写进代码、notebook 或提交记录。
- **SSH key**：在 DSW 终端 `ssh-keygen -t ed25519 -C "dsw"`，把 `~/.ssh/id_ed25519.pub` 加到 GitHub → Settings → SSH keys，然后 `git remote set-url origin git@github.com:<你的用户名>/cuda-kernels.git`。注意密钥在系统盘上，实例删除或长期停机清盘后要重新生成（需自行核实 DSW 能否直连 github.com 的 22 端口，不行就用 HTTPS）。

### 7. ⚠️ 用完立刻停止实例

- 实例处于「运行中」就按时长计费，**不跑代码也计费；关浏览器/退出登录不会停止**。用完回控制台点「停止」；不再需要就「删除」。
- 免费额度用尽或到期后实例不会自动停，会转为按量付费。闲置自动关机策略不保证触发。
- 账单有数小时延迟，停机后收到扣费通知通常是之前的用量。
- 持久化：官方文档写明未扩容的免费系统盘，实例停止超过 15 天内容会被清空；删除实例则系统盘一起释放。**每次结束前先 `git push`**，大文件放 OSS/NAS 挂载目录。具体保留规则以控制台为准（需自行核实）。

### 需自行核实

- 所选镜像是否带 `nvcc`、完整 CUDA Toolkit、`cmake`
- GPU 规格的库存与按量单价、OSS/NAS 存储费用
- 免费试用资格、额度、有效期、可抵扣规格和地域（只能新用户领一次，且须先领后用）
- 容器内 `ncu` / Nsight Compute、Nsight Systems 能否使用（可能缺工具或缺性能计数器权限，W7 前要验证）
- 系统盘/挂载存储的持久化行为与保留期限
- DSW 能否访问 github.com（HTTPS/SSH）及网速

## CMake（W2，在 DSW 里）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/vector_add
./build/bin/matrix_add_2d
```

## 目录

| 路径 | 内容 |
|---|---|
| `CMakeLists.txt` | 统一构建（每个 kernel 一个 target） |
| `run_w1.sh` | DSW 终端里一键检查环境、编译、运行 W1 两个 kernel |
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
