# PMPP 第四版 · 第 3 章课后练习答案与解析

《大规模并行处理器程序设计（原书第 4 版）》第 3 章「多维网格和数据」。以下下标从 0 开始。第 1、2 题的矩阵均为单精度方阵，按行连续存储，边长为 `Width`。

## 第 1 题　矩阵乘法的两种线程分工

设输出矩阵为 `P = M × N`。图 3.11 中，一个线程负责一个输出元素：固定 `row` 和 `col`，沿 `k` 计算一行与一列的点积。本题把每个线程的任务扩大到一整行或一整列。

### (a) 一个线程计算一整行

**线程编号决定行号，外层循环遍历列，内层循环计算点积。** 只需给每一行分配一个线程，因此可以采用一维网格。这里把全局线程编号赋给 `row`，即使编号来自 `.x`，它仍然可以表示行。

```cuda
__global__ void matrixMulByRow(const float* M, const float* N,
                               float* P, int Width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Width) return;

    for (int col = 0; col < Width; ++col) {
        float Pvalue = 0.0f;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}
```

设设备数组已分配，且 `Width > 0`。以下配置使线程总数覆盖所有行：

```cuda
int threads = 256;
int blocks = (Width - 1) / threads + 1;
matrixMulByRow<<<blocks, threads>>>(M_d, N_d, P_d, Width);
```

`Pvalue` 在列循环内部初始化。每换一列，线程就开始计算另一个输出元素，累加值也要重新从零开始。

### (b) 一个线程计算一整列

**线程编号决定列号，外层循环遍历行，内层循环计算点积。** 例如，列号为 1 的线程依次写入第 0 行第 1 列、第 1 行第 1 列，直到该列底部。

```cuda
__global__ void matrixMulByCol(const float* M, const float* N,
                               float* P, int Width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= Width) return;

    for (int row = 0; row < Width; ++row) {
        float Pvalue = 0.0f;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}
```

方阵有 `Width` 列，因此启动配置同样需要覆盖 `Width` 个有效线程：

```cuda
int threads = 256;
int blocks = (Width - 1) / threads + 1;
matrixMulByCol<<<blocks, threads>>>(M_d, N_d, P_d, Width);
```

### (c) 两种设计的优缺点

| 观察对象 | 每线程一行 | 每线程一列 |
|---|---|---|
| 单个线程连续两次写入 P | 写同一行的相邻列，相隔 1 个元素 | 写同一列的相邻行，相隔 Width 个元素 |
| 同一步中，相邻线程写入 P | 写不同行的同一列，相隔 Width 个元素 | 写同一行的相邻列，地址连续 |
| 同一步中，相邻线程读取输入 | 读取 M 中不同行的元素，共用 N 中的同一元素 | 共用 M 中的同一元素，读取 N 中相邻的列元素 |

按行分工时，单个线程可以反复使用 M 的同一行。按列分工时，相邻线程在同一步读取 N 和写入 P 的地址连续，有利于 GPU 合并访存。两种方案都只有 `Width` 个有效线程，各算 `Width` 个结果。图 3.11 的逐元素方案则有 `Width × Width` 个有效线程，提供了更多可以并行执行的任务。实际耗时还取决于矩阵尺寸、缓存和硬件。

## 第 2 题　矩阵与向量相乘

输入为方阵 B 和向量 C，输出为向量 A。**一个线程负责一个 A[row]**，将 B 的第 row 行和 C 中对应的元素相乘，再累加。若 B 的第 0 行是 `[1, 2]`，C 是 `[5, 6]`，则 `A[0] = 1 × 5 + 2 × 6 = 17`。

### 核函数

```cuda
__global__ void matrixVectorKernel(float* A, const float* B,
                                   const float* C, int Width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Width) return;

    float sum = 0.0f;
    for (int k = 0; k < Width; ++k) {
        sum += B[row * Width + k] * C[k];
    }
    A[row] = sum;
}
```

### 主机端函数

四个参数依次是输出向量地址、输入矩阵地址、输入向量地址和方阵边长。下面的 `check` 检查 CUDA 调用的返回值；相关头文件放在同一份 `.cu` 文件顶部。

```cuda
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void matrixVectorMul(float* A_h, const float* B_h,
                     const float* C_h, int Width) {
    if (Width <= 0) return;

    size_t vectorBytes = static_cast<size_t>(Width) * sizeof(float);
    size_t matrixBytes = static_cast<size_t>(Width) * vectorBytes;
    float *A_d, *B_d, *C_d;

    check(cudaMalloc(&A_d, vectorBytes));
    check(cudaMalloc(&B_d, matrixBytes));
    check(cudaMalloc(&C_d, vectorBytes));

    check(cudaMemcpy(B_d, B_h, matrixBytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(C_d, C_h, vectorBytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (Width - 1) / threads + 1;
    matrixVectorKernel<<<blocks, threads>>>(A_d, B_d, C_d, Width);
    check(cudaGetLastError());

    check(cudaMemcpy(A_h, A_d, vectorBytes, cudaMemcpyDeviceToHost));
    check(cudaFree(A_d));
    check(cudaFree(B_d));
    check(cudaFree(C_d));
}
```

B 有 `Width × Width` 个元素，A 和 C 各有 `Width` 个元素。分配时传入 `&A_d`，使 CUDA 能把显存地址写入指针变量；启动核函数时传入 `A_d`，使线程能按该地址写入结果。计算结束后只需把输出向量 A 复制回主机。

## 第 3 题　二维网格的线程计数

矩阵有 150 行、300 列，线程块配置为 `dim3 bd(16, 32)`。第一个参数对应 x 方向，每块覆盖 16 列；第二个对应 y 方向，每块覆盖 32 行。横向至少需要 19 个块，因为 18 个块只能覆盖 288 列。纵向至少需要 5 个块，因为 4 个块只能覆盖 128 行。

| 小问 | 计算 | 答案 |
|---|---|---:|
| (a) 每块的线程数 | 16 × 32 | **512** |
| (b) 网格启动的线程总数 | 19 × 5 × 512 | **48,640** |
| (c) 网格的块数 | 19 × 5 | **95** |
| (d) 执行第 05 行赋值的线程数 | 150 × 300 | **45,000** |

网格覆盖 304 列、160 行。所有线程都会计算坐标，但只有满足 `row < 150 && col < 300` 的线程执行赋值。有 48,640 − 45,000 = **3,640** 个线程落在矩阵边界之外。

## 第 4 题　二维矩阵的行主序和列主序

矩阵宽 400、高 500，目标元素的行下标为 20、列下标为 10。

| 小问 | 一维下标 | 代入 | 答案 |
|---|---|---|---:|
| (a) 行主序 | `row * width + col` | 20 × 400 + 10 | **8,010** |
| (b) 列主序 | `col * height + row` | 10 × 500 + 20 | **5,020** |

行主序中，每跨过一整行就跳过 400 个元素。先跨过 20 行，再向右走 10 个元素。列主序中，每跨过一整列就跳过 500 个元素。先跨过 10 列，再向下走 20 个元素。

## 第 5 题　三维张量的一维下标

张量宽 400、高 500、深 300，按行主序存储。坐标 `(x, y, z) = (10, 20, 5)` 对应 `A[5][20][10]`。

| 移动 | 跨过的元素数 |
|---|---:|
| 跳过前 5 个完整的层 | 5 × 500 × 400 = 1,000,000 |
| 在当前层跳过前 20 行 | 20 × 400 = 8,000 |
| 在当前行移动到列下标 10 | 10 |

**一维下标为 1,008,010**。通式为 `index = (z * height + y) * width + x`。深度 300 决定 z 的合法范围为 0～299；跨过一层的距离由每层的高度和宽度决定。

## 参考资料

- [机械工业出版社：《大规模并行处理器程序设计（原书第 4 版）》书目及目录](https://www.cmpedu.com/books/book/5610306.htm)
- [原书第 3 章：Multidimensional Grids and Data（DOI）](https://doi.org/10.1016/B978-0-323-91231-0.00004-5)
