// 向量加法：CPU 版（PMPP 图 2.4）
#include <stdio.h>        // printf 等输入输出
#include <stdlib.h>       // malloc、free
#include <cuda_runtime.h> // CUDA 接口；这个 CPU 版其实用不到，Mac 上标红正常

// A、B、C 是指针：只存数组开头的地址，不存内容
// 内容在 main 里 malloc 出来的那块内存（堆）里
// 指针不知道数组多长，所以要另外传 N
void vecAdd(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        // A[i] 等于 *(A + i)：从开头往后跳 i 格，取出那格的数
        // 这里的 A 和 main 里的 A 指向同一块内存，所以改 C 会直接改到 main 的 C
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000;

    // malloc：在 CPU 内存（堆）里申请 N 个 float 的连续空间，只划地方、不放数
    // sizeof(float) = 一个 float 的字节数（4），乘 N 得到总字节数
    // (float*)：把 malloc 返回的地址转成「指向 float 的指针」
    // 必须先 malloc 再写 A[i]，否则 A 指向的不是自己的空间
    float *A = (float*)malloc(N * sizeof(float));
    float *B = (float*)malloc(N * sizeof(float));
    float *C = (float*)malloc(N * sizeof(float));

    // 填测试数据：这一步才往申请的空间里放内容
    // A = 0,1,2,...；B = 0,2,4,...（2.0f 中的 f 表示 float）
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2.0f * i;
    }

    // 传的是三个数组的起始地址 + 长度，不会复制整个数组
    vecAdd(A, B, C, N);
    // 此时 C[i] 应为 3*i

    // 把 malloc 申请的内存还给系统
    free(A);
    free(B);
    free(C);
    return 0;
}

// 向量加法：GPU版 （PMPP 图2.5）
// 核函数：在 GPU 上运行，每个线程负责一个元素
#include <stdio.h>        // printf
#include <stdlib.h>       // malloc、free（main 里要用）
#include <cuda_runtime.h> // cudaMalloc、cudaMemcpy、cudaFree 等 CUDA 函数

__global__ void vecAddKernel(float* A, float* B, float* C, int N) {
    // 算出当前线程的全局编号 i = 块大小 × 块编号 + 块内编号
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // 线程总数可能多于 N，多出来的线程不做事，防止越界
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// 主机函数：搬数据、启动核函数、取回结果
void vecAdd(float* A, float* B, float* C, int N) {
    int size = N * sizeof(float);   // 整个数组占多少字节
    float *d_A, *d_B, *d_C;         // 存 GPU 显存里的地址，d = device

    // 第 1 部分：在 GPU 显存里另开三块空间，并把 A、B 的内容复制过去
    // 复制之后 CPU 和 GPU 各有一份数据，两边互不相通
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);  // CPU → GPU
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // 第 2 部分：启动核函数，每块 256 个线程，块数向上取整，保证线程总数够 N 个
    // 传的是 d_ 开头的 GPU 地址，核函数只能读写显存
    vecAddKernel<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);

    // 第 3 部分：把结果从 GPU 拷回 CPU 的 C，主机代码才看得到
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost); // GPU → CPU
    // 释放 GPU 显存；CPU 上的 A、B、C 由 main 里的 free 释放
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 1000;
    float *A = (float*)malloc(N * sizeof(float));
    float *B = (float*)malloc(N * sizeof(float));
    float *C = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2.0f * i;
    }
    vecAdd(A, B, C, N);
    printf("%f\n", C[10]);  // 应该是 30.000000
    free(A);
    free(B);
    free(C);
    return 0;
}