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