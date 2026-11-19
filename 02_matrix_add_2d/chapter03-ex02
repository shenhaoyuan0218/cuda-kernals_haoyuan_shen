#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

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

int main(){
    void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
        }
    }

void matrixVectorMul(float* A_h, const float* B_h,
                     const float* C_h, int Width) {
    if (Width <= 0) return;

    size_t vectorBytes = static_cast<size_t>(Width) * sizeof(float); //size_t:专门用来表示大小或元素数量的无符号整数类型 (典例：sizeof()...)
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
}