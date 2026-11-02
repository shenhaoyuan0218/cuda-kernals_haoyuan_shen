// W1: 2D matrix add with 2D grid/block mapping.
#include <cstdio>
#include <vector>
#include "../common/cuda_check.h"

__global__ void matrix_add_2d(const float* a, const float* b, float* c, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int rows = 1024;
    const int cols = 1024;
    const int n = rows * cols;
    const size_t bytes = n * sizeof(float);

    std::vector<float> ha(n), hb(n), hc(n);
    for (int i = 0; i < n; ++i) {
        ha[i] = static_cast<float>(i % 17);
        hb[i] = static_cast<float>(i % 13);
    }

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    matrix_add_2d<<<blocks, threads>>>(da, db, dc, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (hc[i] != ha[i] + hb[i]) ++mismatches;
    }

    std::printf("matrix_add_2d %dx%d grid=(%u,%u) block=(%u,%u) time=%.3f ms mismatches=%d\n",
                rows, cols, blocks.x, blocks.y, threads.x, threads.y, ms, mismatches);

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return mismatches == 0 ? 0 : 1;
}
