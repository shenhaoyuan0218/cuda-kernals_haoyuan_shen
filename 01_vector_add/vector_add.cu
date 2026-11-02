// W1: vector add. Run on Colab/AutoDL with nvcc (MacBook has no NVIDIA GPU).
#include <cstdio>
#include <vector>
#include "../common/cuda_check.h"

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    std::vector<float> ha(n), hb(n), hc(n), href(n);
    for (int i = 0; i < n; ++i) {
        ha[i] = 1.0f;
        hb[i] = 2.0f;
        href[i] = 3.0f;
    }

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<blocks, threads>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (hc[i] != href[i]) ++mismatches;
    }

    std::printf("vector_add n=%d blocks=%d threads=%d time=%.3f ms mismatches=%d\n",
                n, blocks, threads, ms, mismatches);

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return mismatches == 0 ? 0 : 1;
}
