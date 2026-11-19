// 彩色转灰度、图像模糊、方阵乘法。

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

constexpr int CHANNELS = 3;   // 每个彩色像素依次存放 R、G、B
constexpr int BLUR_SIZE = 1;  // 模糊半径为 1，对应 3 × 3 邻域

// 第 3.2 节，图 3.4：彩色图像转灰度图像。
// __global__ 表示这是在 GPU 上执行的核函数。
// 一个线程负责一个像素。Pin 是 RGB 输入，Pout 是灰度输出。
__global__ void colorToGrayscaleConversion(const unsigned char* Pin,
                                           unsigned char* Pout,
                                           int width,
                                           int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //限定线程，“舍弃”多余线程                        
    if (row < height && col < width) {
       // 灰度图每个像素占一个元素，RGB 图每个像素占三个元素。
        int grayOffset = row * width + col;
        
        int rgbOffset = grayOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[grayOffset] = static_cast<unsigned char>(
            0.21f * r + 0.72f * g + 0.07f * b
        ); 
    }
}

// 第 3.3 节，图 3.8：灰度图像模糊。
// 一个线程读取当前像素及其邻居，将平均值写入输出图像。#BLUR_SIZE 为模糊半径
__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < w && row < h) {
        int pixVal = 0;//累加灰度值

        int pixels = 0;//累加会读数

        // Get average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                // Verify we have a valid image pixel
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) { //边界条件判断
                    pixVal += in[curRow * w + curCol];
                    ++pixels; // Keep track of number of pixels in the avg
                }
            }
        }
        // Write our new pixel value out
        out[row * w + col] = (unsigned char)(pixVal / pixels);
    }
}

// 第 3.4 节，图 3.11：方阵乘法 P = M × N。//仅能计算方阵
// 一个线程计算 P[row][col]，即 M 的第 row 行与 N 的第 col 列的点积。

__global__ void matrixmulKernel(float* M, float* N,
                                float* P, int Width) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}

// 主机端辅助函数：检查 CUDA 调用的返回值。#这个是错误检查函数
void check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

// 选择要学习的例子：1 彩色转灰度，2 灰度图像模糊，3 方阵乘法。
// 可以改下面的默认值，或者在 nvcc 命令中加 -DDEMO=2 等参数。
#ifndef DEMO
#define DEMO 1
#endif

#if DEMO == 1
// 彩色转灰度：每个输入像素连续占 3 字节，每个输出像素占 1 字节。
int main() {
    constexpr int width = 2, height = 2; // constexpr：常量表达式
    const unsigned char rgb_h[] = {100, 0, 0,   0, 100, 0, // const：常量
                                   0, 0, 100,   0, 0, 0};
    unsigned char gray_h[width * height];
    unsigned char *rgb_d, *gray_d;

    check(cudaMalloc(&rgb_d, sizeof(rgb_h))); //关于“&”的遗留问题：rgb_d 是 CPU 端的一个变量，它的工作是保存 GPU 地址。 刚开始还没分配显存，所以它暂时存着 nullptr。现在你希望 cudaMalloc 分配显存后，把得到的地址写进 rgb_d 这个变量里。要让函数能写它，就传 &rgb_d，也就是这个变量自身的地址。
    check(cudaMalloc(&gray_d, sizeof(gray_h)));
    check(cudaMemcpy(rgb_d, rgb_h, sizeof(rgb_h), cudaMemcpyHostToDevice));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(rgb_d, gray_d, width, height);
    check(cudaGetLastError()); //检查报错
    check(cudaMemcpy(gray_h, gray_d, sizeof(gray_h), cudaMemcpyDeviceToHost));
    check(cudaFree(rgb_d));
    check(cudaFree(gray_d));

    const unsigned char expected[] = {21, 72, 7, 0};
    bool passed = true;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int index = row * width + col;
            std::printf("%d ", gray_h[index]);
            if (gray_h[index] != expected[index]) passed = false;
        }
        std::puts("");
    }
    std::puts(passed ? "PASS" : "FAIL");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

#elif DEMO == 2
// 独立使用一张灰度图，无需先运行彩色转灰度的例子。
int main() {
    constexpr int width = 3, height = 3;
    const unsigned char image_h[] = {10, 20, 30,
                                     40, 50, 60,
                                     70, 80, 90};
    unsigned char blur_h[width * height];
    unsigned char *image_d, *blur_d;

    check(cudaMalloc(&image_d, sizeof(image_h)));
    check(cudaMalloc(&blur_d, sizeof(blur_h)));
    check(cudaMemcpy(image_d, image_h, sizeof(image_h), cudaMemcpyHostToDevice));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    blurKernel<<<dimGrid, dimBlock>>>(image_d, blur_d, width, height);
    check(cudaGetLastError());
    check(cudaMemcpy(blur_h, blur_d, sizeof(blur_h), cudaMemcpyDeviceToHost));
    check(cudaFree(image_d));
    check(cudaFree(blur_d));

    const unsigned char expected[] = {30, 35, 40, 45, 50, 55, 60, 65, 70};
    bool passed = true;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int index = row * width + col;
            std::printf("%d ", blur_h[index]);
            if (blur_h[index] != expected[index]) passed = false;
        }
        std::puts("");
    }
    std::puts(passed ? "PASS" : "FAIL");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

#elif DEMO == 3
// 两个 2 × 2 方阵相乘。
int main() {
    constexpr int width = 2;
    const float M_h[] = {1, 2, 3, 4};
    const float N_h[] = {5, 6, 7, 8};
    float P_h[width * width];
    float *M_d, *N_d, *P_d;

    check(cudaMalloc(&M_d, sizeof(M_h)));
    check(cudaMalloc(&N_d, sizeof(N_h)));
    check(cudaMalloc(&P_d, sizeof(P_h)));
    check(cudaMemcpy(M_d, M_h, sizeof(M_h), cudaMemcpyHostToDevice));
    check(cudaMemcpy(N_d, N_h, sizeof(N_h), cudaMemcpyHostToDevice));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (width + dimBlock.y - 1) / dimBlock.y);
    matrixmulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);
    check(cudaGetLastError());
    check(cudaMemcpy(P_h, P_d, sizeof(P_h), cudaMemcpyDeviceToHost));
    check(cudaFree(M_d));
    check(cudaFree(N_d));
    check(cudaFree(P_d));

    const float expected[] = {19, 22, 43, 50};
    bool passed = true;
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < width; ++col) {
            int index = row * width + col;
            std::printf("%.0f ", P_h[index]);
            if (P_h[index] != expected[index]) passed = false;
        }
        std::puts("");
    }
    std::puts(passed ? "PASS" : "FAIL");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else
#error "DEMO must be 1, 2, or 3"
#endif
