#include "cuda_runtime.h"
#include <stdlib.h>
#include <iostream>
#include "mma.h"
using namespace std;
using namespace wmma;

__global__ void wmma_ker(half *a, half *b, float *c) {
   // Declare the fragments
   float* a_frag = new float[3*3];
   float* b_frag =  new float[3*3];
   float* c_frag =  new float[3*3];
   // Initialize the output to zero
   wmma::fill_fragment(c_frag, 0.0f);

   // Load the inputs
   wmma::load_matrix_sync(a_frag, a, 16);
   wmma::load_matrix_sync(b_frag, b, 16);

   // Perform the matrix multiplication
   wmma::mma_sync(c_frag, a_frag, b_frag);

   // Store the output
   wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}


int main()
{
   // 定义矩阵大小和分配内存
    const int N = 3;
    const int M = 3;
    const int K = 3;
    
    // 初始化输入矩阵 a 和 b
     half *h_a = new half[N * K];
    half *h_b = new half[K * M];

     // 填充输入数据
    for (int i = 0; i < N * K; i++) {
        h_a[i] = 10.0;
    }

    for (int i = 0; i < K * M; i++) {
        h_b[i] = 10.0;
    }
    // 在GPU上分配设备内存
     half *d_a, *d_b;
    float *d_c;

    cudaMalloc((void**)&d_a, N * K * sizeof(half));
    cudaMalloc((void**)&d_b, K * M * sizeof(half));
    cudaMalloc((void**)&d_c, N * M * sizeof(float));

    // 将数据从主机内存传输到设备内存
    cudaMemcpy(d_a, h_a, N * K  * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * M * sizeof(half), cudaMemcpyHostToDevice);

    // 调用 CUDA 核函数
    dim3 grid(1, 1);
    dim3 block(1, 1);
    wmma_ker<<<grid, block>>>(d_a, d_b, d_c);

    // 等待 GPU 执行完成
    cudaDeviceSynchronize();

    // 将结果从设备内存复制回主机内存
    float *h_c = new float[N * M];
    cudaMemcpy(h_c, d_c, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    // 处理和显示结果
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << h_c[i * M + j] << " ";
        }
        std::cout << " " << std::endl;
    }

    // 清理 GPU 和主机内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return  0;
   
}
