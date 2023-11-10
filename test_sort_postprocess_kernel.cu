#include "cuda_runtime.h"
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>

#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                         \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;           \
  for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int)
using namespace std;

__global__ void sort_postprocess_kernel( int* in, int* out, int64_t* index, int2* i_s_ptr,int nsegments,int nsort) {
  CUDA_KERNEL_LOOP(i, nsegments * nsort) {
    int segment = i / nsort;
    int j = i % nsort;

    int offset = segment * nsort;
    int* in_ = in + offset;
    int* out_ = out + offset;
    int64_t* index_ = index + offset;
    int2* i_s_ptr_ = i_s_ptr + offset;

    int idx = i_s_ptr_[j].y;
    index_[j] = idx;
    out_[j] = in_[idx];
  }
}

int main()
{
   const int kDataLen = 4;

   int in[kDataLen] = {1, 2, 3, 4};
   int out[kDataLen] = {1, 2, 3, 4};
   int64_t index[kDataLen] = {0 ,1 ,2 ,3};
   int2 i_s_ptr[kDataLen];
   int nsegments =4; int nsort=4;

   for (int i = 0; i < kDataLen; i++) {
        i_s_ptr[i].x = i * 2;  // 初始化 x 元素
        i_s_ptr[i].y = i * 2 + 1;  // 初始化 y 元素
    }
  

   int* device_in;
   int* device_out;
   int64_t* device_index;
   int2* device_i_s_ptr;

  cudaMalloc(&device_in, kDataLen * sizeof(int));
  cudaMalloc(&device_out, kDataLen * sizeof(int));
  cudaMalloc(&device_index, kDataLen * sizeof(int64_t));
  cudaMalloc(&device_i_s_ptr, kDataLen * sizeof(int2));

  cudaMemcpy(device_in, in, kDataLen * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_out, out, kDataLen * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_index, index, kDataLen * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(device_i_s_ptr, i_s_ptr, kDataLen * sizeof(int2), cudaMemcpyHostToDevice);

    // Launch the kernel.
   sort_postprocess_kernel<<<1, kDataLen>>>(device_in , device_out , device_index , device_i_s_ptr , nsegments , nsort );

    // Copy output data to host.
    cudaDeviceSynchronize();
    cudaMemcpy(out, device_out, kDataLen * sizeof(int),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < kDataLen; ++i) {
      cout << "out[" << i << "] = " << out[i] << endl;
    }

    cudaDeviceReset();
    return 0;
}
