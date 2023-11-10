#include "cuda_runtime.h"
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>

using namespace std;

__global__ void _bfloat16_to_float_cuda_kernel(uint16_t* __restrict__ input, int nrows, int ncols, float* __restrict__ output) {
   int row_incre = blockDim.y * gridDim.y;
   int col_incre = blockDim.x * gridDim.x;
  #pragma unroll
  for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < nrows; row += row_incre) {
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < ncols;col += col_incre) {
      uint16_t* input_row = input + row * ncols;
      float* output_row = output + row * ncols;
      uint32_t val_fp32 = static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(input_row)[col]) << 16;
      reinterpret_cast<uint32_t*>(output_row)[col] = val_fp32;
    }
  }
}

int main()
{
   const int kDataLen = 4;

   uint16_t input[kDataLen] = {1, 2, 3, 4}; 
   float output[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
   int nrows =4; int ncols =4;


   uint16_t* device_input;
   float* device_output;

  cudaMalloc(&device_input, kDataLen * sizeof(int));
  cudaMalloc(&device_output, kDataLen * sizeof(float));

  cudaMemcpy(device_input, input, kDataLen * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_output, output, kDataLen * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel.
   _bfloat16_to_float_cuda_kernel<<<1, kDataLen>>>(device_input , nrows , ncols , device_output);

    // Copy output data to host.
    cudaDeviceSynchronize();
    cudaMemcpy(output, device_output, kDataLen * sizeof(float),
               cudaMemcpyDeviceToHost);

     for (int i = 0; i < kDataLen; ++i) {
      cout << "output[" << i << "] = " << output[i] << endl;
    }

    cudaDeviceReset();
    return 0;
}
