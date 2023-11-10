#include "cuda_runtime.h"
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>

#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                         \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;           \
  for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int)
using namespace std;

__global__ void avg_pool2d_out_cuda_frame(int nthreads,
    int* bottom_data,  int64_t channels,
     int64_t height, int64_t width, int64_t pooled_height,
    int pooled_width, int kernel_h, int kernel_w,
    int stride_h,  int stride_w, int pad_h, int pad_w,
    int* top_data,  int divisor_override,
     bool count_include_pad,  bool use_divisor) {
  CUDA_KERNEL_LOOP(index, nthreads) {
     int pw = index % pooled_width;
     int ph = (index / pooled_width) % pooled_height;
     int c = (index / pooled_width / pooled_height) % channels;
     int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);

    if (hstart >= hend || wstart >= wend) {
      top_data[index] = 0;
      continue;
    }

    int aveval = 0;
    int* bottom_slice = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    int divide_factor;
    if (use_divisor) {
      divide_factor = divisor_override;
    } else {
      if(count_include_pad) {
        divide_factor = pool_size;
      } else {
        divide_factor = (hend - hstart) * (wend - wstart);
      }
    }
    top_data[index] = static_cast<int>(aveval / divide_factor);
  }
}

int main()
{
   const int kDataLen = 4;

   int bottom_data[kDataLen] = {10, 222, 3333, 40001}; 
   int top_data[kDataLen] = {111, 2222, 33333, 444444};
   int nthreads =100;
   int64_t channels = 20; int64_t height =100;  int64_t width =10;  int64_t pooled_height =10;
    int pooled_width =12; int kernel_h =10; int kernel_w =10;
    int stride_h =19;  int stride_w =23;   int pad_h =1000;   int pad_w = 122;
    int divisor_override =100;
     bool count_include_pad = true;   bool use_divisor =false;
   

   int* device_bottom_data;
   int* device_top_data;
  
  cudaMalloc(&device_bottom_data, kDataLen * sizeof(int));
  cudaMalloc(&device_top_data, kDataLen * sizeof(int));

  cudaMemcpy(device_bottom_data, bottom_data, kDataLen * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_top_data, top_data, kDataLen * sizeof(int), cudaMemcpyHostToDevice);
   // Launch the kernel.
  avg_pool2d_out_cuda_frame<<<1, kDataLen>>>(nthreads , device_bottom_data , channels , height , width , pooled_height , pooled_width ,kernel_h , kernel_w , stride_h , stride_w , pad_h , pad_w , device_top_data, divisor_override ,count_include_pad , use_divisor );

    // Copy output data to host.
    cudaDeviceSynchronize();
    cudaMemcpy(top_data, device_top_data, kDataLen * sizeof(int),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < kDataLen; ++i) {
      cout << "top_data[" << i << "] = " << top_data[i] << endl;
    }

    cudaDeviceReset();
    return 0;
}
