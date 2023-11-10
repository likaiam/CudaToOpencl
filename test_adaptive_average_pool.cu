#include "cuda_runtime.h"
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>

#define START_IND(a,b,c) ((int64_t)((a / b) * c + ((a % b) * c) / b))
#define END_IND(a,b,c) (1 + ((int64_t)(a + 1) * c - 1) / b)
using namespace std;

__global__ void adaptive_average_pool(int *input, int *output,
                          int isizeH, int isizeW,
                          int osizeH, int osizeW,
                          int64_t istrideD, int64_t istrideH, int64_t istrideW)
{
   
    int oh, ow;

    // select input/output plane based on thread/block ID
    int o_plane = blockIdx.x;
    int i_plane = o_plane;

    output = output + o_plane*osizeH*osizeW;
    input = input + i_plane*istrideD;

    int ostartH = blockDim.y*blockIdx.y + threadIdx.y;
    int oendH = osizeH;
    const int ostepH = blockDim.y*gridDim.y;

    int ostartW = threadIdx.x;
    int oendW = osizeW;
    const int ostepW = blockDim.x;

    for(oh = ostartH; oh < oendH; oh += ostepH) {

      int istartH = START_IND(oh, osizeH, isizeH);
      int iendH   = END_IND(oh, osizeH, isizeH);
      int kH = iendH - istartH;

      for(ow = ostartW; ow < oendW; ow += ostepW) {

        int istartW = START_IND(ow, osizeW, isizeW);
        int iendW   = END_IND(ow, osizeW, isizeW);
        int kW = iendW - istartW;

        // Compute the average pooling over corresponding input pixels
        int *ptr_input = input + istartH*istrideH + istartW*istrideW;
        int *ptr_output = output + oh*osizeW + ow;
        int sum = 0;
        int ih, iw;
        for(ih = 0; ih < kH; ++ih) {
          for(iw = 0; iw < kW; ++iw) {
            int val = ptr_input[iw*istrideW];
            sum += val;
          }
          ptr_input += istrideH; // next input line
        }
        // Update output
        *ptr_output = sum / kH / kW;
      }
    }
}

int main()
{
    const int kDataLen = 4;

   int input[kDataLen] = {1, 2, 3, 4};
   int output[kDataLen] = {1, 2, 3, 4};
   int isizeH =10;  int isizeW =10;
   int osizeH =100; int osizeW=100;
  int64_t istrideD =10; int64_t istrideH =1; int64_t istrideW=100;
    
   int* device_input;
   int* device_output;
  cudaMalloc(&device_input, kDataLen * sizeof(int));
  cudaMalloc(&device_output, kDataLen * sizeof(int));
  
  cudaMemcpy(device_input, input, kDataLen * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_output, output, kDataLen * sizeof(int), cudaMemcpyHostToDevice);
  
    // Launch the kernel.
   adaptive_average_pool<<<1, kDataLen>>>(device_input, device_output, isizeH ,isizeW , osizeH ,osizeW ,istrideD ,istrideH ,istrideW);
 
    // Copy output data to host.
    cudaDeviceSynchronize();
    cudaMemcpy(output, device_output, kDataLen * sizeof(int),
               cudaMemcpyDeviceToHost);
 
    for (int i = 0; i < kDataLen; ++i) {
      cout << "output[" << i << "] = " << output[i] << endl;
    }
     
    cudaDeviceReset();
    return 0;
}


