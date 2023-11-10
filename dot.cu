#include "cuda_runtime.h"
#include <iostream>
#include <stdio.h>

#define N 65536
const int threadsPerBlock = 128;
const int blocksPerGrid = 128;

__global__ void k_dot (int* a, int *b, int *c){
    __shared__ float cache[128];
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int cacheIndex=threadIdx.x;
    float temp = 0;
    while (tid<65536){
        temp += a[tid]*b[tid];
        tid += blockDim.x*gridDim.x;
    }
    cache[cacheIndex] = temp;
    __syncthreads();
    int i = blockDim.x/2;
    while(i>0){
        if(cacheIndex<i)
            cache[cacheIndex] += cache[cacheIndex+i];
        __syncthreads();
        i /= 2;
    }
    if (cacheIndex ==0){
        c[blockIdx.x] = cache[0];
    }
}

int main(void){
int a[N], b[N], c, partial_c[blocksPerGrid];
    int *d_a, *d_b, *d_partial_c;
    cudaMalloc((void**)&d_a,N*sizeof(int));
    cudaMalloc((void**)&d_b,N*sizeof(int));
    cudaMalloc((void**)&d_partial_c,128*sizeof(int));
    for( int i=0;i<N;i++)
    {
        a[i]=1;
        b[i]=2;
    }
    cudaMemcpy(d_a,a,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,N*sizeof(int),cudaMemcpyHostToDevice);
    k_dot<<<128,128>>>(d_a,d_b,d_partial_c);
    cudaMemcpy(partial_c,d_partial_c,128*sizeof(int),cudaMemcpyDeviceToHost);
    /*
    * 将所有线程块的点积结果相加即为最终结果
    */
    c=0;
    for(int i=0;i<128;i++)
    {
        c += partial_c[i];
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial_c);
    /*
    * 检查结果是否正确
    */
    if(c==2*N) printf("success\n");
    else printf("fail\n");
    return 0;
}

