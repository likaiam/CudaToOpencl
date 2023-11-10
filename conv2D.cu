#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include"cuda_runtime.h"
 
#define FILTER_WIDTH 3
int filter_size = FILTER_WIDTH;
int arr_size = 1024;
int result_size = arr_size + FILTER_WIDTH - 1;
 
__global__ void convolution_2D_basic(float* filter, float* arr, float* result, int filter_size, int arr_size)
{
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	if (Row < 1024 - 3 + 1 && Col < 1024 - 3 + 1)
	{
		float pixVal = 0;
		int startCol = Col;
		int startRow = Row;	
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				int curRow = startRow + i;
				int curCol = startCol + j;
				if (curRow > -1 && curRow<1024&&curCol>-1 && curCol < 1024)
				{
					pixVal += filter[i*3 + j] * arr[curRow*1024 + curCol];
				}
			}
		}
		result[Row*1024 + Col] = pixVal;
	}
}

void Conv2Kernel(float** arr, float** pFilter, int filter_size, int arr_size, int result_size)
{
	int arr_size_1D = arr_size * arr_size;
	int filter_size_1D = filter_size * filter_size;
	int result_size_1D = result_size * result_size;

	float *arr_1D = (float*)malloc(arr_size_1D * sizeof(float));
	float *result_1D = (float*)malloc(result_size_1D * sizeof(float));
	float *filter_1D = (float*)malloc(filter_size_1D * sizeof(float));

	for (int i = 0; i < arr_size; i++) {
		for (int j = 0; j < arr_size; j++) {
			arr_1D[i*arr_size + j] = arr[i][j] * 1.0;
		}
	}

	for (int i = 0; i < filter_size; i++) {
		for (int j = 0; j < filter_size; j++) {
			filter_1D[i*filter_size + j] = pFilter[i][j] * 1.0;
		}
	}

	float *device_input_arr, *device_output_arr, *device_filter_arr;
	cudaMalloc((void**)&device_input_arr, sizeof(float) * arr_size_1D);
	cudaMalloc((void**)&device_output_arr, sizeof(float) * result_size_1D);
	cudaMalloc((void**)&device_filter_arr, sizeof(float) * filter_size_1D);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(device_input_arr, arr_1D, sizeof(float) * arr_size_1D, cudaMemcpyHostToDevice);
	cudaMemcpy(device_output_arr, result_1D, sizeof(float) * result_size_1D, cudaMemcpyHostToDevice);
	cudaMemcpy(device_filter_arr, filter_1D, sizeof(float) * filter_size_1D, cudaMemcpyHostToDevice);

	dim3 ThreadNum = (64, 64);
	dim3 BlockNum  = ((arr_size - 0.5) / ThreadNum.x + 1, (arr_size - 0.5) / ThreadNum.x + 1, 1);

	cudaEventRecord(start, 0);
	convolution_2D_basic <<<BlockNum, ThreadNum >>> (device_input_arr, device_output_arr, device_filter_arr, filter_size, arr_size);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy(result_1D, device_output_arr, sizeof(float)*arr_size_1D, cudaMemcpyDeviceToHost);

	float GPU_time;
	cudaEventElapsedTime(&GPU_time, start, stop);

	printf("-------------------GPU version Done!------------------\n");
	printf("GPU_Time: %f \n", GPU_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(device_input_arr);
	cudaFree(device_output_arr);
	cudaFree(device_filter_arr);
}
 
void Conv2(float** filter, float** arr, float** result, int filter_size, int arr_size) {
	float temp;
 
	for (int i = 0; i < arr_size - filter_size + 1; i++) {
		for (int j = 0; j < arr_size - filter_size + 1; j++) {
			temp = 0;
			for (int m = 0; m < filter_size; m++) {
				for (int n = 0; n < filter_size; n++) {
					temp += filter[m][n] * arr[i + m][j + n];
				}
			}
			result[i][j] = temp;
		}
	}
}
 
int main()
{
	int dev = 0;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, dev);
	std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
	std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
	std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
	std::cout << "每个Grid的Block数：" << devProp.maxGridSize[0] << " x " << devProp.maxGridSize[1] << " x " << devProp.maxGridSize[2] << std::endl;
	std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
	std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
 
 
	clock_t CPU_start, CPU_stop;
 
	// Array, filter, result
	float** pFilter = new float*[filter_size];
	for (int i = 0; i < filter_size; i++)
	{
		pFilter[i] = new float[filter_size];
	}
 
	float** arr = new float*[arr_size];
	for (int i = 0; i < arr_size; i++)
	{
		arr[i] = new float[arr_size];
	}
 
	float** res = new float*[result_size];
	for (int i = 0; i < result_size; i++)
	{
		res[i] = new float[result_size];
	}
 
	//initialization
	for (int i = 0; i < filter_size; i++) {
		for (int j = 0; j < filter_size; j++)
			pFilter[i][j] = rand() % 11;
	}
 
	for (int i = 0; i < arr_size; i++) {
		for (int j = 0; j < arr_size; j++)
			arr[i][j] = rand() % 11;
	}
 
	CPU_start = clock();
	Conv2(pFilter, arr, res, filter_size, arr_size);
	CPU_stop = clock();
	float CPU_time = (float)(CPU_stop - CPU_start) / CLOCKS_PER_SEC;
	printf("-------------------CPU version Done!------------------\n");
	printf("CPU time:%f \n", CPU_time);
 
	Conv2Kernel(arr, pFilter, filter_size, arr_size, result_size);
 
}
