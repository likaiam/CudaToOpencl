#include <CL/cl.h>
#include <stdio.h>

#define N 128
// 用于读取SPIR-V文件的函数
unsigned char* readSPIR(const char* filename, size_t* size) {
    // 打开SPIR-V文件
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open SPIR-V file");
        return NULL;
    }
    
    printf("open file success!\n");
    // 获取文件大小
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);

    // 分配内存并读取文件内容
    unsigned char* data = (unsigned char*)malloc(*size);
    if (!data) {
        fclose(file);
        perror("Failed to allocate memory for SPIR-V data");
        return NULL;
    }

    fread(data, 1, *size, file);
    fclose(file);
    return data;
}

int main() {
    // 初始化OpenCL环境
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // 读取SPIR-V内核代码
    size_t spirSize;
    unsigned char* spirCode = readSPIR("output_shader.spv", &spirSize);
    if (!spirCode) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
    
    printf("spirSize:%d\n",spirSize);
    // 使用clCreateProgramWithIL创建程序
    cl_int err;
    cl_program program = clCreateProgramWithIL(context, spirCode, spirSize, &err);
    if (err != CL_SUCCESS) {
        perror("Failed to create program with IL");
        free(spirCode);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    // 编译程序
    cl_int buildErr = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (buildErr != CL_SUCCESS) {
        char buildLog[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        printf("Build error: %s\n", buildLog);

        clReleaseProgram(program);
        free(spirCode);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }
 
    // 创建内核
    cl_kernel kernel = clCreateKernel(program, "_Z20convolution_2D_basicPfS_S_ii", &err);
    if (err != CL_SUCCESS) {
        perror("Failed to create kernel");
        clReleaseProgram(program);
        free(spirCode);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 1;
    }

    // 分配主机内存并初始化数据
      float *host_a =(float *)malloc(sizeof(float) * 1024*1024);
      float *host_b = (float *)malloc(sizeof(float) * 1026*1026);
      float *host_c = (float *)malloc(sizeof(float) * 9);
      int d = 3;
      int e =1024;
  
      for (int i = 0; i < 1024*1024; i++) {
          host_a[i] = 1.0;
      }
     
     for (int i = 0; i < 1026*1026; i++) {
            host_b[i] = 2.0;
        }
        for (int i = 0; i < 9; i++) {
            host_c[i] = 3.0;
        }
      // 创建OpenCL缓冲区
      
      cl_mem buffer_a = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * 1024*1024, host_a, &err);
      cl_mem buffer_b = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * 1026*1026, host_b, &err);
      cl_mem buffer_c = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * 9, host_c, &err);
  
      // 设置内核参数
      err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_a);
      err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_b);
      err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_c);
      err |= clSetKernelArg(kernel, 3, sizeof(int), &d);
      err |= clSetKernelArg(kernel, 4, sizeof(int), &e);
  
      if (err != CL_SUCCESS) {
          printf("Error setting kernel arguments: %d\n", err);
          return 1;
      }
  
      // 执行内核

      size_t globalSize[3] = {1024,1024,1};
      size_t localSize[3] = {64,64,1};
      //size_t globalSize = 128*128;
      //size_t localSize = 128;
      err = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, &globalSize, &localSize, 0, NULL, NULL);
      if (err != CL_SUCCESS) {
          printf("Error enqueueing kernel: %d\n", err);
          return 1;
      }
 
     // 等待内核完成
     clFinish(queue);
 
     // 从设备读取结果
     err = clEnqueueReadBuffer(queue, buffer_b, CL_TRUE, 0, sizeof(int) * 1026*1026, host_b, 0, NULL, NULL);
     if (err != CL_SUCCESS) {
         printf("Error reading data from device: %d\n", err);
         return 1;
     }
     
     // 打印结果
     printf("Result: ");
     for (int i = 0; i < 128; i++) {
        printf("host_b: %f\n",host_b[i]);
     }
    
     printf("\n");


    // 清理资源
    free(spirCode);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

