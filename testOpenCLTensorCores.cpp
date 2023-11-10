#include <stdlib.h>
#include <iostream>
#include <string>
#include <cstring>
#include <CL/cl.h>
using namespace std;


#define testpp(...)#__VA_ARGS__
const char* pp = testpp(
  __kernel void mma_8x8x4_fp16_acc_fp32(__global float *out) {
    float c[8] = {0., 0., 0., 0., 0., 0., 0., 0.};
    float d[8] = {0., 0., 0., 0., 0., 0., 0., 0.};
    unsigned short a[4] = {15360, 15360, 15360, 15360};
    unsigned short b[4] = {15360, 15360, 15360, 15360};
    const unsigned *A = (const unsigned *)&a;
    const unsigned *B = (const unsigned *)&b;
    const float *C = (const float *)&c;
    float *D = (float *)&d;
    asm(
      "mma.sync.aligned.row.col.m8n8k4.f32.f16.f16.f32"
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
      :
        "r"(A[0]), "r"(A[1]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7])
    );
    int tidx = get_global_id(0);
    for (int i = 0; i < 8; i++) out[tidx*8+i] = D[i];
  }

);
float out[256];

int main()
{
        cl_int err = 0;
        cl_uint numPlatforms;
        cl_platform_id platform = NULL;
        int ans;
        err = clGetPlatformIDs(0, NULL, &numPlatforms);
        if (err != CL_SUCCESS)
        {
                cout << "Error: Getting Platforms\n";
                return EXIT_FAILURE;
        }

        cout<< " numPlatforms:"  << numPlatforms <<endl;
        if (numPlatforms > 0)
        {
                cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
                err = clGetPlatformIDs(numPlatforms, platforms, NULL);
                if (err != CL_SUCCESS)
                {
                        cout << "Error: Getting Platform Ids.(clGetPlatformIDs)\n";
                        return EXIT_FAILURE;
                }
                cout << "available platforms: " << " ";
                for (unsigned int i = 0; i < numPlatforms; ++i)
                {
                        char pbuff[100];
                        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuff), pbuff, NULL);
                        cout << i << ":" << pbuff << "\n";
                }
                cout << "select platform: ";
                cin >> ans;
                if (ans < numPlatforms) platform = platforms[ans];
                else {cout << "invalid platform choice" << endl;  return EXIT_FAILURE;}
                free(platforms);
        }
        else
        {
                cout << "no platforms found\n";
                return EXIT_FAILURE;
        }
        cl_uint num_devices = 0;
        cl_device_id* devices = NULL;
        cl_device_id device = NULL;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, devices, &num_devices);
        if (num_devices == 0) //no GPU available.
        {
                cout << "No GPU device available." << endl;
                return EXIT_FAILURE;
        }
        else
        {
                devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
                err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
                cout << "available devices: " << " ";
                for (unsigned int i = 0; i < num_devices; ++i)
                {
                        char pbuff[100];
                        err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(pbuff), pbuff, NULL);
                        cout << i << ":" << pbuff << "\n";
                }
                cout << "select device: ";
                cin >> ans;
                if (ans < num_devices) device = devices[ans];
                else {cout << "invalid device choice" << endl; return EXIT_FAILURE;}
                free(devices);

        }
        cl_context context = nullptr;
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        cl_command_queue commandQueue = nullptr;
        commandQueue = clCreateCommandQueue(context, device, 0, &err);
        size_t ppsize[] = { strlen(pp) };
        cl_program pprog = clCreateProgramWithSource(context, 1, &pp, ppsize, &err);
        if (err != CL_SUCCESS)
        {
                cout << "Error: Loading Binary into cl_program (clCreateProgramWithBinary)\n";
                return EXIT_FAILURE;
        }
        err = clBuildProgram(pprog, 1, &device, NULL, NULL, NULL);
        if (err != CL_SUCCESS)
        {
                if (err == CL_BUILD_PROGRAM_FAILURE) {
                        // Determine the size of the log
                        size_t log_size;
                        clGetProgramBuildInfo(pprog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

                        // Allocate memory for the log
                        char* log = (char*)malloc(log_size);

                        // Get the log
                        clGetProgramBuildInfo(pprog, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                        // Print the log
                        cout << log << endl;
                }
                cout << err;
                printf("Error: Building Program (clBuildProgram)\n");

                return EXIT_FAILURE;
        }

        cl_kernel testkernel = clCreateKernel(pprog, "mma_8x8x4_fp16_acc_fp32", &err);
        if (err != CL_SUCCESS)
        {
                size_t log_size;
                clGetProgramBuildInfo(pprog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

                // Allocate memory for the log
                char* log = (char*)malloc(log_size);

                // Get the log
                clGetProgramBuildInfo(pprog, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                // Print the log
                cout << log << endl;
                cout << "Error: Creating Kernel from program.(clCreateKernel)\n";
                return EXIT_FAILURE;
        }
        cl_mem outbuff = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * 256, &out, &err);
        err = clSetKernelArg(testkernel, 0, sizeof(cl_mem), (void*)&outbuff);
        size_t globalThreads = 32;
        size_t localThreads = 32;
        err = clEnqueueNDRangeKernel(commandQueue, testkernel, 1, NULL, &globalThreads, &localThreads, 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
                cout << "Error: Enqueueing kernel\n";
                return EXIT_FAILURE;
        }
        err = clFinish(commandQueue);
        if (err != CL_SUCCESS)
        {
                cout << "Error: Finish command queue\n";
                return EXIT_FAILURE;
        }
        err = clEnqueueReadBuffer(commandQueue, outbuff, CL_TRUE, 0, sizeof(float) * 256, &out, 0, NULL, NULL);
        for (int i = 0; i < 16; i++){
          for (int j = 0; j < 16; j++)
          {
                cout << out[i*16+j] << " ";
          }
          cout << endl;
        }
}
