// ----------------------------------------------------------------
// Multi GPU Scheduler using CUDA
// ----------------------------------------------------------------

#ifndef MGPUS_CU
#define MGPUS_CU

#include <string.h>

#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <ctime>


namespace sched {
namespace mgpu {

/**
 * @file
 * example.cu
 *
 * @brief cuda example file for single GPU workload
 */

void releasehost(float * h_A, float * h_B, float * h_C, float * h_check)
{
    // release host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_check);
    printf("** Released CPU Memory Information.\n");
}

void releasegpu(float * A, float * B, float * C)
{
  // release device memory
  ERROR_CHECK( cudaFree(A));
  ERROR_CHECK( cudaFree(B));
  ERROR_CHECK( cudaFree(C));
  printf("** Released GPU Memory Information.\n");
}

__global__ void mgpuMultiplyAddOperator(int n, float * A, float * B, float * C)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
      C[index] = (A[index] + B[index]) * 2.0f;
    }
}

bool isCorrect(const int n, float * answer, float * attempt) {
  for(int i = 0; i < n; i++) {
    if (attempt[i] != answer[i]) return false;
  }
  return true;
}


void MultiGPUApplication(const int n)
{

  printf("** Multiple GPU Example with n = %d\n", n);
  printf("** Important Assumption for tests; numGPUs = 2\n");
  /* CPU data initializations */
  printf("** CPU Data Initializations -> Started\n");

  float * h_A = (float*) malloc(sizeof(float) * n);     // Host Array A
  float * h_B = (float*) malloc(sizeof(float) * n);     // Host Array B
  float * h_C = (float*) malloc(sizeof(float) * n);     // Host Array C

  float * h_check = (float*) malloc(sizeof(float) * n);   // Host Array check

  for (int i = 0; i < n; i++) {
    h_A[i] = (float) i;
    h_B[i] = (float) i;

    /*  Use to check if data is correct */
    h_check[i] = (float) (h_A[i] + h_B[i]) * 2.0f;
  }

  printf("** CPU Data Initializations -> Finished\n");

  /* Clock setup */
  std::clock_t start;
  float elapsedTime;
  start = std::clock();


  /* Memory Allocations */
  printf("** GPUs Data Initializations -> Started\n");

  float *A[2];
  float *B[2];
  float *C[2];
  const int Ns[2] = {n/2, n-(n/2)};

  // allocate the memory on the GPUs
  for(int dev=0; dev<2; dev++) {
      cudaSetDevice(dev);
      ERROR_CHECK( cudaMalloc((void**) &A[dev], Ns[dev] * sizeof(float)));
      ERROR_CHECK( cudaMalloc((void**) &B[dev], Ns[dev] * sizeof(float)));
      ERROR_CHECK( cudaMalloc((void**) &C[dev], Ns[dev] * sizeof(float)));
  }

  // Initialize device arrays with host data
  for(int dev=0,pos=0; dev<2; pos+=Ns[dev], dev++) {
      cudaSetDevice(dev);
      ERROR_CHECK( cudaMemcpyAsync( A[dev], h_A+pos, Ns[dev] * sizeof(float),
                                    cudaMemcpyHostToDevice));
      ERROR_CHECK( cudaMemcpyAsync( B[dev], h_B+pos, Ns[dev] * sizeof(float),
                                    cudaMemcpyHostToDevice));
  }

  printf("** GPUs Data Initializations -> Finished\n");

  /* Set Kernel Parameters */
  printf("** Kernel Multiply-Add Op -> Started\n");
  const int threadsPerBlock = 1024;
  const int blocksPerGrid = ((n + threadsPerBlock - 1) / threadsPerBlock);
  const int bytes = 0;

  const int num_streams = 2;
  cudaStream_t streams[num_streams];

  dim3 blocks  (threadsPerBlock, 1, 1);
  dim3 grid   (blocksPerGrid, 1, 1);

  for(int dev=0; dev<2; dev++) {
    cudaSetDevice(dev);
    cudaStreamCreate(&streams[dev]);
    mgpuMultiplyAddOperator<<<grid, blocks, bytes, streams[dev]>>>(Ns[dev], A[dev], B[dev], C[dev]);
    printf("** Current GPU Set Device = %d\n", dev);
  }
  ERROR_CHECK( cudaPeekAtLastError() );
  ERROR_CHECK( cudaDeviceSynchronize() );

  for(int dev=0,pos=0; dev<2; pos+=Ns[dev], dev++) {
      cudaSetDevice(dev);
      ERROR_CHECK( cudaMemcpy( h_C+pos, C[dev], Ns[dev] * sizeof(float),
                                    cudaMemcpyDeviceToHost));
  }
  printf("** Kernel Multiply-Add Op -> Finished\n");

  elapsedTime = ( std::clock() - start ) / (float) CLOCKS_PER_SEC;

  printf("********************************************\n");
  printf("** Elapsed Time (Init + Exec) = %f (ms)\n", elapsedTime);

  bool Validity = isCorrect(n, h_check, h_C);
  if (Validity) printf("** Solution is valid.\n");
  else printf("** Solution is valid.\n");
  printf("********************************************\n");

  /* Free Memory Allocations */
  releasehost(h_A, h_B, h_C, h_check);

  for(int gpu = 0; gpu < 2; gpu++)
    releasegpu(A[gpu], B[gpu], C[gpu]);

  return;
}

} // namespace: mgpu
} // namespace: sched

#endif // MGPUS_CU
