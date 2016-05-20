// ----------------------------------------------------------------
// Multi GPU Scheduler using CUDA
// ----------------------------------------------------------------

#ifndef SINGLEGPU_CU
#define SINGLEGPU_CU

#include <string.h>

#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

namespace sched {
namespace sgpu {

/**
 * @file
 * example.cu
 *
 * @brief cuda example file for single GPU workload
 */

 bool isCorrect(const int n, float * answer, float * attempt) {
   for(int i = 0; i < n; i++) {
     if (attempt[i] != answer[i]) return false;
   }
   return true;
 }

void release(float * h_A, float * h_B, float * h_C, float * h_check,
             float * A, float * B, float * C)
{
    // release host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_check);

    // release device memory
    ERROR_CHECK( cudaFree(A));
    ERROR_CHECK( cudaFree(B));
    ERROR_CHECK( cudaFree(C));
}

__global__ void MultiplyAddOperator(int n, float * A, float * B, float * C)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n) {
      C[index] = (A[index] + B[index]) * 2.0f;
    }
}


void SingleGPUApplication(const int n)
{
  cudaSetDevice(0);
  printf("** Single GPU Example with n = %d\n", n);
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

  /* CUDA event setup */
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* CUDA event start */
  float elapsedTime;
  cudaEventRecord(start, 0);

  /* Memory Allocations */
  printf("** GPU Data Initializations -> Started\n");
  int memsize = sizeof(float) * n;

  float * A;
  float * B;
  float * C;
  ERROR_CHECK( cudaMalloc((void**) &A, memsize));
  ERROR_CHECK( cudaMalloc((void**) &B, memsize));
  ERROR_CHECK( cudaMalloc((void**) &C, memsize));

  // Initialize device arrays with host data
  ERROR_CHECK( cudaMemcpy(A, h_A, memsize, cudaMemcpyHostToDevice));
  ERROR_CHECK( cudaMemcpy(B, h_B, memsize, cudaMemcpyHostToDevice));
  printf("** GPU Data Initializations -> Finished\n");

  /* Set Kernel Parameters */
  printf("** Kernel Multiply-Add Op -> Started\n");
  int threadsPerBlock = 1024;
  int blocksPerGrid = ((n + threadsPerBlock - 1) / threadsPerBlock);

  dim3 blocks  (threadsPerBlock, 1, 1);
  dim3 grid   (blocksPerGrid, 1, 1);

  MultiplyAddOperator<<<grid, blocks>>>(n, A, B, C);

  ERROR_CHECK( cudaMemcpy(h_C, C, memsize, cudaMemcpyDeviceToHost));
  printf("** Kernel Multiply-Add Op -> Finished\n");

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);

  /* Destroy CUDA event */
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("********************************************\n");
  printf("** Elapsed Time (Init + Exec) = %f (ms)\n", elapsedTime);

  bool Validity = isCorrect(n, h_check, h_C);
  if (Validity) printf("** Solution is valid.\n");
  else printf("** Solution is valid.\n");
  printf("********************************************\n");

  /* Free Memory Allocations */
  release(h_A, h_B, h_C, h_check, A, B, C);

  return;
}

} // namespace: sgpu
} // namespace: sched

#endif // SINGLEGPU_CU
