// ----------------------------------------------------------------
// Multi GPU Scheduler using CUDA
// ----------------------------------------------------------------

#ifndef MTX_CU
#define MTX_CU

#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>

namespace sched {
namespace mtx {

#define BLOCK_WIDTH 8

__global__ void GPUMatrixMultiply(const int WIDTH, float * A, float * B, float * C)
{
    __shared__ float sh_A [BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float sh_B [BLOCK_WIDTH][BLOCK_WIDTH];

    unsigned int col = BLOCK_WIDTH*blockIdx.x + threadIdx.x;
    unsigned int row = BLOCK_WIDTH*blockIdx.y + threadIdx.y;

    #pragma unroll
    for (int m = 0; m < WIDTH/BLOCK_WIDTH; m++)
    {
        sh_A[threadIdx.y][threadIdx.x] = A[row*WIDTH + (m*BLOCK_WIDTH + threadIdx.x)];
        sh_B[threadIdx.y][threadIdx.x] = B[(m*BLOCK_WIDTH + threadIdx.y) * WIDTH + col];
     __syncthreads();

      for (int k = 0; k < BLOCK_WIDTH; k++) {
        C[row*WIDTH + col]+= sh_A[threadIdx.x][k] * sh_B[k][threadIdx.y];

      }
     __syncthreads();
    }
}

void MTXMultiplyApplication(const int n)
{

  // Malloc n * n memory on host to store the matrix
  float m_hA[n][n], m_hB[n][n], m_hC[n][n], m_hCheckC[n][n];

  /*float ** m_hA = (float**)malloc(sizeof(float) * n);
  float ** m_hB = (float**)malloc(sizeof(float) * n);
  float ** m_hC = (float**)malloc(sizeof(float) * n);
  float ** m_hCheckC = (float**)malloc(sizeof(float) * n);
  printf("Data on Host malloc-ed.");

  for(int i = 0; i < n; i++) {
    m_hA[i]= (float*) malloc(n * sizeof(float));
    m_hB[i]= (float*) malloc(n * sizeof(float));
    m_hC[i]= (float*) malloc(n * sizeof(float));
    m_hCheckC[i]= (float*) malloc(n * sizeof(float));
  }*/

  int memsize = sizeof(float) * n * n;

  float * A;
  float * B;
  float * C;
  ERROR_CHECK( cudaMalloc((void**) &A, memsize));
  ERROR_CHECK( cudaMalloc((void**) &B, memsize));
  ERROR_CHECK( cudaMalloc((void**) &C, memsize));

  printf("Data on Host & dev malloced.\n");


  // float invRandMax = 1000.0f / RAND_MAX; // Produces random numbers between 0 and 1000
  for (int l = 0; l < n; ++l)
  {
    for (int m = 0; m < n; m++) {
      /* code */
      m_hA[l][m] = 1.0f;
      m_hB[l][m] = 2.0f;

      m_hCheckC[l][m] = 0.0f; // Init to 0
    }
  }

  // Init to 0 on device
  ERROR_CHECK( cudaMemcpy(C, m_hCheckC, memsize, cudaMemcpyHostToDevice));

  for (int x = 0; x < n; x++) {
    for (int y = 0; y < n; y++) {
      for (int z = 0; z < n; z++) {
          m_hCheckC[x][y] += m_hA[x][z] * m_hB[z][y];
      }
    }
  }


  // Initialize device arrays with host data
  ERROR_CHECK( cudaMemcpy(A, m_hA, memsize, cudaMemcpyHostToDevice));
  ERROR_CHECK( cudaMemcpy(B, m_hB, memsize, cudaMemcpyHostToDevice));

  //const int bytes(0);
  dim3 blocks  (BLOCK_WIDTH, BLOCK_WIDTH, 1);
  dim3 grid   (n/BLOCK_WIDTH, n/BLOCK_WIDTH, 1);

  GPUMatrixMultiply<<<grid, blocks>>>(n, A, B, C);
  ERROR_CHECK(cudaPeekAtLastError());

  ERROR_CHECK( cudaMemcpy(m_hC, C, memsize, cudaMemcpyDeviceToHost));
  printf("Memory copied back from device.\n");

  bool correct(true);
  for (int k = 0; k < n; ++k)
  {
    for (int m = 0; m < n; m++) {
      correct = correct && (m_hC[k][m] == m_hCheckC[k][m]);
    }
  }

  printf("Was it correct? %d\n", (int)correct);

  if (!correct) {
    for (int i = 0 ; i<n ; i++ )
    {
        for (int j = 0 ; j < n ; j++ )
        {
            printf ("%f && %f -", m_hC[i][j], m_hCheckC[i][j]) ;
        }
        printf ("\n") ;
    }
  }

  return;


}


}
}

#endif
