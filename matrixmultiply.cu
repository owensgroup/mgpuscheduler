#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <thread>
#include <fstream>
#include <algorithm>

#include "matrixmultiply.cuh"
#include "scheduler.cuh"

/**
* @brief Macro for error checking for all GPU calls
* @param[in] ans	The GPU call itself, which evaluates to the cudaError_t returned.
*/
#ifndef ERROR_CHECK
#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file,
  int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "Cuda error in file '%s' in line '%d': %s\n",
      file, line, cudaGetErrorString(code));
    if (abort) exit(code);
  }
}
#endif

__global__ void GPUMatrixMultiply(const int matSize, float * A, float * B, float * C)
{
    // NOTE: shared[] is used to hold both sh_A and sh_B.
    extern __shared__ float shared[];
    unsigned int blockWidth = blockDim.x;
    unsigned int sharedOffsetB = blockWidth*blockWidth;

    float *sh_A = &shared[0];
    float *sh_B = &shared[sharedOffsetB];

    unsigned int rowC = blockWidth * blockIdx.y + threadIdx.y;
    unsigned int colC = blockWidth * blockIdx.x + threadIdx.x;

    float temp = 0;
    unsigned int sharedRow = threadIdx.y;
    unsigned int sharedCol = threadIdx.x;
    unsigned int posShared = sharedRow * blockWidth + sharedCol;
    unsigned int posA, posB; // When they're separate, below

    #pragma unroll
    //for (int m = 0; m < (matSize-1)/blockWidth+1; m++)
    for (int block = 0; block < gridDim.x; ++block)
    {
      unsigned int blockOffset = block*blockWidth; // For A this is the column offset, for B, the row offset
      if (rowC < matSize && blockOffset + threadIdx.x < matSize)
        sh_A[posShared] = A[rowC*matSize + (blockOffset + threadIdx.x)];
      else sh_A[posShared] = 0.0f; // Not sure about this
      if (colC < matSize && blockOffset + threadIdx.y < matSize)
        sh_B[posShared] = B[(blockOffset + threadIdx.y) * matSize + colC];
      else sh_B[posShared] = 0.0f; // Not sure about this

     __syncthreads();

      for (int k = 0; k < blockWidth; k++) {
        posA = sharedRow * blockWidth + k;
        posB = k * blockWidth + sharedCol;
        temp += sh_A[posA] * sh_B[posB];
      }
     __syncthreads();
    }

    if (rowC < matSize && colC < matSize)
      C[rowC*matSize + colC] = temp;
}


void MatrixMultiply::FreeHostMemory()
{
  /* 2D Memory release
  for (int i = 0; i < m_vectorSize; i++) {
    if (m_hA[i]) free(m_hA[i]);
    if (m_hB[i]) free(m_hB[i]);
    if (m_hC[i]) free(m_hC[i]);
    if (m_hCheckC[i]) free(m_hCheckC[i]);
  }*/

  if (m_hA) ERROR_CHECK(cudaFreeHost(m_hA));
  if (m_hB) ERROR_CHECK(cudaFreeHost(m_hB));
  if (m_hC) ERROR_CHECK(cudaFreeHost(m_hC));
  if (m_hCheckC) free(m_hCheckC);
  m_hA = m_hB = m_hC = m_hCheckC = NULL;
}

void MatrixMultiply::FreeDeviceMemory()
{
  ERROR_CHECK(cudaSetDevice(m_deviceNum));
  if (m_dA) ERROR_CHECK(cudaFree(m_dA));
  if (m_dB) ERROR_CHECK(cudaFree(m_dB));
  if (m_dC) ERROR_CHECK(cudaFree(m_dC));
  m_dA = m_dB = m_dC = NULL;
}

/**
* @brief Initialize host vectors for a single MatrixMultiply run.
* @param[in] vectorSize	The size of each vector.
*/
void MatrixMultiply::InitializeData(int matrixSize, int blockWidth, int kernelNum)
{
  m_matrixSize = matrixSize;
  m_blockWidth = blockWidth;
  m_kernelNum = kernelNum;

  //m_hA = (float*) malloc(sizeof(float) * matrixSize * matrixSize);
  //m_hB = (float*) malloc(sizeof(float) * matrixSize * matrixSize);
  //m_hC = (float*) malloc(sizeof(float) * matrixSize * matrixSize);

  ERROR_CHECK(cudaHostAlloc(&m_hA, sizeof(float) * matrixSize * matrixSize, cudaHostAllocPortable));
  ERROR_CHECK(cudaHostAlloc(&m_hB, sizeof(float) * matrixSize * matrixSize, cudaHostAllocPortable));
  ERROR_CHECK(cudaHostAlloc(&m_hC, sizeof(float) * matrixSize * matrixSize, cudaHostAllocPortable));
  m_hCheckC = (float*) malloc(sizeof(float) * matrixSize * matrixSize);

  m_blocksRequired = matrixSize % blockWidth == 0 ? (matrixSize / blockWidth) : 1 + (matrixSize / blockWidth);
  m_globalMemRequired = 3 * sizeof(float) * matrixSize * matrixSize;

  // Floating pt ops (M=multiply, A=add):
  // Per thread (matrixSize * matrixSize):
  //  2M + 2A for row/col
  //  1M + 1A for posShared
  //  1M + 1A for C index -> TOT = 8
  //  Per block (gridDim.x or m_blocksRequired)
  //    1M for blockOffset
  //    2 * 1A for row check / col check
  //    2 * (1M + 2A) for A/B index -> TOT = 9
  //    Per block thread (blockWidth)
  //      1M + 1A for posA
  //      1M + 1A for posB
  //      1M + 1A for temp -> TOT = 6

  m_floatingPointOps = (float)(matrixSize * matrixSize * (8 + m_blocksRequired * (9 + blockWidth * 6)));

  // Float memory accesses (R=read, W=write)
  // Per thread (matrixSize * matrixSize):
  //  1W -> TOT = 1
  //  Per block (gridDim.x or m_blocksRequired)
  //    1R + 1W for A to sh_A
  //    1R + 1W for B to sh_B -> TOT = 4
  //    Per block thread (blockWidth)
  //      2R for temp -> TOT = 2

  // Remember to first multiply by 4 for floats
  m_memBytesReadWrite = 4.0f * (matrixSize * matrixSize * (1 + m_blocksRequired * (4 + blockWidth * 2)));

  //float invRandMax = 1000.0f / RAND_MAX; // Produces random numbers between 0 and 1000
  for (int i = 0; i < matrixSize*matrixSize; i++) {
    m_hA[i] = 2.0f; // std::rand() * invRandMax;
    m_hB[i] = 1.0f; // std::rand() * invRandMax;
    //m_hA[i] = std::rand() * invRandMax; // This doesn't work...
    //m_hB[i] = std::rand() * invRandMax;
    m_hCheckC[i] = 0.0f;
  }
/*
  if (Scheduler::m_verbose)
  {
    for (int x = 0; x < matrixSize; x++) { // row number of output
      for (int y = 0; y < matrixSize; y++) { // column number of output
        for (int z = 0; z < matrixSize; z++) { // four elements are added for this output
          //m_hCheckC[matrixSize*x + y] += m_hA[matrixSize*x + z] * m_hB[matrixSize*z + y];
        }
      }
    }
  }
  return;*/
}

/**
* @brief Find a device with enough resources, and if available, decrement the available resources and return the id.
*/
int MatrixMultiply::AcquireDeviceResources(std::vector< DeviceInfo > *deviceInfo)
{
  // Once we acquire a lock we need to keep it. So LOCK OUTSIDE this method, not inside
  int deviceNum, freeDeviceNum = -1;
  for (deviceNum = 0; deviceNum < (int)deviceInfo->size(); ++deviceNum)
  {
    DeviceInfo &device = deviceInfo->operator[](deviceNum);
    if (m_globalMemRequired < device.m_remainingGlobalMem && m_blocksRequired < device.m_remainingBlocksDimX)
    {
      if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << " acquired GPU " << deviceNum << " **\n";
      if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << ":  Prev Memory: " << device.m_remainingGlobalMem / (1024.0*1024.0) << "MB, Blocks: " << device.m_remainingBlocksDimX << "\n";

      freeDeviceNum = deviceNum;
      device.m_remainingGlobalMem -= m_globalMemRequired;
      device.m_remainingBlocksDimX -= m_blocksRequired;

      if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << ": - Used Memory: " << m_globalMemRequired/(1024.0*1024.0) << "MB, Blocks: " << m_blocksRequired << "\n";
      if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << ": =  New Memory: " << device.m_remainingGlobalMem / (1024.0*1024.0) << "MB, Blocks: " << device.m_remainingBlocksDimX << "\n";
      if (Scheduler::m_verbose) std::flush(std::cout);

      break;
    }
    else
    {
      //if (Scheduler::m_verbose) std::cout << "\n**********************************************************************\n";
      //if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << " >> Unable to be acquired by device " << deviceNum << "\n";
      //if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << ": Kernel Memory: " << m_globalMemRequired / (1024.0*1024.0) << "MB, Blocks: " << m_blocksRequired << "\n";
      //if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << ": Device Memory: " << device.m_remainingGlobalMem / (1024.0*1024.0) << "MB, Blocks: " << device.m_remainingBlocksDimX << "\n";
      //if (Scheduler::m_verbose) std::cout << "**********************************************************************\n";
    }
  }

  return freeDeviceNum;
}

/**
* @brief Execution is complete, release the GPU resources for other threads.
*/
void MatrixMultiply::ReleaseDeviceResources(std::vector< DeviceInfo > *deviceInfo)
{
  // Lock this method
  std::lock_guard< std::mutex > guard(Scheduler::m_deviceInfoMutex); // Automatically unlocks when destroyed

  DeviceInfo &device = deviceInfo->operator[](m_deviceNum);

  if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << " released GPU " << m_deviceNum << " **\n";
  if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << ": Prev Memory: " << device.m_remainingGlobalMem / (1024.0*1024.0) << "MB, Blocks: " << device.m_remainingBlocksDimX << "\n";

  device.m_remainingGlobalMem += m_globalMemRequired;
  device.m_remainingBlocksDimX += m_blocksRequired;

  if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << ": + Used Memory: " << m_globalMemRequired / (1024.0*1024.0) << "MB, Blocks: " << m_blocksRequired << "\n";
  if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << ": =  New Memory: " << device.m_remainingGlobalMem / (1024.0*1024.0) << "MB, Blocks: " << device.m_remainingBlocksDimX << "\n";
  if (Scheduler::m_verbose) std::flush(std::cout);

  // Result is already in host memory, so free GPU memory
  FreeDeviceMemory();
}

/**
* @brief Execution is complete. Record completion event and timers, verify result, and free host memory.
*/
void MatrixMultiply::FinishHostExecution(bool freeHostMemory)
{
  // Update timers
  ERROR_CHECK(cudaEventElapsedTime(&m_kernelExecTimeMS, m_startExecEvent, m_finishExecEvent));
  ERROR_CHECK(cudaEventElapsedTime(&m_totalExecTimeMS, m_startCudaMallocEvent, m_finishDownloadEvent));

  // Compute MFLOP/s, MB/s for this kernel
  m_MFLOPs = m_floatingPointOps / ((2 ^ 20) * (1000 * m_kernelExecTimeMS));
  m_MBps = m_memBytesReadWrite / ((2 ^ 20) * (1000 * m_kernelExecTimeMS));

  // Verify the result
  bool correct(true);
  for (int m = 0; m < m_matrixSize*m_matrixSize; m++) {
    correct = correct && (ceil(m_hC[m]) == ceil(m_hCheckC[m]));
  }

  if (Scheduler::m_verbose) printf("Kernel %d >> Device: %d, Queue: %.3fms, Kernel: %.3fms, Total: %.3fms, MFLOP/s: %.2f, MB/s: %.2f, Correct: %s\n",
    m_kernelNum, m_deviceNum, m_queueTimeNS, m_kernelExecTimeMS, m_totalExecTimeMS, m_MFLOPs, m_MBps, correct ? "True" : "False");

  // Free memory
  if (freeHostMemory)
    FreeHostMemory();
}

/**
* @brief Generate data for the entire batch of MatrixMultiply's being run.
*/
void BatchMatrixMultiply::GenerateData()
{
  m_data.resize(m_batchSize);

  // Get a random generator with a normal distribution, mean = meanVectorSize, stdDev = 0.1*meanVectorSize
  std::normal_distribution< float > normalDist((float)m_meanMatrixSize, 0.1f*m_meanMatrixSize);

  // Seed by the batch size for both the std::rand generator and the std::default_random_engine, used by distribution
  std::srand(m_batchSize);
  std::default_random_engine randomGen(m_batchSize);

  if (Scheduler::m_verbose) std::cout << "** Generating data **\n\tBatch Size: " << m_batchSize << ", Matrix Size: "
    << m_meanMatrixSize << ", Block Width: " << m_blockWidth << "\n";

  for (int kernelNum = 0; kernelNum < m_batchSize; ++kernelNum)
  {
    m_data[kernelNum] = new MatrixMultiply;
    m_data[kernelNum]->InitializeData((int)normalDist(randomGen), m_blockWidth, kernelNum);
  }

  if (Scheduler::m_verbose) std::cout << "** Done generating data **\n\n";
}

void BatchMatrixMultiply::ComputeBatchResults()
{
  // Sum up the per-kernel floating point ops and mem bytes accessed
  m_batchFloatingPointOps = m_batchMemBytesReadWrite = 0;
  for (int kernel = 0; kernel < (int)m_data.size(); ++kernel)
  {
    m_batchFloatingPointOps += m_data[kernel]->m_floatingPointOps;
    m_batchMemBytesReadWrite += m_data[kernel]->m_memBytesReadWrite;
  }

  // Use queue times to find which kernel was run first, and which last.
  struct MatrixMultiplyComp
  {
    bool operator()(const MatrixMultiply *lhs, const MatrixMultiply *rhs)
    {
      return lhs->m_queueTimeNS < rhs->m_queueTimeNS;
    }
  };

  std::sort(m_data.begin(), m_data.end(), MatrixMultiplyComp());

  m_batchKernelExecTimeMS = m_batchTotalExecTimeMS = -1;
  m_batchGFLOPs = m_batchGBps = -1;
  if (m_data.size() < 2)
    return;

  const MatrixMultiply &firstKernel = **m_data.begin();
  const MatrixMultiply &lastKernel = **m_data.rbegin();
  ERROR_CHECK(cudaEventElapsedTime(&m_batchKernelExecTimeMS, firstKernel.m_startExecEvent, lastKernel.m_finishExecEvent));
  ERROR_CHECK(cudaEventElapsedTime(&m_batchTotalExecTimeMS, firstKernel.m_startCudaMallocEvent, lastKernel.m_finishDownloadEvent));

  // Compute GFLOP/s, GB/s for this batch
  m_batchGFLOPs = m_batchFloatingPointOps / ((2 ^ 30) * (1000 * m_batchKernelExecTimeMS));
  m_batchGBps = m_batchMemBytesReadWrite / ((2 ^ 30) * (1000 * m_batchKernelExecTimeMS));
}

void BatchMatrixMultiply::OutputResultsCSV(const std::string &kernelName)
{
  // First output data for each kernel
  std::string filenameKernel = kernelName + std::string("KernelResults.csv");

  // Append in case running from a script (without, file is overwritten)
  std::ofstream csvKernelFile;
  csvKernelFile.open(filenameKernel.c_str(), std::ios::app);

  // Only output header if file is empty
  csvKernelFile.seekp(0, std::ios_base::beg);
  std::size_t posFirst = csvKernelFile.tellp();
  csvKernelFile.seekp(0, std::ios_base::end);
  std::size_t posLast = csvKernelFile.tellp();
  if (posLast-posFirst == 0)
  {
    csvKernelFile << "BatchSize, KernelName, MeanMatrixSize, BlockWidth, MaxDevices, KernelNum, QueueTimeMS"
                  << ", KernelExecTimeMS, TotalExecTimeMS, FloatingPtOps, MemBytes, MFLOPs, MBps\n";
  }

  for (int kernelNum = 0; kernelNum < (int)m_data.size(); ++kernelNum)
  {
    const MatrixMultiply &kernel = *m_data[kernelNum];
    csvKernelFile << m_batchSize << ", " << kernelName.c_str() << ", " << m_meanMatrixSize << ", " << m_blockWidth
                  << ", " << Scheduler::m_maxDevices << ", " << kernel.m_kernelNum << ", " << kernel.m_queueTimeNS
                  << ", " << kernel.m_kernelExecTimeMS << ", " << kernel.m_totalExecTimeMS
                  << ", " << kernel.m_floatingPointOps << ", " << kernel.m_memBytesReadWrite
                  << ", " << kernel.m_MFLOPs << ", " << kernel.m_MBps << "\n";
  }

  // Second output data summary for this batch run
  std::string filenameBatch = kernelName + std::string("BatchResults.csv");

  // Append in case running from a script (without, file is overwritten)
  std::ofstream csvBatchFile;
  csvBatchFile.open(filenameBatch.c_str(), std::ios::app);

  // Only output header if file is empty
  csvBatchFile.seekp(0, std::ios_base::beg);
  posFirst = csvBatchFile.tellp();
  csvBatchFile.seekp(0, std::ios_base::end);
  posLast = csvBatchFile.tellp();
  if (posLast - posFirst == 0)
  {
    csvBatchFile << "BatchSize, KernelName, MeanMatrixSize, BlockWidth, MaxDevices, BatchKernelExecTimeMS"
                 << ", BatchTotalExecTimeMS, FloatingPtOps, MemBytes, GFLOPs, GBps\n";
  }

  csvBatchFile << m_batchSize << ", " << kernelName.c_str() << ", " << m_meanMatrixSize << ", " << m_blockWidth
               << ", " << Scheduler::m_maxDevices << ", " << m_batchKernelExecTimeMS
               << ", " << m_batchTotalExecTimeMS << ", " << m_batchFloatingPointOps
               << ", " << m_batchMemBytesReadWrite << ", " << m_batchGFLOPs << ", " << m_batchGBps << "\n";
}

// NVCC having trouble parsing the std::thread() call when this is a member function, so keeping it non-member friend
void RunKernelThreaded(BatchMatrixMultiply *batch, int kernelNum)
{
  MatrixMultiply &kernel = *(batch->m_data[kernelNum]);

  // Acquire a GPU
  int deviceNum = -1;
  bool firstAttempt = true;

  typedef std::chrono::high_resolution_clock clock;
  auto startQueue = clock::now();

  while (deviceNum < 0)
  {
    if (firstAttempt)
    {
      if (Scheduler::m_verbose) std::cout << "** Kernel " << kernelNum << " queued for next available GPU **\n";
      firstAttempt = false;
    }
    else
    {
      // Sleep to let others try and get resources, and let the GPUs move data back
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Try to acquire GPU resources (using a lock)
    {
      // Lock this block
      std::lock_guard< std::mutex > guard(Scheduler::m_deviceInfoMutex); // Automatically unlocks when destroyed
      if (deviceNum < 0)
        deviceNum = kernel.AcquireDeviceResources(&Scheduler::m_deviceInfo);

      // Failed to get the lock, continue next iteration to try again
      if (deviceNum < 0)
        continue;

      // Succeeded in getting the lock - queue up all the work before we release the lock
      // Get the time spent in the queue
      auto finishQueue = clock::now();
      std::chrono::duration< double > diff;
      kernel.m_queueTimeNS = (float)(1000 * diff.count());

      // Keep this locked until we queue up all the work
      if (Scheduler::m_verbose) std::cout << "** Kernel " << kernelNum << " acquired GPU " << deviceNum << " **\n";

      // Store the device number for use in ReleaseDeviceResources() - not strictly necessary, could be passed in
      kernel.m_deviceNum = deviceNum;

      // Set the device and create the stream and events
      ERROR_CHECK(cudaSetDevice(deviceNum));
      ERROR_CHECK(cudaStreamCreate(&kernel.m_stream));
      ERROR_CHECK(cudaEventCreate(&kernel.m_startExecEvent));
      ERROR_CHECK(cudaEventCreate(&kernel.m_finishExecEvent));
      ERROR_CHECK(cudaEventCreate(&kernel.m_startCudaMallocEvent));
      ERROR_CHECK(cudaEventCreate(&kernel.m_finishDownloadEvent));

      // Mark the start total execution event
      ERROR_CHECK(cudaEventRecord(kernel.m_startCudaMallocEvent, kernel.m_stream));

      // Allocate memory on the GPU for input and output data
      std::size_t vectorBytes(kernel.m_matrixSize * kernel.m_matrixSize * sizeof(float));
      ERROR_CHECK(cudaMalloc((void**)&kernel.m_dA, vectorBytes));
      ERROR_CHECK(cudaMalloc((void**)&kernel.m_dB, vectorBytes));
      ERROR_CHECK(cudaMalloc((void**)&kernel.m_dC, vectorBytes));

      // Upload the input data for this stream
      ERROR_CHECK(cudaMemcpyAsync(kernel.m_dA, kernel.m_hA, vectorBytes,
        cudaMemcpyHostToDevice, kernel.m_stream));
      ERROR_CHECK(cudaMemcpyAsync(kernel.m_dB, kernel.m_hB, vectorBytes,
        cudaMemcpyHostToDevice, kernel.m_stream));

      // Mark the start kernel execution event
      ERROR_CHECK(cudaEventRecord(kernel.m_startExecEvent, kernel.m_stream));

      // Run the kernel
      size_t sharedMemBytes = 2 * sizeof(float) * kernel.m_blockWidth * kernel.m_blockWidth;
      dim3 dimBlock(kernel.m_blockWidth, kernel.m_blockWidth, 1); // Same dims as other kernel
      dim3 dimGrid(kernel.m_blocksRequired, kernel.m_blocksRequired, 1);
      //std::cout << "Grid: " << kernel.m_blocksRequired << ", Block: " << kernel.m_blockWidth << ", Shared: " << sharedMemBytes << "\n";
      GPUMatrixMultiply << < dimGrid, dimBlock, sharedMemBytes, kernel.m_stream >> >(kernel.m_matrixSize, kernel.m_dA, kernel.m_dB, kernel.m_dC);
      ERROR_CHECK(cudaPeekAtLastError());

      // Record the time (since stream is non-zero, waits for stream to be complete)
      ERROR_CHECK(cudaEventRecord(kernel.m_finishExecEvent, kernel.m_stream));

      // Download the output data for this stream
      ERROR_CHECK(cudaMemcpyAsync(kernel.m_hC, kernel.m_dC, vectorBytes,
        cudaMemcpyDeviceToHost, kernel.m_stream));

      // Mark the end of total execution event
      ERROR_CHECK(cudaEventRecord(kernel.m_finishDownloadEvent, kernel.m_stream));

      // Need to synchronize before releasing resources
      ERROR_CHECK(cudaStreamSynchronize(kernel.m_stream));

    } // Release the lock
  }

  // Release the resources (using a lock)
  kernel.ReleaseDeviceResources(&Scheduler::m_deviceInfo);

  // Exiting the function terminates this thread
}

/**
* @brief Run the experiment on a large batch of MatrixMultiply kernels, by using separate CUDA streams per run.
*/
void BatchMatrixMultiply::RunExperiment(const std::string &kernelName, int numRepeat)
{
  Scheduler::GetDeviceInfo();
  GenerateData();

  for (int n = 0; n < numRepeat; ++n)
  {
    // Call each kernel instance with a std::thread object
    std::thread *threads = new std::thread[m_data.size()];
    for (int kernelNum = 0; kernelNum < (int)m_data.size(); ++kernelNum)
      threads[kernelNum] = std::thread(RunKernelThreaded, this, kernelNum);

    // Wait for all threads to finish
    for (int kernelNum = 0; kernelNum < (int)m_data.size(); ++kernelNum)
      threads[kernelNum].join();

    // Validate and print results
    if (Scheduler::m_verbose) std::cout << "\n** Kernel Results **\n";
    bool freeHostMemory = n == (numRepeat - 1); // Only free host memory on the last iteration
    for (int kernelNum = 0; kernelNum < (int)m_data.size(); ++kernelNum)
    {
      m_data[kernelNum]->FinishHostExecution(freeHostMemory);
    }

    // Compute accumulated batch results
    ComputeBatchResults();

    // Record results to CSV
    OutputResultsCSV(kernelName);
  }
}
