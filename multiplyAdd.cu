#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <thread>
#include <fstream>
#include <algorithm>

#include "multiplyAdd.cuh"
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

__global__ void GPUMultiplyAdd(int n, float * A, float * B, float * C)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < n) {
    C[index] = (A[index] + B[index]) * 2.0f;
  }
}

void MultiplyAdd::FreeHostMemory()
{
  if (m_hA) free(m_hA);
  if (m_hB) free(m_hB);
  if (m_hC) free(m_hC);
  if (m_hCheckC) free(m_hCheckC);
  m_hA = m_hB = m_hC = m_hCheckC = NULL;
}

void MultiplyAdd::FreeDeviceMemory()
{
  if (m_dA) ERROR_CHECK(cudaFree(m_dA));
  if (m_dB) ERROR_CHECK(cudaFree(m_dB));
  if (m_dC) ERROR_CHECK(cudaFree(m_dC));
  m_dA = m_dB = m_dC = NULL;
}

/**
* @brief Initialize host vectors for a single MultiplyAdd run.
* @param[in] vectorSize	The size of each vector.
*/
void MultiplyAdd::InitializeData(int vectorSize, int threadsPerBlock, int kernelNum)
{
  m_vectorSize = vectorSize;
  m_kernelNum = kernelNum;

  m_hA = (float*)malloc(sizeof(float) * vectorSize);
  m_hB = (float*)malloc(sizeof(float) * vectorSize);
  m_hC = (float*)malloc(sizeof(float) * vectorSize);
  m_hCheckC = (float*)malloc(sizeof(float) * vectorSize);

  m_blocksRequired = vectorSize % threadsPerBlock == 0 ? (vectorSize / threadsPerBlock) : 1 + (vectorSize / threadsPerBlock);
  m_globalMemRequired = 3 * sizeof(float) * vectorSize;

  m_floatingPointOps = (float)(2 * m_vectorSize);  // One add, one multiply per vector element
  m_memBytesReadWrite = (float)(3 * sizeof(float) * m_vectorSize); // Two reads, one write, per vector element

  // Fill in A and B with random numbers (should be seeded prior to call)
  float invRandMax = 1000.0f / RAND_MAX; // Produces random numbers between 0 and 1000
  for (int n = 0; n < vectorSize; ++n)
  {
    m_hA[n] = std::rand() * invRandMax;
    m_hB[n] = std::rand() * invRandMax;
  
    // Fill in the hCheckC array for validation on the result
    m_hCheckC[n] = (m_hA[n] + m_hB[n]) * 2.0f; // MUST match the kernel for MultiplyAdd!
  }
}

/**
* @brief Find a device with enough resources, and if available, decrement the available resources and return the id.
*/
int MultiplyAdd::AcquireDeviceResources(std::vector< DeviceInfo > *deviceInfo)
{
  // Lock this method
  std::lock_guard< std::mutex > guard(m_deviceInfoMutex); // Automatically unlocks when destroyed

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

      if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << ": - Used Memory: " << m_globalMemRequired / (1024.0*1024.0) << "MB, Blocks: " << m_blocksRequired << "\n";
      if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << ": =  New Memory: " << device.m_remainingGlobalMem / (1024.0*1024.0) << "MB, Blocks: " << device.m_remainingBlocksDimX << "\n";
      if (Scheduler::m_verbose) std::flush(std::cout);

      break;
    }
    else
    {
      if (Scheduler::m_verbose) std::cout << "\n**********************************************************************\n";
      if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << " >> Unable to be acquired by device " << deviceNum << "\n";
      if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << ": Kernel Memory: " << m_globalMemRequired / (1024.0*1024.0) << "MB, Blocks: " << m_blocksRequired << "\n";
      if (Scheduler::m_verbose) std::cout << "** Kernel " << m_kernelNum << ": Device Memory: " << device.m_remainingGlobalMem / (1024.0*1024.0) << "MB, Blocks: " << device.m_remainingBlocksDimX << "\n";
      if (Scheduler::m_verbose) std::cout << "**********************************************************************\n";
    }
  }

  return freeDeviceNum;
}

/**
* @brief Execution is complete, release the GPU resources for other threads.
*/
void MultiplyAdd::ReleaseDeviceResources(std::vector< DeviceInfo > *deviceInfo)
{
  // Lock this method
  std::lock_guard< std::mutex > guard(m_deviceInfoMutex); // Automatically unlocks when destroyed

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
void MultiplyAdd::FinishHostExecution(bool freeHostMemory)
{
  // Update timers
  ERROR_CHECK(cudaEventElapsedTime(&m_kernelExecTimeMS, m_startExecEvent, m_finishExecEvent));
  ERROR_CHECK(cudaEventElapsedTime(&m_totalExecTimeMS, m_startCudaMallocEvent, m_finishDownloadEvent));

  // Compute MFLOP/s, MB/s for this kernel
  m_MFLOPs = m_floatingPointOps / ((2 ^ 20) * (1000 *m_kernelExecTimeMS));
  m_MBps = m_memBytesReadWrite / ((2 ^ 20) * (1000 * m_kernelExecTimeMS));

  // Verify the result
  bool correct(true);
  for (int n = 0; n < m_vectorSize; ++n)
    correct = correct && (m_hC[n] == m_hCheckC[n]);

  if (Scheduler::m_verbose) printf("Kernel %d >> Device: %d, Queue: %.3fms, Kernel: %.3fms, Total: %.3fms, MFLOP/s: %.2f, MB/s: %.2f, Correct: %s\n", 
        m_kernelNum, m_deviceNum, m_queueTimeNS, m_kernelExecTimeMS, m_totalExecTimeMS, m_MFLOPs, m_MBps, correct ? "True" : "False");

  // Free memory
  if (freeHostMemory)
    FreeHostMemory();
}

/**
* @brief Generate data for the entire batch of MultiplyAdd's being run.
*/
void BatchMultiplyAdd::GenerateData()
{
  m_data.resize(m_batchSize);

  // Get a random generator with a normal distribution, mean = meanVectorSize, stdDev = 0.1*meanVectorSize
  std::normal_distribution< float > normalDist((float)m_meanVectorSize, 0.1f*m_meanVectorSize);

  // Seed by the batch size for both the std::rand generator and the std::default_random_engine, used by distribution
  std::srand(m_batchSize);
  std::default_random_engine randomGen(m_batchSize);

  if (Scheduler::m_verbose) std::cout << "** Generating data **\n\tBatch Size: " << m_batchSize << ", Vector Size: " 
            << m_meanVectorSize << ", Threads Per Block: " << m_threadsPerBlock << "\n";

  for (int kernelNum = 0; kernelNum < m_batchSize; ++kernelNum)
  {
    m_data[kernelNum] = new MultiplyAdd;
    m_data[kernelNum]->InitializeData((int)normalDist(randomGen), m_threadsPerBlock, kernelNum);
  }

  if (Scheduler::m_verbose) std::cout << "** Done generating data **\n\n";
}

void BatchMultiplyAdd::ComputeBatchResults()
{
  // Sum up the per-kernel floating point ops and mem bytes accessed
  m_batchFloatingPointOps = m_batchMemBytesReadWrite = 0;
  for (int kernel = 0; kernel < (int)m_data.size(); ++kernel)
  {
    m_batchFloatingPointOps += m_data[kernel]->m_floatingPointOps;
    m_batchMemBytesReadWrite += m_data[kernel]->m_memBytesReadWrite;
  }

  // Use queue times to find which kernel was run first, and which last.
  struct MultiplyAddComp
  {
    bool operator()(const MultiplyAdd *lhs, const MultiplyAdd *rhs)
    {
      return lhs->m_queueTimeNS < rhs->m_queueTimeNS;
    }
  };

  std::sort(m_data.begin(), m_data.end(), MultiplyAddComp());

  m_batchKernelExecTimeMS = m_batchTotalExecTimeMS = -1;
  m_batchGFLOPs = m_batchGBps = -1;
  if (m_data.size() < 2)
    return;

  const MultiplyAdd &firstKernel = **m_data.begin();
  const MultiplyAdd &lastKernel = **m_data.rbegin();
  ERROR_CHECK(cudaEventElapsedTime(&m_batchKernelExecTimeMS, firstKernel.m_startExecEvent, lastKernel.m_finishExecEvent));
  ERROR_CHECK(cudaEventElapsedTime(&m_batchTotalExecTimeMS, firstKernel.m_startCudaMallocEvent, lastKernel.m_finishDownloadEvent));

  // Compute GFLOP/s, GB/s for this batch
  m_batchGFLOPs = m_batchFloatingPointOps / ((2 ^ 30) * (1000 * m_batchKernelExecTimeMS));
  m_batchGBps = m_batchMemBytesReadWrite / ((2 ^ 30) * (1000 * m_batchKernelExecTimeMS));
}

void BatchMultiplyAdd::OutputResultsCSV(const std::string &kernelName)
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
    csvKernelFile << "BatchSize, KernelName, MeanVectorSize, ThreadsPerBlock, MaxDevices, KernelNum, QueueTimeMS"
                  << ", KernelExecTimeMS, TotalExecTimeMS, FloatingPtOps, MemBytes, MFLOPs, MBps\n";
  }

  for (int kernelNum = 0; kernelNum < (int)m_data.size(); ++kernelNum)
  {
    const MultiplyAdd &kernel = *m_data[kernelNum];
    csvKernelFile << m_batchSize << ", " << kernelName.c_str() << ", " << m_meanVectorSize << ", " << m_threadsPerBlock
      << ", " << Scheduler::m_maxDevices << ", " << kernel.m_kernelNum << ", " << kernel.m_queueTimeNS 
      << ", " << kernel.m_kernelExecTimeMS << ", " << kernel.m_totalExecTimeMS << ", " << kernel.m_floatingPointOps 
      << ", " << kernel.m_memBytesReadWrite << ", " << kernel.m_MFLOPs << ", " << kernel.m_MBps << "\n";
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
    csvBatchFile << "BatchSize, KernelName, MeanVectorSize, ThreadsPerBlock, MaxDevices, BatchKernelExecTimeMS"
                << ", BatchTotalExecTimeMS, FloatingPtOps, MemBytes, GFLOPs, GBps\n";
  }

  csvBatchFile << m_batchSize << ", " << kernelName.c_str() << ", " << m_meanVectorSize << ", " << m_threadsPerBlock
                << ", " << Scheduler::m_maxDevices << ", " << m_batchKernelExecTimeMS << ", " << m_batchTotalExecTimeMS
                << ", " << m_batchFloatingPointOps << ", " << m_batchMemBytesReadWrite << ", " << m_batchGFLOPs 
                << ", " << m_batchGBps << "\n";
}

// NVCC having trouble parsing the std::thread() call when this is a member function, so keeping it non-member friend
void RunKernelThreaded(BatchMultiplyAdd *batch, int kernelNum)
{
  MultiplyAdd &kernel = *(batch->m_data[kernelNum]);

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

    // Try to acquire GPU resources (using a lock)
    deviceNum = kernel.AcquireDeviceResources(&Scheduler::m_deviceInfo);
  }

  // Get the time spent in the queue
  auto finishQueue = clock::now();
  std::chrono::duration< double > diff;
  kernel.m_queueTimeNS = (float)(1000 * diff.count());

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
  std::size_t vectorBytes(kernel.m_vectorSize * sizeof(float));
  ERROR_CHECK(cudaMalloc((void**)&kernel.m_dA, vectorBytes));
  ERROR_CHECK(cudaMalloc((void**)&kernel.m_dB, vectorBytes));
  ERROR_CHECK(cudaMalloc((void**)&kernel.m_dC, vectorBytes));

  // Upload the input data for this stream
  ERROR_CHECK(cudaMemcpyAsync(kernel.m_dA, kernel.m_hA, kernel.m_vectorSize * sizeof(float),
    cudaMemcpyHostToDevice, kernel.m_stream));
  ERROR_CHECK(cudaMemcpyAsync(kernel.m_dB, kernel.m_hB, kernel.m_vectorSize * sizeof(float),
    cudaMemcpyHostToDevice, kernel.m_stream));

  // Mark the start kernel execution event
  ERROR_CHECK(cudaEventRecord(kernel.m_startExecEvent, kernel.m_stream));

  // Run the kernel
  const int bytes(0);
  dim3 dimBlock(batch->m_threadsPerBlock, 1, 1);
  dim3 dimGrid(kernel.m_blocksRequired, 1, 1);
  GPUMultiplyAdd<<< dimGrid, dimBlock, bytes, kernel.m_stream >>>(kernel.m_vectorSize, kernel.m_dA, kernel.m_dB, kernel.m_dC);
  ERROR_CHECK(cudaPeekAtLastError());

  // Record the time (since stream is non-zero, waits for stream to be complete)
  ERROR_CHECK(cudaEventRecord(kernel.m_finishExecEvent, kernel.m_stream));

  // Download the output data for this stream
  ERROR_CHECK(cudaMemcpyAsync(kernel.m_hC, kernel.m_dC, kernel.m_vectorSize * sizeof(float),
    cudaMemcpyDeviceToHost, kernel.m_stream));

  // Mark the end of total execution event
  ERROR_CHECK(cudaEventRecord(kernel.m_finishDownloadEvent, kernel.m_stream));

  // Need to synchronize before releasing resources
  ERROR_CHECK(cudaStreamSynchronize(kernel.m_stream));

  // Release the resources (using a lock)
  kernel.ReleaseDeviceResources(&Scheduler::m_deviceInfo);

  // Exiting the function terminates this thread
}

/**
* @brief Run the experiment on a large batch of MultiplyAdd kernels, by using separate CUDA streams per run.
*/
void BatchMultiplyAdd::RunExperiment(const std::string &kernelName, int numRepeat)
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