#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>

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

  ERROR_CHECK(cudaStreamCreate(&m_stream));
  ERROR_CHECK(cudaEventCreate(&m_startQueueEvent));
  ERROR_CHECK(cudaEventCreate(&m_startExecEvent));
  ERROR_CHECK(cudaEventCreate(&m_finishExecEvent));

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
  int deviceNum, freeDeviceNum = -1;
  for (deviceNum = 0; deviceNum < (int)deviceInfo->size(); ++deviceNum)
  {
    DeviceInfo &device = deviceInfo->operator[](deviceNum);
    if (m_globalMemRequired < device.m_remainingGlobalMem && m_blocksRequired < device.m_remainingBlocksDimX)
    {
      freeDeviceNum = deviceNum;
      device.m_remainingGlobalMem -= m_globalMemRequired;
      device.m_remainingBlocksDimX -= m_blocksRequired;
      break;
    }
  }

  return freeDeviceNum;
}

/**
* @brief Execution is complete, release the GPU resources for other threads.
*/
void MultiplyAdd::ReleaseDeviceResources(std::vector< DeviceInfo > *deviceInfo)
{
  std::cout << "** Kernel " << m_kernelNum << " released GPU " << m_deviceNum << " **\n";

  DeviceInfo &device = deviceInfo->operator[](m_deviceNum);
  device.m_remainingGlobalMem += m_globalMemRequired;
  device.m_remainingBlocksDimX += m_blocksRequired;

  // Result is already in host memory, so free GPU memory
  FreeDeviceMemory();
}

/**
* @brief Execution is complete. Record completion event and timers, verify result, and free host memory.
*/
void MultiplyAdd::FinishHostExecution()
{
  // Update timers
  ERROR_CHECK(cudaEventElapsedTime(&m_queueTimeMillisec, m_startQueueEvent, m_startExecEvent));
  ERROR_CHECK(cudaEventElapsedTime(&m_execTimeMillisec, m_startExecEvent, m_finishExecEvent));

  // Verify the result - In current OpenMP version this is blocking other threads, so increasing Queue time..
  bool correct(true);
  for (int n = 0; n < m_vectorSize; ++n)
    correct = correct && (m_hC[n] == m_hCheckC[n]);

  printf("Kernel %d >> Device: %d, Queue: %.3fms, Execution: %.3fms, Correct: %s\n", m_deviceNum, m_kernelNum, m_queueTimeMillisec, m_execTimeMillisec, correct ? "True" : "False");

  // Free memory
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

  std::cout << "** Generating data **\n\tBatch Size: " << m_batchSize << ", Vector Size: " << m_meanVectorSize << ", Threads Per Block: " << m_threadsPerBlock << "\n";

  for (int kernelNum = 0; kernelNum < m_batchSize; ++kernelNum)
  {
    m_data[kernelNum] = new MultiplyAdd;
    m_data[kernelNum]->InitializeData((int)normalDist(randomGen), m_threadsPerBlock, kernelNum);
  }

  std::cout << "** Done generating data **\n\n";
}

/**
* @brief Run the experiment on a large batch of MultiplyAdd kernels, by using separate CUDA streams per run.
*/
void BatchMultiplyAdd::RunExperiment()
{
  Scheduler::GetDeviceInfo(m_numDevices);
  GenerateData();

  // Mark start queue events (needs to be done here, b/c CPU threads will block eachother)
  for (int kernelNum = 0; kernelNum < (int)m_data.size(); ++kernelNum)
    ERROR_CHECK(cudaEventRecord(m_data[kernelNum]->m_startQueueEvent, m_data[kernelNum]->m_stream));

  // Use an openMP loop to have multiple CPU threads trying to get an available GPU
  // Split up the runs of MultiplyAdd over multiple threads
  int numThreads = m_numCPUThreads == -1 ? omp_get_max_threads() : m_numCPUThreads;
  if (numThreads > omp_get_max_threads()) numThreads = omp_get_max_threads(); // Probably not necessary, but keep for now
#pragma omp parallel for schedule(static) num_threads(numThreads) default(none) shared(m_data, m_deviceInfo, m_threadsPerBlock)
  for (int kernelNum = 0; kernelNum < (int)m_data.size(); ++kernelNum)
  {
    MultiplyAdd &kernel = *m_data[kernelNum];

    // Acquire a GPU
    int deviceNum = -1;
    bool firstAttempt = true;
    while (deviceNum < 0)
    {
      if (firstAttempt)
      {
        std::cout << "** Kernel " << kernelNum << " queued for next available GPU **\n";
        firstAttempt = false;
      }

#pragma omp critical // Stalls here and waits for lock, and then locks, executes, unlocks
      {
        deviceNum = kernel.AcquireDeviceResources(&Scheduler::m_deviceInfo);
      }
    }

    std::cout << "** Kernel " << kernelNum << " acquired GPU " << deviceNum << " **\n";

    // Store the device number for updating resources in the stream callback
    kernel.m_deviceNum = deviceNum;

    // Mark the start execution event
    ERROR_CHECK(cudaEventRecord(kernel.m_startExecEvent, kernel.m_stream));
  
    // We've got a GPU, use it
    // Allocate memory on the GPU for input and output data
    std::size_t vectorBytes(kernel.m_vectorSize * sizeof(float));
    ERROR_CHECK(cudaSetDevice(deviceNum));
    ERROR_CHECK(cudaMalloc((void**)&kernel.m_dA, vectorBytes));
    ERROR_CHECK(cudaMalloc((void**)&kernel.m_dB, vectorBytes));
    ERROR_CHECK(cudaMalloc((void**)&kernel.m_dC, vectorBytes));
  
    // Upload the input data for this stream
    ERROR_CHECK(cudaMemcpyAsync(kernel.m_dA, kernel.m_hA, kernel.m_vectorSize * sizeof(float),
                                cudaMemcpyHostToDevice, kernel.m_stream));
    ERROR_CHECK(cudaMemcpyAsync(kernel.m_dB, kernel.m_hB, kernel.m_vectorSize * sizeof(float),
                                cudaMemcpyHostToDevice, kernel.m_stream));

    // Run the kernel
    const int bytes(0);
    dim3 blocks(m_threadsPerBlock, 1, 1);
    dim3 grid(kernel.m_blocksRequired, 1, 1);
    GPUMultiplyAdd<<<grid, blocks, bytes, kernel.m_stream>>>(kernel.m_vectorSize, kernel.m_dA, kernel.m_dB, kernel.m_dC);
    ERROR_CHECK(cudaPeekAtLastError());

    // Download the output data for this stream, wait for it to copy back before continuing
    ERROR_CHECK(cudaMemcpyAsync(kernel.m_hC, kernel.m_dC, kernel.m_vectorSize * sizeof(float), 
                                cudaMemcpyDeviceToHost, kernel.m_stream));

    // Record the time (since stream is non-zero, waits for stream to be complete)
    ERROR_CHECK(cudaEventRecord(kernel.m_finishExecEvent, kernel.m_stream));

    // Need to synchronize before releasing resources
    ERROR_CHECK(cudaStreamSynchronize(kernel.m_stream));

    #pragma omp critical // Stalls here and waits for lock, and then locks, executes, unlocks
    {
      kernel.ReleaseDeviceResources(&Scheduler::m_deviceInfo);
    }
  }

  std::cout << "\n** Kernel Results **\n";
  for (int kernelNum = 0; kernelNum < (int)m_data.size(); ++kernelNum)
  {
    m_data[kernelNum]->FinishHostExecution();
  }

  ERROR_CHECK(cudaDeviceSynchronize());
}