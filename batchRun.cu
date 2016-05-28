#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "batchRun.h"
#include "deviceInfo.cuh"

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

/**
* @brief Initialize host vectors for a single MultiplyAdd run.
* @param[in] vectorSize	The size of each vector.
*/
void MultiplyAdd::InitializeHostData(int vectorSize, int threadsPerBlock)
{
  m_vectorSize = vectorSize;

  m_hA = (float*)malloc(sizeof(float) * vectorSize);
  m_hB = (float*)malloc(sizeof(float) * vectorSize);
  m_hC = (float*)malloc(sizeof(float) * vectorSize);
  m_hCheckC = (float*)malloc(sizeof(float) * vectorSize);

  m_threadsRequired = threadsPerBlock;
  //m_blocksRequired = ((vectorSize + threadsPerBlock - 1) / threadsPerBlock);
  m_blocksRequired = vectorSize % threadsPerBlock == 0 ? (vectorSize / threadsPerBlock) : 1 + (vectorSize / threadsPerBlock);
  m_globalMemRequired = 3 * sizeof(float) * vectorSize;

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
* @brief Get relevant device info for a batch run. Generic to whatever kernel we're running batches of.
*/
void BatchRun::GetDeviceInfo()
{
  // Get device info
  int deviceCount(0);
  ERROR_CHECK(cudaGetDeviceCount(&deviceCount));

  m_deviceInfo.resize(deviceCount);

  for (int deviceNum = 0; deviceNum < deviceCount; ++deviceNum)
    m_deviceInfo[deviceNum].SetDeviceInfo(deviceNum);
}

/**
* @brief Find a device with enough resources, and if available, decrement the available resources and return the id.
*/
int BatchRun::GetFreeDevice(const unsigned long long &globalMemRequired,
  const int &threadsRequired, const int &blocksRequired)
{
  int device, deviceNum = -1;
  for (int device = 0; device < (int)m_deviceInfo.size(); ++device)
  {
    // ************ TODO NEXT *************
  }

  if (device < (int)m_deviceInfo.size()) deviceNum = device;
  return deviceNum;
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

  for (int n = 0; n < m_batchSize; ++n)
  {
    m_data[n] = new MultiplyAdd;
    m_data[n]->InitializeHostData((int)normalDist(randomGen), m_threadsPerBlock);
  }
}


/**
* @brief Run the experiment on a large batch of MultiplyAdd kernels.
*/
void BatchMultiplyAdd::RunExperiment()
{
  GetDeviceInfo();
  GenerateData();

  // Use the openMP loop to have each thread try to get an available GPU
  int numThreads = omp_get_max_threads();

#pragma omp parallel for schedule(static) num_threads(numThreads) default(none) shared(m_data, m_deviceInfo) private(deviceNum)
  for (int kernel = 0; kernel < (int)m_data.size(); ++kernel)
  {
    // Try to acquire a GPU
    int deviceNum = -1;
    while (deviceNum < 0)
    {
#pragma omp critical // Stalls here, and when free locks, executes, unlocks
      {
        deviceNum = GetFreeDevice(m_data[kernel]->m_globalMemRequired, m_data[kernel]->m_threadsRequired, m_data[kernel]->m_blocksRequired);
      }
    }
  
    // Continue using this GPU
  }
}