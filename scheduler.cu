#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "scheduler.cuh"

std::vector< DeviceInfo > Scheduler::m_deviceInfo = std::vector< DeviceInfo >(); // Static member variable initialization
int Scheduler::m_maxDevices = 0;
bool Scheduler::m_verbose = false;
std::mutex Scheduler::m_deviceInfoMutex;

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

void DeviceInfo::SetDeviceInfo(int deviceNum)
{
  // NOTE: This should really be using the driver api to get the amount of memory available, allocate a huge chunk,
  //       and manage it internally (ideally), or keep querying cuMemGetInfo() and reserve some space for fragmentation.
  //       For now, just reserve 2GB of free space and hope this is enough..
  cudaSetDevice(deviceNum);
  cudaDeviceProp deviceProp;
  ERROR_CHECK(cudaGetDeviceProperties(&deviceProp, deviceNum));

  // Total values aren't being used right now, but may want them for CSV output later, keep around for now
  m_totalGlobalMem = deviceProp.totalGlobalMem;
  m_totalCores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
  m_totalBlocksDimX = deviceProp.maxGridSize[0];

  m_remainingGlobalMem = (std::size_t)(m_totalGlobalMem - 2*(1024.0*1024.0*1024.0)); // "Reserve" 2 GB for other resources and fragmentation!...
  m_remainingTotalCores = m_totalCores;
  m_remainingBlocksDimX = m_totalBlocksDimX;
}

/**
* @brief Get relevant device info for a batch run. Generic to whatever kernel we're running batches of.
*/
void Scheduler::GetDeviceInfo()
{
  // Get device info
  int deviceCount(0);
  ERROR_CHECK(cudaGetDeviceCount(&deviceCount));

  int numDevices = deviceCount <= m_maxDevices ? deviceCount : m_maxDevices;
  m_deviceInfo.resize(numDevices);

  for (int deviceNum = 0; deviceNum < numDevices; ++deviceNum)
    m_deviceInfo[deviceNum].SetDeviceInfo(deviceNum);
}
