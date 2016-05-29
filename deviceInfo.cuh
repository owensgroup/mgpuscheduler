#ifndef DEVICE_INFO_H
#define DEVICE_INFO_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

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

class DeviceInfo
{
public:
  /**
  * @brief Stores relevant GPU device properties
  * @param[in] deviceNum	Device number, less than cudaGetDeviceCount().
  */
  void SetDeviceInfo(int deviceNum)
  {
    cudaSetDevice(deviceNum);
    cudaDeviceProp deviceProp;
    ERROR_CHECK(cudaGetDeviceProperties(&deviceProp, deviceNum));

    m_totalGlobalMem = deviceProp.totalGlobalMem;
    m_totalCores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
    m_totalBlocksDimX = deviceProp.maxGridSize[0];

    m_remainingGlobalMem = m_totalGlobalMem;
    m_remainingTotalCores = m_totalCores;
    m_remainingBlocksDimX = m_totalBlocksDimX;
  }

  // Keeping public for now, ease of access
  std::size_t m_totalGlobalMem, m_remainingGlobalMem; // Global memory
  int m_totalBlocksDimX, m_remainingBlocksDimX;       // Grid size, number of blocks per grid, dim X of (X,Y,Z)
  int m_totalCores, m_remainingTotalCores;            // Cores (don't know how to use these, only initialized)
};


#endif // ifndef DEVICE_INFO_H