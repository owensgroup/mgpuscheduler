#ifndef DEVICE_INFO_H
#define DEVICE_INFO_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

/**
* @brief Macro for error checking for all GPU calls
* @param[in] ans	The GPU call itself, which evaluates to the cudaError_t returned.
*/
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
  
    m_remainingGlobalMem = m_totalGlobalMem;
    m_remainingTotalCores = m_totalCores;
  }

  // Keeping public for now, ease of access
  unsigned long long m_totalGlobalMem, m_remainingGlobalMem;
  int m_totalCores, m_remainingTotalCores;
};


#endif // ifndef DEVICE_INFO_H