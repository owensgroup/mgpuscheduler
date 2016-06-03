#ifndef SCHEDULER_CUH
#define SCHEDULER_CUH

#include <vector>
#include <mutex>

class DeviceInfo
{
public:
  /**
  * @brief Stores relevant GPU device properties used by this scheduler
  * @param[in] deviceNum	Device number, less than cudaGetDeviceCount().
  */
  void SetDeviceInfo(int deviceNum);

  // Keeping public for now, ease of access
  std::size_t m_totalGlobalMem, m_remainingGlobalMem; // Global memory
  int m_totalBlocksDimX, m_remainingBlocksDimX;       // Grid size, number of blocks per grid, dim X of (X,Y,Z)
  int m_totalCores, m_remainingTotalCores;            // Cores (don't know how to use these, only initialized)
};

class ScheduledKernel
{
public:
  virtual int  AcquireDeviceResources(std::vector< DeviceInfo > *deviceInfo) = 0; // Acquire GPU resources for some device 
  virtual void ReleaseDeviceResources(std::vector< DeviceInfo > *deviceInfo) = 0; // Release GPU resources for chosen device
  
  std::mutex m_deviceInfoMutex; // Needed for locking GPU resources
};

class Scheduler
{
public:
  static void  GetDeviceInfo();

  static std::vector< DeviceInfo > m_deviceInfo;  // Resources available on GPUs
  static int m_maxDevices;    // Run-time parameters for the GPU(s)
  static bool m_verbose;      // Validate and print results (don't want to do while timing)
};



#endif // #ifndef SCHEDULER_CUH