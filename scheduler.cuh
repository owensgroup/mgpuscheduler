#ifndef SCHEDULER_CUH
#define SCHEDULER_CUH

#include <cuda_runtime.h>
#include <vector>

#include "deviceInfo.cuh"

class ScheduledKernel
{
public:
  virtual int  AcquireDeviceResources(std::vector< DeviceInfo > *deviceInfo) = 0; // Acquire GPU resources for some device 
  virtual void ReleaseDeviceResources(std::vector< DeviceInfo > *deviceInfo) = 0; // Release GPU resources for chosen device
};

class Scheduler
{
public:
  static void  GetDeviceInfo(int maxNumDevices);
  static std::vector< DeviceInfo > m_deviceInfo;
};



#endif // #ifndef SCHEDULER_CUH