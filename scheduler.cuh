#ifndef SCHEDULER_CUH
#define SCHEDULER_CUH

#include <cuda_runtime.h>
#include <vector>

#include "deviceInfo.cuh"

class ScheduledKernel
{
public:
  virtual int  AcquireResources(std::vector< DeviceInfo > *deviceInfo) = 0; // Acquire GPU resources for some device 
  virtual void ReleaseResources(std::vector< DeviceInfo > *deviceInfo) = 0; // Release GPU resources for chosen device
  virtual void FinishExecution() = 0; // Finish anything else that needs to be done after execution
};

class Scheduler
{
public:
  static void  GetDeviceInfo(int maxNumDevices);
  //static int   AcquireResources(void *object); // Not needed as of now, just call ScheduledKernel::AcquireResources
  static void CUDART_CB ReleaseResources(cudaStream_t stream, cudaError_t status, void *object);
  static void CUDART_CB FinishExecution(cudaStream_t stream, cudaError_t status, void *object);

  static std::vector< DeviceInfo > m_deviceInfo;
};



#endif // #ifndef SCHEDULER_CUH