#include "scheduler.cuh"

std::vector< DeviceInfo > Scheduler::m_deviceInfo = std::vector< DeviceInfo >(); // Static member variable initialization

/**
* @brief Get relevant device info for a batch run. Generic to whatever kernel we're running batches of.
*/
void Scheduler::GetDeviceInfo(int maxNumDevices)
{
  // Get device info
  int deviceCount(0);
  ERROR_CHECK(cudaGetDeviceCount(&deviceCount));

  int numDevices = deviceCount <= maxNumDevices ? deviceCount : maxNumDevices;
  m_deviceInfo.resize(numDevices);

  for (int deviceNum = 0; deviceNum < numDevices; ++deviceNum)
    m_deviceInfo[deviceNum].SetDeviceInfo(deviceNum);
}

void CUDART_CB Scheduler::ReleaseResources(cudaStream_t stream, cudaError_t status, void *object)
{
  // Call the method from the kernel class that extends ScheduledKernel
  ((ScheduledKernel*)object)->ReleaseResources(&m_deviceInfo); 
}

void CUDART_CB Scheduler::FinishExecution(cudaStream_t stream, cudaError_t status, void *object)
{
  // Call the method from the kernel class that extends ScheduledKernel
  ((ScheduledKernel*)object)->FinishExecution();
}

/**
* @brief Find a device with enough resources, and if available, decrement the available resources and return the id.
*/
/*
int Scheduler::GetFreeDevice(const std::size_t &globalMem, const int &blocksDimX)
{
  int deviceNum, freeDeviceNum = -1;
  for (deviceNum = 0; deviceNum < (int)m_deviceInfo.size(); ++deviceNum)
  {
    DeviceInfo &device = m_deviceInfo[deviceNum];
    if (globalMem < device.m_remainingGlobalMem && blocksDimX < device.m_remainingBlocksDimX)
    {
      freeDeviceNum = deviceNum;
      device.m_remainingGlobalMem -= globalMem;
      device.m_remainingBlocksDimX -= blocksDimX;
      break;
    }
  }

  return freeDeviceNum;
}
*/
