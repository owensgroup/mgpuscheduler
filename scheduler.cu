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
