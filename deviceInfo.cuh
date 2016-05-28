#ifndef DEVICE_INFO_H
#define DEVICE_INFO_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

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
		cudaGetDeviceProperties(&deviceProp, deviceNum);

		m_totalGlobalMem = deviceProp.totalGlobalMem;
		m_totalCores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
	}

	// Keeping public for now, ease of access
	unsigned long long m_totalGlobalMem;
	int m_totalCores;
};


#endif // ifndef DEVICE_INFO_H