#ifndef SCHEDULER_CU
#define SCHEDULER_CU

#include <string.h>

#include <iostream>
#include <fstream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Error checking for CUDA memory allocations.
 */
#define ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file,
                     int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
     fprintf(stderr,"Cuda error in file '%s' in line '%d': %s\n",
             file, line, cudaGetErrorString(code));
     if (abort) exit(code);
  }
}

#include "singlegpu.cuh"
#include "multiplegpus.cuh"
#include "deviceInfo.cuh"

// =================================================================================================
// =================================================================================================

class CommandLineArgs 
{
public:
  /**
  * @brief Parses the commandline input and stores in CMD struct
  * @param[in] argc          Total number of arguments.
  * @param[in] argv          List of arguments.
  * @param[out] CMD          Struct that stores the commands
  */
	bool ParseCommandLine(int argc, char **argv)
	{
    if (argc != 3) 
    {
      fprintf(stderr, "Usage: %s meanVectorSize batchSize maxDevices\n", argv[0]);
      return false;
    }

    // Set parameters, no error checking currently
    m_meanVectorSize = atoi(argv[1]);
    m_batchSize = atoi(argv[2]);
    m_maxDevices = atoi(argv[3]);

    return true;
	}

  int m_meanVectorSize;
  int m_batchSize;
  int m_maxDevices;
};

void GenerateData(int meanVectorSize, int batchSize)
{

}

int main(int argc, char** argv)
{
	// Parse command line
  CommandLineArgs args;
  if (!args.ParseCommandLine) return -1;

	// Get device info
	int deviceCount(0);
	ERROR_CHECK(cudaGetDeviceCount(&deviceCount));
	std::vector< DeviceInfo > devices(deviceCount);
	for (int deviceNum = 0; deviceNum < deviceCount; ++deviceNum)
		devices[deviceNum].SetDeviceInfo(deviceNum);

  // Setup the host data
  GenerateData(args.m_meanVectorSize, args.m_batchSize);

	// Run the experiment

  //if (!(*build).multigpu) sched::sgpu::SingleGPUApplication((*build).size);
  //else sched::mgpu::MultiGPUApplication((*build).size);

    return 1;
}

// } // namespace: sched

#endif // SCHEDULER_CU
