#include <string.h>

#include <iostream>
#include <fstream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "multiplyAdd.cuh"
#include "matrixmultiply.cuh"
#include "scheduler.cuh"

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
    if (argc != 8)
    {
      fprintf(stderr, "Usage: %s inputSize batchSize maxDevices maxGPUsPerKernel kernelName kernelArgument verboseOutput\n", argv[0]);
      return false;
    }

    // Set parameters, no error checking currently
    m_inputSize = atoi(argv[1]);
    m_batchSize = atoi(argv[2]);
    m_maxDevices = atoi(argv[3]);
    m_maxGPUsPerKernel = atoi(argv[4]);
    m_kernelName = std::string(argv[5]);
    m_kernelArgument = atoi(argv[6]);
    m_verbose = atoi(argv[7]) == 0 ? false : true;

    return true;
  }

  int m_inputSize;
  int m_batchSize;
  int m_maxDevices;
  int m_maxGPUsPerKernel;
  int m_kernelArgument;
  std::string m_kernelName;
  bool m_verbose;
};

int main(int argc, char** argv)
{
  // Parse command line
  CommandLineArgs args;
  if (!args.ParseCommandLine(argc, argv)) return -1;

  // Set the scheduler parameters
  Scheduler::m_maxDevices = args.m_maxDevices;
  Scheduler::m_maxGPUsPerKernel = args.m_maxGPUsPerKernel;
  Scheduler::m_verbose = args.m_verbose;

  // Run the experiment for MultiplyAdd
  if (args.m_kernelName == std::string("MultiplyAdd"))
  {
    int &threadsPerBlock = args.m_kernelArgument;
    BatchMultiplyAdd batchMultAdd(args.m_inputSize, args.m_batchSize, threadsPerBlock);
    batchMultAdd.RunExperiment(std::string("MultiplyAdd"));
  }
  // Run the experiment for MatrixMultiply
  else if (args.m_kernelName == std::string("MatrixMultiply"))
  {
    int &blockWidth = args.m_kernelArgument;
    BatchMatrixMultiply batchMtxMulti(args.m_inputSize, args.m_batchSize, blockWidth);
    batchMtxMulti.RunExperiment(std::string("MatrixMultiply"));
  }

  return 0;
}
