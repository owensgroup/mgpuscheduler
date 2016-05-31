#include <string.h>

#include <iostream>
#include <fstream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "multiplyAdd.cuh"
// #include "matrixmultiply.cuh"
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
    if (argc != 7)
    {
      fprintf(stderr, "Usage: %s meanVectorSize batchSize maxDevices threadsPerBlock maxGPUsPerKernel verboseOutput\n", argv[0]);
      return false;
    }

    // Set parameters, no error checking currently
    m_meanVectorSize = atoi(argv[1]);
    m_batchSize = atoi(argv[2]);
    m_maxDevices = atoi(argv[3]);
    m_threadsPerBlock = atoi(argv[4]);
    m_maxGPUsPerKernel = atoi(argv[5]);
    m_verbose = atoi(argv[6]) == 0 ? false : true;

    return true;
  }

  int m_meanVectorSize;
  int m_batchSize;
  int m_maxDevices;
  int m_threadsPerBlock;
  int m_maxGPUsPerKernel;
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
  BatchMultiplyAdd batchMultAdd(args.m_meanVectorSize, args.m_batchSize, args.m_threadsPerBlock);
  batchMultAdd.RunExperiment(std::string("MultiplyAdd"));

  // Run the experiment for MatrixMultiply
  // BatchMatrixMultiply batchMtxMulti(args.m_meanVectorSize, args.m_batchSize, args.m_threadsPerBlock);
  // batchMtxMulti.RunExperiment(std::string("MatrixMultiply"));

  return 0;
}
