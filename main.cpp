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
      fprintf(stderr, "Usage: %s inputSize batchSize maxDevices kernelName kernelArgument numRepeat verboseOutput\n", argv[0]);
      return false;
    }

    // Set parameters, no error checking currently
    m_inputSize = atoi(argv[1]);
    m_batchSize = atoi(argv[2]);
    m_maxDevices = atoi(argv[3]);
    m_kernelName = std::string(argv[4]);
    m_kernelArgument = atoi(argv[5]);
    m_numRepeat = atoi(argv[6]);
    m_verbose = atoi(argv[7]) == 0 ? false : true;

    return true;
  }

  int m_inputSize;          // Kernel input size (e.g. vector size or matrix size)
  int m_batchSize;          // Batch size, number of kernels to queue up and run as a batch
  int m_maxDevices;         // Number of GPUs to use (auto limited by maximum available)
  int m_kernelArgument;     // Single kernel argument (e.g. threads per block)
  std::string m_kernelName; // Kernel name, for CSV output
  int m_numRepeat;          // Number of times to repeat a single experiment, for better data
  bool m_verbose;           // Verbose flag (needs to be false for gathering real results)
};

int main(int argc, char** argv)
{
  // Parse command line
  CommandLineArgs args;
  if (!args.ParseCommandLine(argc, argv)) return -1;

  // Set the scheduler parameters
  Scheduler::m_maxDevices = args.m_maxDevices;
  Scheduler::m_verbose = args.m_verbose;

  // Run the experiment for MultiplyAdd
  if (args.m_kernelName == std::string("MultiplyAdd"))
  {
    int &threadsPerBlock = args.m_kernelArgument;
    BatchMultiplyAdd batchMultAdd(args.m_inputSize, args.m_batchSize, threadsPerBlock);
    batchMultAdd.RunExperiment(std::string("MultiplyAdd"), args.m_numRepeat);
  }
  // Run the experiment for MatrixMultiply
  else if (args.m_kernelName == std::string("MatrixMultiply"))
  {
    int &blockWidth = args.m_kernelArgument;
    BatchMatrixMultiply batchMtxMulti(args.m_inputSize, args.m_batchSize, blockWidth);
    batchMtxMulti.RunExperiment(std::string("MatrixMultiply"), args.m_numRepeat);
  }

  return 0;
}
