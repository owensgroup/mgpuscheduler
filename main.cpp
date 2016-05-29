#include <string.h>

#include <iostream>
#include <fstream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "multiplyAdd.cuh"

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
    if (argc != 5 && argc != 6)
    {
      fprintf(stderr, "Usage: %s meanVectorSize batchSize maxDevices threadsPerBlock\n", argv[0]);
      return false;
    }

    // Set parameters, no error checking currently
    m_meanVectorSize = atoi(argv[1]);
    m_batchSize = atoi(argv[2]);
    m_numDevices = atoi(argv[3]);
    m_threadsPerBlock = atoi(argv[4]);
    
    m_numCPUThreads = argc == 6 ? atoi(argv[5]) : -1;

    return true;
  }

  int m_meanVectorSize;
  int m_batchSize;
  int m_numDevices;
  int m_threadsPerBlock;
  int m_numCPUThreads;
};

int main(int argc, char** argv)
{
  // Parse command line
  CommandLineArgs args;
  if (!args.ParseCommandLine(argc, argv)) return -1;

  // Run the experiment for MultiplyAdd
  BatchMultiplyAdd batchMultAdd(args.m_meanVectorSize, args.m_batchSize, args.m_numDevices, 
                                args.m_threadsPerBlock, args.m_numCPUThreads);
  batchMultAdd.RunExperiment();

  return 0;
}