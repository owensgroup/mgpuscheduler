#ifndef SCHEDULER_CU
#define SCHEDULER_CU

#include <string.h>

#include <iostream>
#include <fstream>

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


struct CommandLineArgs {
  int size;
  bool multigpu;
}; /* Stores command line arguments */


/**
 * @brief prints an error if the usage of commandline is incorrect
 * @param[in] argv          List of arguments.
 */

 void usage(char ** argv) {
    fprintf(stderr, "Usage: %s n --{sgpu,mgpu}\n", argv[0]);
    exit(1);
 }

/**
 * @brief Parses the commandline input and stores in CMD struct
 * @param[in] argc          Total number of arguments.
 * @param[in] argv          List of arguments.
 * @param[out] CMD          Struct that stores the commands
 */
void input(int argc, char** argv, CommandLineArgs * build) {
  if (argc != 3) {
      usage(argv);
  }

  // Set size n
  int n = atoi(argv[1]);
  (*build).size = n;

  // Use multiple GPUs?
  if (!strcmp(argv[2], "--sgpu")) {
      (*build).multigpu = false;
  } else if (!strcmp(argv[2], "--mgpu")) {
      (*build).multigpu = true;
  } else {
    usage(argv);
  }

  return;
}

int main(int argc, char** argv)
{
    CommandLineArgs * build = new CommandLineArgs;
    input(argc, argv, build);

    if (!(*build).multigpu) SingleGPUApplication((*build).size);
    else MultiGPUApplication((*build).size);
}

#endif // SCHEDULER_CU
