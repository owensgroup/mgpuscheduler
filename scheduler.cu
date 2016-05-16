#ifndef SCHEDULER_CU
#define SCHEDULER_CU

#include <string.h>

#include <iostream>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "singlegpu.cuh"

int main(int argc, char** argv)
{
    int n = 1<<20;
    cudaSetDevice(0);
    SingleGPUApplication(n);
}

#endif // SCHEDULER_CU
