project(mgpuscheduler)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

CUDA_ADD_EXECUTABLE(sched main.cpp scheduler.cu scheduler.cuh multiplyAdd.cu multiplyAdd.cuh deviceInfo.cuh)
list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fopenmp -fgomp) #-Xcompiler says flag is for C++ compiler, not nvcc