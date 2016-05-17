project(mgpuscheduler)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

CUDA_ADD_EXECUTABLE(sched scheduler.cu)
