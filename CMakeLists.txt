project(mgpuscheduler)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

list(APPEND CUDA_NVCC_FLAGS -std=c++11)

IF (UNIX)
    list(APPEND CMAKE_CXX_FLAGS -std=c++11)
ENDIF (UNIX)

CUDA_ADD_EXECUTABLE(sched main.cpp scheduler.cu scheduler.cuh multiplyAdd.cu multiplyAdd.cuh matrixmultiply.cu matrixmultiply.cuh)

IF (UNIX)
    TARGET_LINK_LIBRARIES(sched pthread)
ENDIF (UNIX)
