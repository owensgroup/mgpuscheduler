#ifndef BATCH_MTX_H
#define BATCH_MTX_H

#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

#include "scheduler.cuh"

typedef std::chrono::time_point<std::chrono::high_resolution_clock> Time;

class MatrixMultiply : public ScheduledKernel
{
public:
  MatrixMultiply()
    : m_hA(NULL), m_hB(NULL), m_hC(NULL), m_hCheckC(NULL), m_dA(NULL), m_dB(NULL), m_dC(NULL),
    m_memBytesReadWrite(0), m_floatingPointOps(0)
  {}

  ~MatrixMultiply()
  {
    FreeHostMemory();
    FreeDeviceMemory();
  }

  void FreeHostMemory();
  void FreeDeviceMemory();

  void InitializeData(int matrixSize, int blockWidth, int kernelNum);
  void FinishHostExecution(bool freeHostMemory);

  virtual int  AcquireDeviceResources(std::vector< DeviceInfo > *deviceInfo);
  virtual void ReleaseDeviceResources(std::vector< DeviceInfo > *deviceInfo);

  int   m_matrixSize;                     // Number of elements per matrix
  int   m_blockWidth;                     // Number of elements per blocked multiply
  // float **m_hA, **m_hB, **m_hC, **m_hCheckC;  // Host vectors (2d pointers)
  float *m_hA, *m_hB, *m_hC, *m_hCheckC;  // Host vectors (1d pointers)
  float *m_dA, *m_dB, *m_dC;              // Device vectors

  std::size_t  m_globalMemRequired;  // The amount of global device memory required
  int m_blocksRequired;                     // Blocks required, first-dimension only (for the 3-D blocks per grid)

  float m_floatingPointOps;   // Number of floating point operations used to execute this kernel
  float m_MFLOPs;             // Number of MFLOPs per second for this kernel
  float m_memBytesReadWrite;  // Number of bytes of memory read and written to execute this kernel
  float m_MBps;               // Number of MB per second for this kernel

  int m_kernelNum, m_deviceNum;                  // Kernel num and GPU device this kernel executed on
  float m_queueTimeMS, m_kernelExecTimeMS, m_totalExecTimeMS; // Timers for timing this kernel (NS = nanoseconds, MS = microseconds)

  Time m_queueStarted, m_streamStarted, m_streamFinished; // CPU timers for this kernel stream

  cudaStream_t m_stream;  // Stream for asynchrnous execution - assumes only one GPU for this kernel
  cudaEvent_t m_startExecEvent, m_finishExecEvent; // Events for timing queue and kernel execution
  cudaEvent_t m_startCudaMallocEvent, m_finishDownloadEvent; // Events for timing the total execution, malloc to download
};

class BatchMatrixMultiply
{
public:
  BatchMatrixMultiply(int meanMatrixSize, int batchSize, int blockWidth)
    : m_meanMatrixSize(meanMatrixSize), m_blockWidth(blockWidth), m_batchSize(batchSize)
  {}

  ~BatchMatrixMultiply()
  {
    for (auto it = m_data.begin(); it != m_data.end(); ++it)
      if (*it) delete *it;
  }

  void RunExperiment(const std::string &kernelName, int numRepeat);
  void ComputeBatchResults();
  void OutputResultsCSV(const std::string &kernelName);
  friend void RunKernelThreaded(BatchMatrixMultiply *batch, int kernelNum);

private:
  void GenerateData();

  std::vector< MatrixMultiply* > m_data;   // Data for each run of MatrixMultiply
  int m_meanMatrixSize, m_blockWidth, m_batchSize;    // Run-time parameters for this kernel

  float m_batchTotalExecTimeMS;   // Time from first kernel data cudaMalloc to last kernel data downloaded

  float m_batchFloatingPointOps; // Total floating point ops for this batch
  float m_batchGFLOPs;  // Total GFLOP/s for this batc

  float m_batchMemBytesReadWrite; // Total bytes of memory read and written for this batch
  float m_batchGBps;   // Total GB/s for this batch
};

#endif // #ifndef BATCH_MTX_H
