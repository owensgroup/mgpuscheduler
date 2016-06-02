#ifndef BATCH_MTX_H
#define BATCH_MTX_H

#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

#include "scheduler.cuh"

class MatrixMultiply : public ScheduledKernel
{
public:
  MatrixMultiply()
    : m_hA(NULL), m_hB(NULL), m_hC(NULL), m_hCheckC(NULL), m_dA(NULL), m_dB(NULL), m_dC(NULL)
  {}

  ~MatrixMultiply()
  {
    FreeHostMemory();
    FreeDeviceMemory();
  }


  void FreeHostMemory();
  void FreeDeviceMemory();

  float ** CreateMatrix(int m, int n);
  void InitializeData(int vectorSize, int threadsPerBlock, int kernelNum);
  void FinishHostExecution();

  virtual int  AcquireDeviceResources(std::vector< DeviceInfo > *deviceInfo);
  virtual void ReleaseDeviceResources(std::vector< DeviceInfo > *deviceInfo);

  int   m_vectorSize;                     // Number of elements per vector
  // float **m_hA, **m_hB, **m_hC, **m_hCheckC;  // Host vectors (2d pointers)
  float *m_hA, *m_hB, *m_hC, *m_hCheckC;  // Host vectors (1d pointers)
  float *m_dA, *m_dB, *m_dC;              // Device vectors

  unsigned long long  m_globalMemRequired;  // The amount of global device memory required
  int m_blocksRequired;                     // Blocks required, first-dimension only (for the 3-D blocks per grid)

  int m_kernelNum, m_deviceNum;                  // Kernel num and GPU device this kernel executed on
  float m_queueTimeMS, m_kernelExecTimeMS, m_totalExecTimeMS; // Timers for timing this kernel

  cudaStream_t m_stream;  // Stream for asynchrnous execution - assumes only one GPU for this kernel
  cudaEvent_t m_startQueueEvent, m_startExecEvent, m_finishExecEvent; // Events for timing queue and kernel execution
  cudaEvent_t m_startCudaMallocEvent, m_finishDownloadEvent; // Events for timing the total execution, malloc to download
};

class BatchMatrixMultiply
{
public:
  BatchMatrixMultiply(int meanVectorSize, int batchSize, int threadsPerBlock)
    : m_meanVectorSize(meanVectorSize), m_batchSize(batchSize), m_threadsPerBlock(threadsPerBlock)
  {}

  ~BatchMatrixMultiply()
  {
    for (auto it = m_data.begin(); it != m_data.end(); ++it)
      if (*it) delete *it;
  }

  void RunExperiment(const std::string &kernelName);
  void ComputeBatchResults();
  void OutputResultsCSV(const std::string &kernelName);
  friend void RunKernelThreaded(BatchMatrixMultiply *batch, int kernelNum);

private:
  void GenerateData();

  std::vector< MatrixMultiply* > m_data;   // Data for each run of MatrixMultiply
  int m_meanVectorSize, m_batchSize, m_threadsPerBlock;    // Run-time parameters for this kernel

  float m_batchKernelExecTimeMS;  // Time from first kernel execution started to last kernel execution finished
  float m_batchTotalExecTimeMS;   // Time from first kernel data cudaMalloc to last kernel data downloaded
};

#endif // #ifndef BATCH_MTX_H
