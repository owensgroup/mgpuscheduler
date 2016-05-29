#ifndef BATCH_RUN_H
#define BATCH_RUN_H

#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

#include "scheduler.cuh"

class MultiplyAdd : public ScheduledKernel
{
public:
  MultiplyAdd()
    : m_hA(NULL), m_hB(NULL), m_hC(NULL), m_hCheckC(NULL), m_dA(NULL), m_dB(NULL), m_dC(NULL)
  {}

  ~MultiplyAdd()
  {
    FreeHostMemory();
    FreeDeviceMemory();
  }

  void FreeHostMemory()
  {
    if (m_hA) free(m_hA);
    if (m_hB) free(m_hB);
    if (m_hC) free(m_hC);
    if (m_hCheckC) free(m_hCheckC);
    m_hA = m_hB = m_hC = m_hCheckC = NULL;
  }

  void FreeDeviceMemory()
  {
    if (m_dA) ERROR_CHECK(cudaFree(m_dA));
    if (m_dB) ERROR_CHECK(cudaFree(m_dB));
    if (m_dC) ERROR_CHECK(cudaFree(m_dC));
    m_dA = m_dB = m_dC = NULL;
  }

  void InitializeData(int vectorSize, int threadsPerBlock, int kernelNum);
  bool VerifyResult();

  virtual int  AcquireDeviceResources(std::vector< DeviceInfo > *deviceInfo);
  virtual void ReleaseDeviceResources(std::vector< DeviceInfo > *deviceInfo);
  void FinishHostExecution();

  int   m_vectorSize;                     // Number of elements per vector
  float *m_hA, *m_hB, *m_hC, *m_hCheckC;  // Host vectors
  float *m_dA, *m_dB, *m_dC;              // Device vectors

  unsigned long long  m_globalMemRequired;  // The amount of global device memory required
  int m_blocksRequired;                     // Blocks required, first-dimension only (for the 3-D blocks per grid)

  int m_kernelNum, m_deviceNum;                  // Kernel num and GPU device this kernel executed on
  float m_queueTimeMillisec, m_execTimeMillisec; // Timers for timing this kernel

  cudaStream_t m_stream;  // Stream for asynchrnous execution - assumes only one GPU for this kernel
  cudaEvent_t m_startQueueEvent, m_startExecEvent, m_finishExecEvent; // Events for timing
};

class BatchMultiplyAdd
{
public:
  BatchMultiplyAdd(int meanVectorSize, int batchSize, int numDevices, int threadsPerBlock, int numCPUThreads)
    : m_meanVectorSize(meanVectorSize), m_batchSize(batchSize), m_numDevices(numDevices), 
      m_threadsPerBlock(threadsPerBlock), m_numCPUThreads(numCPUThreads)
  {}

  ~BatchMultiplyAdd()
  {
    for (auto it = m_data.begin(); it != m_data.end(); ++it)
      if (*it) delete *it;
  }

  void RunExperiment();

private:
  void GenerateData();

  std::vector< MultiplyAdd* > m_data;   // Data for each run of MultiplyAdd
  int m_meanVectorSize, m_batchSize;    // Run-time parameters for the data
  int m_numDevices, m_threadsPerBlock;  // Run-time parameters for the GPU(s)
  int m_numCPUThreads;                  // Run-time parameters for the CPU theading
};

#endif // #ifndef BATCH_RUN_H