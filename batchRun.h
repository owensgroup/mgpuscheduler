#ifndef BATCH_RUN_H
#define BATCH_RUN_H

#include <cstdlib>
#include <random>

#include "deviceInfo.cuh"

class BatchRun
{
public:
  void GetDeviceInfo();
  int GetFreeDevice(const unsigned long long &globalMemRequired,
                    const int &threadsRequired, const int &blocksRequired);

  std::vector< DeviceInfo > m_deviceInfo;
};

class MultiplyAdd
{
public:
  MultiplyAdd()
    : m_hA(NULL), m_hB(NULL), m_hC(NULL), m_hCheckC(NULL), m_dA(NULL), m_dB(NULL), m_dC(NULL)
  {}

  ~MultiplyAdd()
  {
    if (m_hA) free(m_hA);
    if (m_hB) free(m_hB);
    if (m_hC) free(m_hC);
    if (m_hCheckC) free(m_hCheckC);
    if (m_dA) free(m_dA);
    if (m_dB) free(m_dB);
    if (m_dC) free(m_dC);
  }

  void InitializeHostData(int vectorSize, int threadsPerBlock);

  int m_vectorSize;
  float *m_hA, *m_hB, *m_hC, *m_hCheckC; // Host vectors
  float *m_dA, *m_dB, *m_dC; // Device vectors

  unsigned long long m_globalMemRequired;
  int m_threadsRequired; // First-dimension only (for the 3-D threads per block)
  int m_blocksRequired; // First-dimension only (for the 3-D blocks per grid)
};

class BatchMultiplyAdd : public BatchRun
{
public:
  BatchMultiplyAdd(int meanVectorSize, int batchSize, int numDevices, int threadsPerBlock)
    : m_meanVectorSize(meanVectorSize), m_batchSize(batchSize), 
      m_numDevices(numDevices), m_threadsPerBlock(threadsPerBlock)
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
};



#endif // #ifndef BATCH_RUN_H