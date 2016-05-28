#ifndef MULTIPLY_ADD_H
#define MULTIPLY_ADD_H

#include <cstdlib>

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
  void InitializeHostData(int vectorSize)
  {
    m_vectorSize = vectorSize;

    m_hA = (float*)malloc(sizeof(float) * vectorSize);
    m_hB = (float*)malloc(sizeof(float) * vectorSize);
    m_hC = (float*)malloc(sizeof(float) * vectorSize);
    m_hCheckC = (float*)malloc(sizeof(float) * vectorSize);

    // Fill in A and B with random numbers (should be seeded prior to call)
    float invRandMax = 1000.0f / RAND_MAX; // Produces random numbers between 0 and 1000
    for (int n = 0; n < vectorSize; ++n)
    {
      m_hA[n] = std::rand() * invRandMax;
      m_hB[n] = std::rand() * invRandMax;
    }
  }

  int m_vectorSize;
  float *m_hA, *m_hB, *m_hC, *m_hCheckC; // Host vectors
  float *m_dA, *m_dB, *m_dC; // Device vectors
};

#endif // #ifndef MULTIPLY_ADD_H