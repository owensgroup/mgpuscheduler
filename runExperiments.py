#!/usr/bin/python

import sys, subprocess, argparse
from timeit import default_timer as timer

def main(argv):

   # See https://docs.python.org/2/howto/argparse.html for more details
   parser = argparse.ArgumentParser()
   parser.add_argument("executable", help="The scheduler executable path")
   args = parser.parse_args()

   # Debugging experiment values
   threadsPerBlockList = [1024]
   #vectorSizeList = [1024, 16384, 65536]
   vectorSizeList = [1024]
   matrixSizeList = [32]
   maxDevicesList = [2]
   batchSizeList = [128]
   #batchSizeList = [128, 512, 1024]

   # Real experiment values
   #threadsPerBlockList = [256, 512, 1024]
   #vectorSizeList = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144] # MUST be same number of elements as matrixSizeList
   #matrixSizeList = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] # MUST be same number of elements as vectorSizeList
   #maxDevicesList = [1, 2]
   #batchSizeList = [32, 64, 128, 256, 512, 1024]
   maxGPUsPerKernel = 1
   verboseFlag = 1 # Use 1 to debug for now, then 0 to run
   
   # "Usage: <ExecPath> meanVectorSize batchSize maxDevices threadsPerBlock maxGPUsPerKernel kernelName verboseOutput\n"
   # NOTE: Use timer only for output, not for actual timing (which is done in the program itself)
   for inputIndex in range(len(vectorSizeList)):
      startInputSize = timer()
      for batchSize in batchSizeList:
         startBatchSize = timer()
         for threadsPerBlock in threadsPerBlockList:
            for maxDevices in maxDevicesList:
               # First, run MultiplyAdd
               vectorSize = vectorSizeList[inputIndex]
               print('Input: {0}, Batch: {1}, Threads: {2}, GPUs: {3}...'.format(vectorSize, batchSize, threadsPerBlock, maxDevices))
               startCall = timer() 
               subprocess.call([args.executable, str(vectorSize), str(batchSize), str(maxDevices), str(threadsPerBlock), str(maxGPUsPerKernel), 'MultiplyAdd', str(verboseFlag)]) 
               endCall = timer()
               print('\t... Done, {0}s'.format(endCall-startCall))

               # Second, run MatrixMultiply
               #matrixSize = matrixSizeList[inputIndex]
               #print('Input: {0}, Batch: {1}, Threads: {2}, GPUs: {3}...'.format(matrixSize, batchSize, threadsPerBlock, maxDevices))
               #startCall = timer() 
               #subprocess.call([args.executable, str(matrixSize), str(batchSize), str(maxDevices), str(threadsPerBlock), str(maxGPUsPerKernel), 'MatrixMultiply', str(verboseFlag)]) 
               #endCall = timer()
               #print('\t... Done, {0}s'.format(endCall-startCall))

         endBatchSize = timer()
         print('------------------------------------------------')
         print('BatchSize {0} finished in {1}s'.format(batchSize, endBatchSize-startBatchSize))
         print('------------------------------------------------')
      endInputSize = timer()
      print '================================================'
      print('VectorSize {0} and MatrixSize {1} finished in {2}s'.format(vectorSizeList[inputIndex], matrixSizeList[inputIndex], endInputSize-startInputSize))
      print('================================================')

if __name__ == "__main__":
   main(sys.argv[1:])