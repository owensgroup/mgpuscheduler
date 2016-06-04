#!/usr/bin/python

import sys, subprocess, argparse
from timeit import default_timer as timer

def main(argv):

   # Example execution:
   # python runExperiments.py D:\Research\GPUScheduler\build\Release\sched.exe
   
   parser = argparse.ArgumentParser()
   parser.add_argument("executable", help="The scheduler executable path") # See https://docs.python.org/2/howto/argparse.html for more details
   args = parser.parse_args()

   # Real experiment values
   threadsPerBlockList = [256, 512, 1024]
   blockWidthList = [8, 16, 32] 
   vectorSizeList = [32768, 65536, 131072, 262144, 524288, 1048576] # MUST be same number of elements as matrixSizeList
   matrixSizeList = [128, 256, 512, 1024, 2048, 4096] # MUST be same number of elements as vectorSizeList
   maxDevicesList = [1, 2]
   batchSizeList = [8, 16, 32, 64, 128, 256, 512]
   numRepeat = 1
   verboseFlag = 1 # Use 1 to debug for now, then 0 to run
   
   # "Usage: <ExecPath> inputSize batchSize maxDevices kernelName kernelArgument verboseOutput\n"
   # NOTE: Use timer only for output, not for actual timing (which is done in the program itself)
   for inputIndex in range(len(vectorSizeList)):
      startInputSize = timer()
      for batchSize in batchSizeList:
         startBatchSize = timer()
         for kernelArgumentIndex in range(len(threadsPerBlockList)):
            for maxDevices in maxDevicesList:
               # First, run MultiplyAdd
               vectorSize = vectorSizeList[inputIndex]
               threadsPerBlock = threadsPerBlockList[kernelArgumentIndex]
               print('Input: {0}, Batch: {1}, Threads: {2}, GPUs: {3}...'.format(vectorSize, batchSize, threadsPerBlock, maxDevices))
               startCall = timer() 
               subprocess.call([args.executable, str(vectorSize), str(batchSize), str(maxDevices), 'MultiplyAdd', str(threadsPerBlock), str(numRepeat), str(verboseFlag)]) 
               endCall = timer()
               print('\t... Done, {0}s'.format(endCall-startCall))

               # Second, run MatrixMultiply
               matrixSize = matrixSizeList[inputIndex]
               blockWidth = blockWidthList[kernelArgumentIndex]
               print('Input: {0}, Batch: {1}, Threads: {2}, GPUs: {3}...'.format(matrixSize, batchSize, blockWidth, maxDevices))
               startCall = timer() 
               subprocess.call([args.executable, str(matrixSize), str(batchSize), str(maxDevices), 'MatrixMultiply', str(blockWidth), str(numRepeat), str(verboseFlag)]) 
               endCall = timer()
               print('\t... Done, {0}s'.format(endCall-startCall))

         endBatchSize = timer()
         print('------------------------------------------------')
         print('BatchSize {0} finished in {1}s'.format(batchSize, endBatchSize-startBatchSize))
         print('------------------------------------------------')
      endInputSize = timer()
      print('================================================')
      print('VectorSize {0} and MatrixSize {1} finished in {2}s'.format(vectorSizeList[inputIndex], matrixSizeList[inputIndex], endInputSize-startInputSize))
      print('================================================')

if __name__ == "__main__":
   main(sys.argv[1:])