
# Multi-GPU Scheduler
Gets information on CUDA capable devices available on a workstation/server,
and finds the optimal distribution of a workload. Schedule should dynamically
work for both single and multi-gpu architecture.

**Important Note:** This scheduler does not ensure that the given kernel is optimized for best possible performance, it assumes that optimization techniques like loop unrolling or use of shared memory is in fact already implemented in the kernel.

## Compile
If you add more files, make sure to create the appropriate `CUDA_ADD_EXECUTABLE`
for them. Name set for `CUDA_ADD_EXECUTABLE` will be the name of binary executable
file.

```
mkdir build && cd build
cmake .. && make -j8
```
* Usage: `./sched meanVectorSize batchSize maxDevices threadsPerBlock maxGPUsPerKernel verboseOutput`

## Default Applications
We have provided two simple working CUDA applications to show how the scheduler handles workload.
* **Fused Multiply-Add:** As the name suggest, it is a very simple kernel that adds two randomly generated floats and multiplies them with another constant float.
* **Matrix Multiplication:** A slightly more complicated application is Matrix Multiplication on GPU, which uses shared memory resources as well as optimization techniques like block-access (maximizing memory bandwidth) and loop unrolling (maximizing arithmetic throughput). It also has a basic `MemSetKernel` to initialize working arrays in parallel.

## Adding your own Application to MGPUScheduler
Default applications mentioned above can be very easily modified to write your own application for the scheduler. The hierarchical structure of this project is as follows;

> **Main** - main.cpp used as the `int main(int argc, char** argv)` call to get the input arguments, which are easily changeable if your application requires more complicated I/O system. Scheduler is initialized with three simple commands, which gets all the necessary parameters of the system:
```
// Set the scheduler parameters
Scheduler::m_maxDevices = args.m_maxDevices;
Scheduler::m_maxGPUsPerKernel = args.m_maxGPUsPerKernel;
Scheduler::m_verbose = args.m_verbose;
```
Once scheduler is initialized, running the experiment for your application is done by creating the batch for the work and then actually running the experiment:
```
// Run the experiment for MatrixMultiply
BatchMatrixMultiply batchMtxMulti(args.m_meanVectorSize, args.m_batchSize, args.m_threadsPerBlock);
batchMtxMulti.RunExperiment(std::string("MatrixMultiply"));
```
>> **Scheduler** - scheduler.cu & scheduler.cuh is used to get the CUDA capable devices information installed and active in the system. Nothing is required to change in these files.
>>> **Fused Multiply-Add** - multiplyAdd.cu & multiplyAdd.cuh
>>> <br> **Matrix Multiplication** - matrixmultiply.cu & matrixmultiply.cuh
<br> This is where majority of your own application is created. The header file is used to define two classes per application `MatrixMultiply` and `BatchMatrixMultiply`, where one handles the core of application and the other handles the batch operation of said application. Required variables can also be initialized in the header file if they are suppose to be accessed by the entire class.
* `__global__ void GPUMatrixMultiply(const int WIDTH, float * A, float * B, float * C)` In this example, this is the actual kernel where matrix multiplication happens.
* `void MatrixMultiply::FreeHostMemory()` Used to free host memory.
* `void MatrixMultiply::FreeDeviceMemory()` Used to free device memory.
* `void MatrixMultiply::InitializeData(int vectorSize, int threadsPerBlock, int kernelNum)` Initialize host data in here, it can also be used to compute a CPU check.
* `int MatrixMultiply::AcquireDeviceResources(std::vector< DeviceInfo > *deviceInfo)` Find a device with enough resources, and if available, decrement the available resources and return the id.
* `void MatrixMultiply::ReleaseDeviceResources(std::vector< DeviceInfo > *deviceInfo)` Execution is complete, release the GPU resources for other threads.
* `void MatrixMultiply::FinishHostExecution()` Execution is complete. Record completion event and timers, verify result, and free host memory.
* `void BatchMatrixMultiply::GenerateData()` Generate data for the entire batch of MatrixMultiply's being run.
* `void BatchMatrixMultiply::ComputeBatchResults()`
* `void BatchMatrixMultiply::OutputResultsCSV(const std::string &kernelName)`
* `void BatchMatrixMultiply::RunExperiment(const std::string &kernelName)` Is where you define the experiment, including cudaMallocs, Memcopies, and kernel executions.



## How to contribute?
- `fork` using GitHub; https://github.com/owensgroup/mgpuscheduler
- Using command line `cd` into a directory you'd like to work on.
- `git clone https://github.com/owensgroup/mgpuscheduler.git`
- `git remote set-url -push origin https://github.com/username/mgpuscheduler.git` This sets the url of the push command to your `username` repository that you forked. That way we can create pull request and make sure nothing accidentally breaks in the main repo. Be sure to change the `username` to your username in the command.
- Make changes to the file you'd like.
- `git add filename`
- `git commit -m "comment here"`
- `git push` You'll be prompted for username and password for your github.
- Once you've pushed the changes on your fork, you can create a pull request on Github to merge the changes.
