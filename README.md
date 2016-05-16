#Multi-GPU Scheduler
Gets information on CUDA capable devices available on a workstation/server,
and finds the optimal distribution of a workload. Schedule should dynamically
work for both single and multi-gpu architecture.

- `CMakeLists.txt` - to support cross platform builds.
- `scheduler.cu`  - use as main file to schedule the parameters.
- `singlegpu.cuh` - a simple example of floating point multiply add operation on
single gpu.

##Compile
If you add more files, make sure to create the appropriate `CUDA_ADD_EXECUTABLE`
for them. Name set for `CUDA_ADD_EXECUTABLE` will be the name of binary executable
file.

```
mkdir build
cd build
cmake ..
make
```
