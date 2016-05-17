#Multi-GPU Scheduler
Gets information on CUDA capable devices available on a workstation/server,
and finds the optimal distribution of a workload. Schedule should dynamically
work for both single and multi-gpu architecture.

- `CMakeLists.txt` - to support cross platform builds.
- `scheduler.cu`  - use as main file to schedule the parameters.
- `singlegpu.cuh` - a simple example of floating point multiply add operation on
single gpu.
- `multiplegpus.cuh` - I haven't debugged this, it assumes that we're only using 2 gpus.

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

##How to contribute? (optional)
- `fork` using GitHub; https://github.com/neoblizz/mgpuscheduler
- Using command line `cd` into a directory you'd like to work on.
- `git clone https://github.com/neoblizz/mgpuscheduler.git`
- `git remote set-url -push origin https://github.com/username/mgpuscheduler.git` This sets the url of the push command to your `username` repository that you forked. That way we can create pull request and make sure nothing accidentally breaks in the main repo. Be sure to change the `username` to your username in the command.
- Make changes to the file you'd like.
- `git add filename`
- `git commit -m "comment here"`
- `git push` You'll be prompted for username and password for your github.
- Once you've pushed the changes on your fork, you can create a pull request on Github to merge the changes.
