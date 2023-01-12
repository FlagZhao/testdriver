# 1 Compilation
For compilation, you can use commands below:
```
mkdir build
cd build
cmake ..
make
```
And then you will got a library file `libtestdriver.so`.

# 2 Test
We test this library on NVIDIA Orin platform. The test application is matixMul in [CUDA samples]()
```
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/compute-sanitizer:/usr/local/cuda/lib64/:${LD_LIBRARY_PATH}
LD_PRELOAD=/path/to/libtestdriver.so /path/to/matrixMul
```
An error named "function sanitizerMemcpyDeviceToHost failed with error The current operation cannot be performed" occurs from testdriver.c:262
