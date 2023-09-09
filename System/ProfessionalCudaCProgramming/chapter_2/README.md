# Chapter 2 Professional CUDA C Programming

CUDA is a parallel computing platform and programming model with a small set of
extension to the C language.

## 2.1 Introducing the CUDA Programming Model

The CUDA programming model provides the following special features to harness the
computing power of GPU architectures

+ A way to organize threads on the GPU through a hierarchy structure.
+ A way to access memory on the GPU through a hierarchy structure.

### 2.1.1 CUDA Programming Structure

The CUDA programming model enables you to execute applications on heterogeneous
computing systems by simply annotating code with a small set of extensions to the C
programming language. A heterogeneous environment consists of CPUs complemented by
GPUs, each with its own memory separated by a PCI-Express bus. Therefore, you should
note the following distinction:

+ **Host**: the CPU and its memory (host memory).
+ **Device**: the GPU and its memory (device memory).

Starting with CUDA 6, NVIDIA introduced a programming model improvement called
*Unified Memory*, which bridges the divide between host and device memory spaces.
This improvement allows you to access both the CPU and GPU memory using a single
pointer, while the system automatically migrates data between the host and device.

A key component of the CUDA programming model is the kernel, the code that runs
on the GPU device. Behind the scenes, CUDA manages scheduling programmer-written
kernels on GPU threads. From the host, you define how your algorithm is mapped
to the device based on application data and GPU device capability.

The host can operate independently of the device for most operations. When a kernel
has been launched, control is returned immediately to the host, freeing the CPU to
perform additional tasks complemented by data parallel code running on the device.

A typical processing flow of a CUDA program follows this pattern:

1. Copy data from CPU memory to GPU memory.
2. Invoke kernels to operate on the data stored in GPU memory.
3. Copy data back from GPU memory to CPU memory.

### 2.1.2 Managing Memory

To allow you to have full control and achieve the best performance, the CUDA runtime
provides functions to allocate device memory, release device memory, and transfer data
between the host memory and device memory.

The function used to perform GPU memory allocation is `cudaMalloc`, and its function
signature is:

```c
cudaError_t cudaMalloc(void** devPtr, size_t size);
```

The function used to transfer data between the host and device is `cudaMemcpy`, and its
function signature is:

```c
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
```

This function copies the specified bytes from the source memory area, pointed to by `src`,
to the destination memory area, pointed to by `dst`, with the direction specified by
`kind`, where `kind` takes one of the following types:

+ `cudaMemcpyHostToHost`
+ `cudaMemcpyHostToDevice`
+ `cudaMemcpyDeviceToHost`
+ `cudaMemcpyDeviceToDevice`

This function exhibits synchronous behavior because the host application blocks until
`cudaMemcpy` returns and the transfer is complete. Every CUDA call, except kernel launches,
returns an error code of an enumerated type `cudaError_t`. For example, if GPU memory is
successfully allocated, it returns `cudaSuccess`. Otherwise, it returns
`cudaErrorMemoryAllocation`. You can convert an error code to a human-readable error message
with the following CUDA runtime function:

```c
char * cudaGetErrorString(cudaError_t error);
```

### 2.1.3 Organizing Threads

When a kernel function is launched from the host side, execution is moved to a device where
a large number of threads are generated and each thread executes the statements specified
by the kernel function.

All threads spawned by a single kernel launch are collectively called a *grid*. All threads
in a grid share the same global memory space. A grid is made up of many thread blocks. A
thread block is a group of threads that can cooperate with each other using:

+ Block-local synchronization.
+ Block-local shared memory.

Threads rely on the following two unique coordinates to distinguish themselves from each other:

+ `blockIdx` (block index within a grid)
+ `threadIdx` (thread index within a block)

The coordinate variable is of type `uint3`, a CUDA built-in vector type, derived from the basic
integer type. It is a structure containing three unsigned integers, and the 1st, 2nd, and 3rd
components are accessible through the fields `x`, `y` and `z` respectively.

CUDA organizes grids and blocks in three dimensions. The dimensions of a grid and a block
are specified by the following two built-in variables:

+ `blockDim` (block dimension, measured in threads)
+ `gridDim` (grid dimension, measured in blocks)

These variables are of type `dim3`, an integer vector type based on `uint3` that is used to
specify dimensions. When defining a variable of type `dim3`, any component left unspecified
is initialized to 1. Each component in a variable of type `dim3` is accessible through its
`x`, `y`, and `z` fields.

Here, we give an example here:

[checkDimension.cu](./checkDimension.cu)

For a given data size, the general steps to determine the grid and block dimensions are:

+ Decide the block size.
+ Calculate the grid dimension based on the application data size and block size.

To determine the block dimension, you usually need to consider:

+ Performance characteristics of the kernel.
+ Limitations on GPU resources.

### 2.1.4 Launching a CUDA Kernel

A CUDA kernel call is a direct extension to the C function syntax that adds a kernel's
*execution configuration* in side triple-angle-brackets:

```c
kernel_name<<<grid, block>>>(argument list);
```

A kernel call is asynchronous with respect to the host thread. After a kernel is invoked,
control returns to the host side immediately. You can call the following function to force
the host application to wait for all kernels to complete.

```c
cudaError_t cudaDeviceSynchronize();
```

### 2.1.5 Writing Your Kernel

A kernel function is the code to be executed on the device side. In a kernel function, you
define the computation for a *single* thread, and the data access for that thread. When the
kernel is called, many different CUDA threads perform the same computation in parallel. A
kernel is defined using the `__global__` declaration specification. A kernel function must
have a `void` return type.

Below is a summary of function type qualifiers used in CUDA C programming. Function qualifiers
specify whether a function executes on the host or on the device and whether it is callable
form the host or from the device.

+ `__global__`: executed on the device, callable from the host or from the device for devices
of compute capability 3.
+ `__device__`: executed on the device, callable from the device only.
+ `__host__`: executed on the host, callable from the host only.

The `__device__` and `__host__` qualifiers can be used together, in which case the function
is compiled for both the host and the device.

## 2.2 Organizing Parallel Threads

For matrix operations, a natural approach is to use a layout that contains a 2D grid with
2D blocks to organize the threads in your kernel.

Typically, there are three kinds of indices for a 2D case you need to manage:

+ Thread and block index.
+ Coordinate of a given point in the matrix.
+ Offset in linear global memory.

In the first step, you can map the thread and block index to the coordinate of a matrix
with the following formula:

```c++
ix = threadIdx.x + blockIdx.x * blockDim.x
iy = threadIdx.y + blockIdy.y * blockDim.y
```

In the second step, you can map a matrix coordinate to a global memory location/index
with the following formula.

```c++
idx = iy * nx + ix
```

Below illustrates the corresponding relationship among block and thread indices,
matrix coordinates, and linear global memory indices.

![Matrix coordinate](https://s2.loli.net/2023/09/05/84ICtMbB39RwKi7.png)

### 2.2.2 Summing Matrices with a 2D Gird and 2D Blocks

First, a validation host function should be written to verify that the matrix
addition kernel produces the correct results:

```c++
void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
  float *ia = A;
  float *ib = B;
  float *ic = C;

  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      ic[ix] = ia[ix] + ib[ix];
    }
    ia += nx; ib += nx; ic += nx;
  }
}
```

Then you create a new kernel to sum the matrix with a 2D thread block:

```c++
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny) {
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy * nx + ix;

  if (ix < nx && iy < ny) {
    MatC[idx] = MatA[idx] + MatB[idx];
  }
}
```

## 2.3 Managing Devices

NVIDIA provides several means by which you can query and manage GPU devices. It is
important to learn how to query this information.

There are two basic and powerful means to query and manage GPU devices:

+ The CUDA runtime API functions.
+ The NVIDIA Systems Management Interface (`nvidia-smi`) command-line utility
