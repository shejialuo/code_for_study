# 6. GPU programming with CUDA

## 6.1 GPUs and GPGPU

+ GPU:graphics processing units.
+ GPGPU: General Purpose computing on GPUs.

Currently the most widely used APIs for GPGPU are CUDA and OpenCL.
CUDA was developed for use on Nvidia GPUs. OpenCL, on the other hand,
was designed to be highly portable.

## 6.2 GPU architectures

A typical GPU can be thought of as being composed of one or more
SIMD processors. Nvidia GPUs are composed of *Streaming Multiprocessors* or *SM*s.
One SM can have several control units and many more datapaths.
So an SM can be thought of as consisting of one or more SIMD processors.
The SMs, however, operate asynchronously: there is no penalty if one
branch of an `if-else` executes on one SM, and the other executes
on another SM.

In Nvidia parlance, the datapaths are called cores, *Streaming Processors*,
or *SP*s. Also note that Nvidia uses the term *SIMT* instead of
SIMD. SIMT stands for Single Instruction Multiple Thread,
and the term is used because threads on an SM that are executing
the same instruction may not execute simultaneously: to hide memory
access latency, some threads may block while memory is accessed and
other threads, that have already accessed the data, may proceed with execution.

Each SM has a relatively small block of memory that is shared among its SPs.
This memory can be accessed very quickly be the SPs. All of the SMs on
a single chip also have access to a much larger block of memory among
all the SPs. Accessing this memory is relatively slow.

![Simplified block diagram of a GPU](https://s2.loli.net/2022/09/07/j2UMxfWPAkNE1s6.png)

The GPU and its associated memory are usually physically separate
from the CPU and its associated memory. In Nvidia documentation,
the CPU together with its associated memory is often called the *host*,
and the GPU together with its memory is called the *device*.

![Simplified block diagram of a CPU and a GPU](https://s2.loli.net/2022/09/07/D3HQplGtBWmVsj8.png)

## 6.3 Heterogeneous computing

Writing a program that runs on a GPU is an example of *heterogeneous* computing.
The reason is that the programs make use of both a host processor and a GPU.

## 6.4 CUDA hello

So let's start talking about the CUDA API, the API we'll be using
to program heterogeneous CPU-GPU systems.

### 6.4.1 The source code

```c
#include <stdio.h>
#include <cuda.h>

/* Device code: runs on GPU */
__global__ void Hello(void) {
  printf("Hello from thread %d!\n", threadIdx.x);
}

/* Host code: Runs on CPU */
int main(int argc, char* argv[]) {
  int thread_count;
  thread_count = strtol(argv[1], NULL, 10);

  /* Start thread_count threads on GPU */
  Hello <<<1, thread_count>>>();

  cudaDeviceSynchronize(); // Wait for GPU to finish

  return 0;
}
```

### 6.4.2 Compiling and running the program

```sh
nvcc -o cuda_hello cuda_hello.cu
./cuda_hello 10
```

## 6.5 A closer look

Notice that our kernel code uses the SPMD paradigm: each thread
runs a copy of the same code on its own data.

To summarize:

+ Execution begins in `main`, which is running on the host.
+ The number of threads is taken from the command line
+ The call to `Hello` starts the kernel
  + The `<<<1, thread_count>>>` in the call specifies that
  `thread_count` copies of the kernel should be started on the device.
  + When the kernel is started, the struct `threadIdx` is initialized
  by the system.
+ The call to `cudaDeviceSynchronize` in `main` forces the host
to wait until all of the threads have complete kernel execution
before continuing and terminating.

## 6.6 Threads, blocks, and grids

You're probably wondering why we put a "1" in the angle brackets
in our call to `Hello`:

```c
Hello <<<1, thread_count >>>();
```

When a CUDA kernel runs, each individual thread will execute its code
on an SP. With "1" as the first value in angle brackets, all
of the threads that are started by the kernel call will run on
a single SM. If our GPU has two SMs, we can try to use both of
them with the kernel call.

```c
Hello <<<2, thread_count/2>>> ();
```

However, what if `thread_count` is odd. CUDA organizes threads into
blocks and grids. A *thread block* is a collection of threads that
run a single SM. In a kernel call the first value in the angle
bracket specifies the number of thread blocks. The second value is
the number of threads in each thread block.

A *grid* is just the collection of thread blocks started by a kernel.

There are several built-in variables that a thread can use to
get information on the grid started by the kernel. The following four
variables are structs that are initialized in each thread's memory when
a kernel begins execution:

+ `threadIdx`: the rank of index of the thread in its thread block.
+ `blockDim`: the dimensions, shape, or size of the thread blocks.
+ `blockIdx`: the rank or index of the block within the grid.
+ `gridDim`: the dimensions, shape, or size of the grid.

All of these structs have three fields, `x`, `y`, and `z`, and
the fields all have unsigned integer types.

When we call a kernel with something like:

```c
int blk_ct, the_per_blk;
Hello <<<blk_ct, the_per_blk>>>();
```

The three-element structures `gridDim` and `blockDim` are initialized
by assigning the values in angle brackets to the `x` fields.

```c
girdDim.x = blk_ct;
blockDim.x = the_per_blk;
```

The `y` and `z` fields are initialized to 1. If we want to use
values other than 1 for the `y` and `z` fields, we should declare
two variables of type `dim3`, and pass them into the call to the kernel.

```c
dim3 grid_dims, block_dims;
grid_dims.x = 2;
grid_dims.y = 3;
grid_dims.z = 1;
block_dims.x = 4;
block_dims.y = 4;
block_dims.z = 4;
Kernel <<<grid_dims, block_dims>>>(...);
```

## 6.7 Vector addition

```c
__global__ void Vec_add(
  const float x[],
  const float y[],
  float z[],
  const int n) {
  // every thread in each block handles one addition
  int my_elt = blockDim.x * blockIdx.x + threadIdx.x;
  // total thread may be > n
  if (my_elt < n) {
    z[my_elt] = x[my_elt] + y[my_elt]
  }
}

int main(int argc, char* argv[]) {
  int n, th_per_blk, blk_ct;
  char i_g;
  float *x, *y, *z, *cz;
  double diff_norm;

  Get_args(argc, argv, &n, &blk_ct, &th_per_blk, &i_g);
  Allocate_vectors(&x, &y, &z, &cz, n);
  Init_vectors(x, y, n, i_g);

  Vec_add <<<blk_ct, th_per_blk>>>(x, y, z, n);

  // Check for correctness
  Serial_vec_add(x, y, cz, n);
  diff_norm = Two_norm_diff(z, cz, n);
  printf("Two-norm of difference between host and ");
  printf("device = %e\n", diff_norm);
  Free_vectors(x, y, z , cz);
  return 0;
}
```

### 6.7.1 The Kernel

In the kernel, we use data-parallel thinking, every thread handles one addition.

Also, we make a `Serial_vec_add` to do serial addition.

```c
void Serial_vec_add(
  const float x[],
  const float y[],
  float cz[],
  const int n) {
  for(int i = 0; i < n; ++i) {
    cz[i] = x[i] + y[i];
  }
}
```

### 6.7.2 Get_args

```c
void Get_args(
  const int argc,
  char *argv[],
  int* n_p,
  int* blk_ct_p,
  int* th_per_blk_p,
  char* i_g) {
  if(argc != 5) {
    exit(-1);
  }
  *n_p = strtol(argv[1], NULL, 10);
  *blk_ct_p = strtol(argv[2], NULL, 10);
  *th_per_blk_p = strtol(argv[3], NULL, 10);
  *i_g = argv[4][0];

  if(*n_p > (*blk_ct_p) * (*th_per_blk_p)) {
    exit(-1);
  }
}
```

### 6.7.3 Allocate_vectors and managed memory

```c
void Allocate_vectors(
  float** x_p,
  float** y_p,
  float** z_p,
  float** cz_p,
  int n) {
  cudaMallocManaged(x_p, n * sizeof(float));
  cudaMallocManaged(y_p, n * sizeof(float));
  cudaMallocManaged(z_p, n * sizeof(float));

  /* cz is only used on host */
  *cz_p = (float*)malloc(n * sizeof(float));
}
```

### 6.7.4 Other functions called from main

```c
double Two_norm_diff(
  const float z[],
  const float cz[],
  const int n) {
  double diff, sum = 0.0;
  for(int i = 0; i < n; ++i) {
    diff = z[i] - cz[i];
    sum += diff * diff;
  }
  return sqrt(sum);
}

void Free_vectors(
  float* x,
  float* y,
  float* z,
  float* cz) {
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);

  free(cz);
}

void Init_vectors(
  float* x,
  float* y,
  int n,
  char i_g) {
  srand((unsigned)time(NULL));
  for(int i = 0; i <n; ++i) {
    x[i] = 0 + 1.0 * rand() / RAND_MAX * 100
    y[i] = 0 + 1.0 * rand() / RAND_MAX * 100
  }
}
```

## 6.8 Returning results from CUDA kernels

There are several things that you should be aware of regarding
CUDA kernels. FIrst, they always have return type `void`, so
they can't be used to return a value. They also can't return anything to the
host through the standard C pass-by-reference. The reason for
this is that addresses on the host are, in most systems, invalid on the
device, and vice versa.

There are several possible approaches to "returning" a result to
the host from a kernel. One is to declare pointer variables and
allocate a single memory location. On a system that supports unified
memory, the computed value will be automatically copied back
to host memory.

## 6.9 CUDA trapezoidal rule 1

### 6.9.1 The trapezoidal rule

```c
float Serial_trap(
  const float a,
  const float b,
  const int n) {
  float x, h = (b - a) / n;
  float trap = 0.5 * (f(a) + f(b));

  for (int i = 1; i < n; ++i) {
    x = a + i * h;
    trap += f(x);
  }
  trap = trap * h;
  return trap;
}
```

### 6.9.2 A CUDA implementation

We're mainly interested in two types of tasks:

+ The evaluation of the function $f$ at $x_{i}$.
+ The addition of $f(x_{i})$ into `trap`.

This suggests that each thread in our CUDA implementation might
carry out one iteration of the serial `for` loop. We can assign
a unique integer rank to each thread as we did with the vector
addition program.

```c
int my_i = blockDim.x * blockIdx.x + threadIdx.x
float my_x = a + my_i * h;
float my_trap = f(my_x);
float trap += my_trap;
```

However, it's immediately obvious that there are several problems here:

+ We haven't initialized `h` or `trap`.
+ The `my_i` value can be too large or too small.
+ The variable `trap` must be shared among the threads. There
is a race condition.
+ The variable `trap` in the serial code is returned by the function.
+ We need to multiply the total in `trap` by `h` after all
of the threads have added their results.

```c
__global__ void Dev_trap(
  const float a,
  const float b,
  const float h,
  const int n,
  float* trap_p) {
  int my_i = blockDim.x * blockIdx.x + threadIdx.x;

  if (0 < my_i && my_i < n) {
    float my_x = a + my_i * h;
    float my_trap = f(my_x);
    // User atomicAdd to make sure that there is no race condition
    atomicAdd(trap_p, my_trap)
  }
}

void Trap_wrapper(
  const float a,
  const float b,
  const int n,
  float* trap_p,
  const int blk_ct,
  const int th_per_blk) {
  *trap_p = 0.5 * (f(a) + f(b));
  float h = (b - a) /n;
  Dev_trap <<<blk_ct, th_per_blk>>>(a, b, h, n, trap_p);
  cudaDeviceSynchronize();
  *trap_p = h *(*trap_p);
}
```

## 6.10 CUDA trapezoidal rule 2: improving performance

One way to improve the performance is to carry out a tree-structure global sum
that's similar to the tree-structured global sum we introduced
in the MPI chapter.

### 6.10.1 Tree-structured communication

There are two standard implementations of a tree-structured sum
in CUDA. One implementation uses shared memory, and in devices with
compute capability < 3 this is the best implementation. However,
in devices with compute capability >= 3 there are several functions
called *warp shuffles*, that allow a collection of threads
within a *warp* to read variables stored by other threads in the warp.

### 6.10.2 Local variables, registers, shared and global memory

We can think of the GPU memory as a hierarchy with three "levels".
At the bottom, is the slowest, largest level: global memory.
In the middle is a faster, smaller level: shared memory. At
the top is the fastest, smallest level: the registers.

### 6.10.3 Warps and warp

In particular, if we can implement a global sum in registers,
we expect its performance to be superior to an implementation
that uses shared or global memory, and the *warp shuffle* functions
introduced in CUDA 3.0 allow us to do this.

In CUDA a *warp* is a set of threads with consecutive ranks
belonging to a thread block. The number of threads in a warp is currently 32.
There is a variable initialized by the system that stores the size
of a warp:

```c
int warpSize;
```

The threads in a warp operate in SIMD fashion. So threads in different
warps can execute different statements with no penalty, while
threads within the same warp must execute the same statement.

The rank of a thread within a warp is called the thread's *lane*,
and it can be computed using the formula

```c
lane = threadIdx.x % warpSize;
```

The warp shuffle functions allow the threads in a warp to read
from registers used by another thread in the same warp.
