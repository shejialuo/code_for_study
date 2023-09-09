# Chapter 3 CUDA Execution Model

## 3.1 Introducing the CUDA Execution Model

### 3.1.1 GPU Architecture Overview

The GPU architecture is built around a scalable array of *Streaming Multiprocessors*(SM).
GPU hardware parallelism is achieved through the replication of this architectural
building block.

+ CUDA cores
+ Shared Memory/L1 Cache
+ Register File
+ Load/Store Units
+ Special Function Units
+ Warp Scheduler

Each SM in a GPU is designed to support concurrent execution of hundreds of threads, and
there are generally multiple SMs per GPU. When a kernel grid is launched, the thread blocks
of that kernel grid are distributed among available SMs for execution. Once scheduled on
an SM, the threads of a thread block execute concurrently only on that assigned SM.
Multiple thread blocks may be assigned to the same SM at once and are scheduled based on
the availability of SM resources.

CUDA employs a *Single Instruction Multiple Thread*(SIMT) architecture to manage and
execute threads in groups of 32 called *warps*. All threads in a warp execute the same
instruction at the same time. Each thread has itw own instruction address counter and
register state, and carries out the current instruction on its own data.

A thread block is scheduled on only one SM. Once a thread block is scheduled on an
SM, it remains there until execution completes. An SM can hold more than one thread block
at the same time.

Shared memory and registers are precious resources in an SM. Shared memory is partitioned
among thread blocks resident on SM and registers are partitioned among threads. Threads
in a thread block can cooperate and communicate with each other through these resources.

Sharing data among parallel threads may cause a race condition: Multiple threads accessing
the same data with an undefined ordering, which results in unpredictable program
behavior. CUDA provides a means to synchronize threads with a thread block to ensure that
all threads reach certain points in execution before making further progress. However,
no primitives are provided for inter-block synchronization.

While warps within a thread block may be scheduled in any order, the number of active
warps is limited by SM resources. When a warp idles for any reason, the SM is free to
schedule another available warp from any thread block that is resident on the same SM.
Switching between concurrent warps has **no** overhead because hardware resources are
partitioned among all threads and blocks on an SM, so the state of the newly scheduled
warp is already stored on the SM.

### 3.1.2 Profile-Driven Optimization

*Profiling* is the act of analyzing program performance by measuring:

+ The space (memory) or time complexity of application code.
+ The use of particular instructions.
+ The frequency and duration of function calls.

Profiling is a critical step in program development, especially for optimizing HPC
application code. Profiling often requires a basic understanding of the execution
model of a platform to help make application optimization decisions. Developing
an HPC application usually involves two major steps:

+ Developing the code for correctness.
+ Improving the code for performance.

It's natural to use a profile-driven approach for the second step. Profile-driven
development is particularly important in CUDA programming:

+ A native kernel implementation generally does not yield the best performance.
Profiling tools can help you find the critical regions of your code that
are performance bottlenecks.
+ CUDA partitions the compute resources in an SM among multiple resident thread
blocks. This partitioning causes some resources to become performance limiters.
+ CUDA provides an abstraction of the hardware architecture enabling you to
control thread concurrency.

CUDA provides two primary profiling tools:

+ `nvvp`: visual profiler.
+ `nvprof`: A command-line profiler.

There are three common limiters to performance for a kernel that you may encounter:

+ Memory bandwidth
+ Compute resources
+ Instruction and memory latency

## 3.2 Understanding the Nature of Warp Execution

### 3.2.1 Warps and Thread Blocks

Warps are the basic unit of execution in an SM. When you launch a grid of thread blocks,
the thread blocks in the grid are distributed among SMs. Once a thread block is scheduled
to an SM, threads in the thread block are further partitioned into warps. A warp consists
of 32 consecutive threads an all threads in a warp are executed in
*Single Instruction Multiple Thread* fashion; that is, all threads execute the same
instruction, and each thread carries out that operation on its own private data.

A warp is never split between different thread blocks. If thread block size is not an even
multiple of warp size, some threads in the last warp are left inactive.

### 3.2.2 Warp Divergence

Because all threads in a warp must execute identical instructions on the same cycle, if one
thread executes an instruction, all threads in a warp must execute that instruction. This
could become a problem if threads in the same warp take different paths through an application.
For example, consider the following statement:

```c++
if (cond) {

} else {

}
```

Suppose for 16 threads in a warp executing this code, `cond` is `true`, but for the other 16
`cond` is `false`. Then half of the warp will need to execute the instructions in the `if`
block, and the other half will need to execute the instructions in the `else` block. Threads
in the same warp executing different instructions is referred to as *warp divergence*.

If threads of a warp diverge, the warp serially executes each branch path, disabling threads
that do not take that path. Warp divergence can cause significantly degraded performance.

To obtain the best performance, you should avoid different execution paths within the same
warp. It may be possible to partition data in such a way as to ensure all threads in the
same warp take the same control path in an application.

For example, suppose you have two branches, as shown in the following simple arithmetic
kernel. You can simulate a poor partitioning of data with an even and odd threads
approach, causing warp divergence. The condition `tid % 2 == 0` makes even numbered
threads take the `if` clause and odd numbered threads take the `else` clause.

```c++
__global__ void mathKernel1(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  a = b = 0.0f;

  if (tid % 2 == 0) {
    a = 100.0f;
  } else {
    b = 200.f;
  }

  c[tid] = a + b;
}
```

If you interleave data using a warp approach (instead of a thread approach), you can
avoid warp divergence and achieve 100 percent utilization of the device. The condition
`(tid / warpSize) % 2 == 0` forces the branch granularity to be a multiple of warp
size; the even warps take the `if` clause, and the odd warps take the `else` clause.
This kernel produces the same result but in a different order.

```c++
__global__ void mathKernel2() {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  a = b = 0.0f;

  if ((tid / warpSize) % 2 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}
```

### 3.2.3 Resource Partitioning

The local execution context of a warp mainly consists of the following resources:

+ Program counters.
+ Registers.
+ Shared memory.

The execution context of each warp processed by an SM is maintained on-chip during the
entire lifetime of the warp. Therefore, switching from one execution context to another
has no cost.

Each SM has a set of 32-bit registers stored in a register file that are partitioned
among threads, and a fixed amount of shared memory that is partitioned among thread
blocks.

Resource availability generally limits the number of resident thread blocks per SM.
The number of registers and the amount of shared memory per SM vary for devices of
different compute capability. If there are insufficient registers or shared memory
on each SM to process at least one block, the kernel launch will fail.

A thread block is called an *active block* when compute resources, such as registers
and shared memory, have been allocated to it. The warps it contains are called
*active warps*. Active warps can be further classified into the following three types:

+ Selected warp
+ Stalled warp
+ Eligible warp

The warp schedulers on an SM select active warps on every cycle and dispatch them to
execution units. A warp that is actively executing is called a *selected warp*. If
an active warp is ready for execution but not currently executing, it is an
*eligible warp*. A warp is eligible for execution if both of the following two
conditions is met:

+ Thirty-two CUDA cores are available for execution.
+ All arguments to the current instruction are ready.

If a warp stalls, the warp scheduler picks up an eligible warp to execute in its
place.

Compute resource partitioning requires special attention in CUDA programming: The
compute resources limit the number of active warps. Therefore, you must be aware
of the restrictions imposed by the hardware, and the resources used by your
kernel. In order to maximize GPU utilization, you need to maximize the number
of active warps.

### 3.2.4 Latency Hiding

The number of clock cycles between an instruction being issued and being completed
is defined as instruction *latency*. Full compute resource utilization is achieved
when all warp schedulers have an eligible warp at every clock cycle. This ensures
that the latency of each instruction can be hidden by issuing other instructions
in other resident warps.

Compared with C programming on the CPU, *latency binding* is particularly important
in CUDA programming. CPU cores are designed to *minimize latency* for one or two
threads at a time, whereas GPUs are designed to handle a large number of concurrent
and lightweight threads in order to *maximize throughput*. GPU instruction latency
is hidden by computation from other warps.

When considering instruction latency, instructions can be classified into two basic
types:

+ Arithmetic instructions: 10-20 clock cycles.
+ Memory instructions: 400-800 clock cycles.

*Little's Law* can provide a reasonable approximation. Originally a theorem in queue
theory, it can also be applied to GPUs:

$$
Warps = Latency \times Throughput
$$

### 3.2.5 Occupancy

Instructions are executed sequentially within each CUDA core. When one warp stalls, the
SM switches to executing other eligible warps. *Occupancy* is the ratio of active warps
to maximum number of warps, per SM.

$$
occupancy = \frac{active_warps}{max_warps}
$$

To enhance occupancy, you need to resize the thread block configuration or re-adjust
resource usage to permit more simultaneously active warps and improve utilization
of compute resources. Manipulating thread blocks to either extreme can restrict
resource utilization.

+ *Small thread blocks*: Too few threads per block leads to hardware limits on the
number of warps per SM to be reached before all resources are fully utilized.
+ *Large thread blocks*: Too many threads per block leads to fewer per-SM hardware
resources available to each thread.

### 3.2.6 Synchronization

Barrier synchronization is a primitive that is common in many parallel programming
languages. In CUDA, synchronization can be performed at two levels:

+ **System-level**: Wait for all work on both the host and the device to complete.
`cudaDeviceSynchronize`
+ **Block-level**: Wait for all threads in a thread block to reach the same point
in execution on the device. `__syncthreads`.

## 3.3 Avoiding Branch Divergence

Sometimes, control flow depends on thread indices. Conditional execution within a warp
may cause warp divergence that can lead to poor kernel performance. By rearranging
data access patterns, you can reduce or avoid warp divergence.

### 3.3.1 The Parallel Reduction Problem

Suppose you want to calculate the sum of an array of integers with $N$ elements sequentially:

```c++
int sum = 0;
for (int i = 0; i < N; i++) {
  sum += array[i];
}
```

Due to the associative and communicative properties of addition, the elements of this
array can be summed in any order. So you can perform parallel addition in the following
way:

1. Partition the input vector into small chunks.
2. Have a thread calculate the partial sum for each chunk.
3. Add the partial results from each chunk into a final sum.

A common way to accomplish parallel addition is using an iterative pairwise implementation:
A chunk contains only a pair of elements, and a thread sums those two elements to produce
one partial result. These partial results are then stored *in_place* in the original input
vector. These new values are used as the input to be summed in the next iteration.

Depending on whether output elements are stored in-place for each iteration, pairwise
parallel sum implementations can be further classified into the following two types:

+ **Neighbored pair**: Elements are paired with their immediate neighbor.
+ **Interleaved pair**: Paired elements are separated by a given stride.

![Neighbored pair sum](https://s2.loli.net/2023/09/05/nJXuMTE7VQkWh5Z.png)

![Interleaved pair sum](https://s2.loli.net/2023/09/05/fzwXUgmHrxlD5bd.png)

The following C function is a recursive implementation of the interleaved pair approach:

```c
int recursiveReduce(int *data, const int size) {
  if (size == 1) {
    return data[0];
  }

  const int stride = size / 2;
  for (int i = 0; i < stride; i++) {
    data[i] += data[i + stride];
  }
  return recursiveReduce(data, stride);
}
```

While the code above implements addition, any commutative and associative operation
could replace addition. For example, `max`, `min`, `average` and `product`.

This general problem of performing a commutative and associative operation across
a vector is known as the *reduction* problem.

### 3.3.2 Divergence in Parallel Reduction

As a starting point, we will experiment with a kernel implementing the neighbored
pair approach illustrate below.

![Parallel neighbored pair sum](https://s2.loli.net/2023/09/05/BTysqGr1KVwzLel.png)

```c++
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  }

  for (int stride = 1; stride < blockDim.x; stride *=2) {
    if (tid % (2 * stride) == 0) {
      idata[tid] += idata[tid + stride];
    }

     __syncthreads();
  }



  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}
```

### 3.3.3 Improving Divergence in Parallel Reduction

Examine the kernel `reduceNeighbored` and note the following conditional statement:

```c++
if (tid % (2 * stride) == 0)
```

Because this statement is only true for even numbered threads, it causes highly
divergent warps. In the first iteration of parallel reduction, only even threads
executes the body of this conditional statement but all threads must be scheduled.
On the second iteration, only one fourth of all threads are active but still all
threads must be scheduled. Warp divergence can be reduced by rearranging the array
index of each thread to force neighboring threads to perform the addition. Below
illustrates this implementation, the store location of partial sums has not
changed, but the working threads have been updated.

![Parallel neighbored pair sum with less divergence](https://s2.loli.net/2023/09/06/eWmyxX7hIFiVCRl.png)

```c++
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  }

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x) {
      idata[index] += idata[index + stride];
    }

    __syncthreads();
  }
}
```

With a thread block size of 512 threads, the first 8 warps execute the first round of
reduction, and the remaining 8 warps do nothing. In the second round, the first 4
warps execute the reduction, and the remaining 12 warps do nothing. Therefore, there is
no divergence at all. Divergence only occurs in the last 5 rounds when the total number
of threads at each round is less than warp size.

### 3.3.4 Reducing with Interleaved Pairs

The kernel for interleaved reduction is as follows:

```c++
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x;

  if (idx >= n) {
    return;
  }

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }

    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }
}
```

The performance improvement is primarily a result of the global memory load and
store patterns in `reduceInterleaved`.

## 3.4 Unrolling Loops

*Loop unrolling* is a tchnique that attempts to optimize loop execution by reducing
the frequency of branches and lop maintenance instructions. Unrolling in CUDA can
mean a variety of things. However, the goal is still the same: improving performance
by reducing instruction overheads and creating more independent instructions to
schedule.

The following kernel is a revision to the `reduceInterleaved` kernel: For each thread
block, data from two data blocks is summed. Each thread works on more than one data
block and processes a single element from each data block.

```c++
__global__ void reduceUnrolling(int *g_idata, int *g_odata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  int *idata = g_idata + blockIdx.x * blockDim.x * 2;

  if (idx + blockDim.x < n) {
    g_idata[idx] += g_idata[idx + blockDim.x];
  }

  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      idata[tid] += idata[tid + stride];
    }

    __syncthreads();
  }

  if (tid == 0) {
    g_odata[blockIdx.x] = idata[0];
  }

}
```

More independent memory load/store operations in a single thread yield better performance
as memory latency can be better hidden.

## 3.5 Dynamic Parallelism

CUDA *Dynamic Parallelism* allows new GPU kernels to be created and synchronized directly
on the GPU.

### 3.5.1 Nested Execution

With dynamic parallelism, the kernel execution concepts can also be applied to kernel
invocation directly on the GPU. The same kernel invocation syntax is used to launch a
new kernel within a kernel.

In dynamic parallelism, kernel executions are classified into two types: parent and child.
A *parent thread*, *parent thread block*, or *parent grid* has launched a new grid, the
*child gird*. A *child thread*, *child thread block* or *child gird* has been launched by
a parent. A child grid must complete before the parent thread, parent thread block, or
parent grids are considered complete.

Grid launches in a device thread are visible across a thread block. This means that a thread
may synchronize on the child grids launched by that thread or by other threads in the
same thread block.

Parent and child grids share the same global and constant memory storage, but have distinct
local and shared memory.

```c++
__global__ void nestedHelloWorld(const int iSize, int iDepth) {
  int tid = threadIdx.x;

  printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid,
         blockIdx.x);

  if (iSize == 1) {
    return;
  }

  int nthreads = iSize >> 1;
  if (tid == 0 && nthreads > 0) {
    nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
    printf("-------> nested execution depth: %d\n", iDepth);
  }
}
```
