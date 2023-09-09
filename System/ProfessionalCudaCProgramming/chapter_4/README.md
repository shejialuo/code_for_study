# Chapter 4 Global Memory

## 4.1 Introducing the CUDA Memory Model

Having a large amount of low-latency, high-bandwidth can be very beneficial to performance.
You must rely on the memory model to achieve optimal latency and bandwidth.

Both GPUs and CPUs use similar principles and models in memory hierarchy design. The key
difference between GPU and CPU memory models is that the CUDA programming model exposes more
of the memory hierarchy and gives you more explicit control over its behavior.

To programmers, there are generally two classifications of memory:

+ Programmable: You explicitly control what data is placed in programmable memory.
+ Non-programmable: You have no control over data placement, and rely on automatic techniques
to achieve good performance.

In the CPU memory hierarchy, L1 cache and L2 cache are examples of non-programmable memory.
On the other hand, the CUDA memory model exposes many types of programmable memory to you:

+ Registers
+ Shared memory
+ Local memory
+ Constant memory

Below illustrates the hierarchy of these memory spaces. Each has a different scope, lifetime,
and caching behavior. A thread in a kernel has its own private local memory. A thread block
has its own shared memory, visible to all threads in the same thread block, and whose contents
persist for the lifetime of the thread block. All threads can access global memory. There are
also two read-only memory spaces accessible by all threads: the constant and texture memory
spaces.

[GPU memory hierarchy](https://s2.loli.net/2023/09/06/dxlacWv215DPbVn.png)

### 4.1.1 Registers

Registers are the fastest memory space on a GPU. An automatic variable declared in a kernel
without any other type qualifiers is generally stored in a register. Register variables share
their lifetime with the kernel.

Registers are scarce resources that are partitioned among active warps in an SM. Using fewer
registers in your kernels may allow more thread blocks to reside on an SM. More concurrent
thread blocks per-SM can increase occupancy and improve performance.

If a kernel uses more registers than the hardware limit, the excess registers will spill
over to local memory. This *register spilling* can have adverse performance consequences.

### 4.1.2 Local Memory

The name "local memory" is misleading: values spilled to local memory reside in the same
physical location as global memory, so local memory access are characterized by high
latency and low bandwidth and are subject to the requirements for efficiency memory access.

### 4.1.3 Shared Memory

Variables decorated with the `__shared__` attribute in a kernel are stored in shared
memory. Because shared memory is on-chip, it has a much higher bandwidth and much lower
latency than local or global memory. It is used similarly to CPU L1 cache, but is also
programmable.

Each SM has a limited amount of shared memory that is partitioned among thread blocks.
Shared memory shares its lifetime with a thread block. When a thread block is finishing
executing, its allocation of shared memory will be released and assigned to other
thread blocks.

### 4.1.4 Constant Memory

Constant memory resides in device memory and is cached in a dedicated, per-SM constant
cache. A constant variable is decorated with `__constant__`.

Constant variables must be declared with global scope, outside of any kernels. A limited
amount of constant memory can be declared. Constant memory is statically declared and
visible to all kernels in the same compilation unit.

### 4.1.5 Texture Memory

Texture memory resides in device memory and is cached in a per-SM, read-only cache. Texture
memory is a type of global memory that is accessed through a dedicated read-only cache.

### 4.1.6 Global Memory

Global memory is the largest, highest-latency, and most commonly used memory on a GPU. The
name *global* refers to its scope and lifetime. Its state can be accessed on the device from
any SM throughout the lifetime of the application.

A variable in global memory can either be declared statically or dynamically. You can declare
a global variable statically in device code using `__device__`.

When a warp performs a memory load/store, the number of transactions required to satisfy that
request typically depends on the following two factors:

+ Distribution of memory addresses across the threads of that warp.
+ Alignment of memory addresses per transaction.

In general, the more transactions necessary to satisfy a memory request, the higher the potential
for unused bytes to be transferred, causing a reduction in throughput efficiency.

### 4.1.7 GPU Caches

Like CPU caches, GPU caches are non-programmable memory. There are four types of cache in
GPU devices:

+ L1
+ L2
+ Read-only constant
+ Read-only texture

On the CPU, both memory loads and stores can be cached. However, on the GPU only memory load
operations can be cached; memory store operations cannot be cached.

## 4.2 Memory Management

### 4.2.1 Pinned Memory

The GPU cannot safely access data in pagetable host memory because it has no control over
when the host operating system may choose to physically move that data. When transferring
host memory to device memory, the CUDA driver first allocates temporary *page-locked* of
*pinned* host memory, copies the source host data to pinned memory, and then transfers
the data from pinned memory to device memory.

The CUDA runtime allows you to directly allocate pinned host memory using:

```c++
cudaError_t cudaMallocHost(void **devPtr, size_t count);
```

This function allocates `count` bytes for host memory that is page-locked and accessible to
the device. Since the pinned memory can be accessed directly by the device, it can be read
and written with much higher bandwidth than pageable memory.

Pinned host memory must be freed with:

```c++
cudaError_t cudaFreeHost(void *ptr);
```

### 4.2.2 Zero-Copy Memory

In general, the host cannot directly access device variables, and the device cannot directly
access host variables. There is one exception to this rule: *zero-copy* memory. Both the
host and device can access zero-copy memory.

GPU threads can directly access zero-copy memory. There are several advantages to using
zero-copy memory in CUDA kernels, such as:

+ Leveraging host emory when there is insufficient device memory.
+ Avoiding explicit data transfer between the host and device.
+ Improving PCIe transfer rates.

When using zero-copy memory to share data between the host and device, you must synchronize
memory accesses across the host and device. Modifying data in zero-copy memory from both
the host and device at the same time will result in undefined behavior.

Zero-copy memory is pinned memory that is mapped into the device address space. You can
create a mapped, pinned memory region with the following function:

```c++
cudaError_t cudaHostAlloc(void **pHost, size_t count, unsigned int flags);
```

This function allocates `count` bytes of host memory that is page-locked and accessible to
the device. Memory allocated by this function must be freed with `cudaFreeHost`. The `flags`
parameter enables further configuration of special properties of the allocated memory:

+ `cudaHostAllocDefault`: identical to `cudaMallocHost`.
+ `cudaHostAllocPortable`: returns pinned memory that can be used by all CUDA contexts.
+ `cudaHostAllocWriteCombined`: returns write-combined memory.
+ `cudaHostAllocMapped`: returns host memory that is mapped into the device address space.

You can obtain the device pointer for mapped pinned memory using the following function:

```c++
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
```

Using zero-copy memory as a supplement to device memory with frequent read/write operations
will significantly slow performance. Because every memory transaction to mapped memory
must pass over the PCIe bus, a significant amount of latency is added even when compared
to global memory.

### 4.2.3 Unified Virtual Addressing

Devices with compute capability 2.0 and later support a special addressing mode called
*Unified Virtual Addressing* (UVA). With UVA, host memory and device memory share a single
virtual address space.

### 4.2.4 Unified Memory

With CUDA 6.0, a new feature called *Unified Memory* was introduced to simplify memory
management in the CUDA programming model. Unified Memory creates a pool of managed memory,
where each allocation from this memory pool is accessible on both CPU and GPU with the
same memory address.

*Managed memory* refers to Unified Memory allocations that are automatically managed by
the underlying system and is interoperable with device-specific allocations, such ast hose
created using the `cudaMalloc` routine.

You can allocate managed memory dynamically using the following CUDA runtime function:

```c++
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags=0);
```

This function allocates `size` bytes of managed memory and returns a pointer in `devPtr`.
The pointer is valid on all devices and the host.

## 4.3 Memory Access Patterns

Maximizing your application's use of global memory bandwidth is a fundamental step in
kernel performance tuning.

There aer two characteristics of device memory accesses that you should strive for when
optimizing your application:

+ *Aligned memory accesses*: the first address of a device memory transaction is an even
multiple of the cache granularity being used to service the transaction.
+ *Coalesced memory accesses*: occur when all 32 threads in a warp access a contiguous chunk
of memory.
