# Chapter 5 Shared Memory and Constant Memory

## 5.1 Introducing CUDA Shared Memory

Shared memory is smaller, low-latency on chip memory that offers much higher bandwidth
than global memory. You can think of it as a program-managed cache. Shared memory is
generally useful as:

+ An intra-block thread communication channel.
+ A program-managed cache for global memory data.
+ Scratch pad memory for transforming data to improve global data access patterns

### 5.1.1 Shared Memory

Each SM contains a small low-latency memory pool shared by all threads in the thread block
currently executing on that SM. A shared memory variable is declared with the `__shared__`
qualifier.

A fixed amount of shared memory is allocated to each thread block when it starts executing.
This shared memory address space is shared by all threads in a thread block. Shared memory
accesses are issued per warp. Ideally, each request to access shared memory by a warp is
serviced in one transaction. If multiple threads access the same word in shared memory,
one thread fetches the word, and sends it to the other threads via multicast.

### 5.1.2 Shared Memory Banks and Access Mode

#### Memory Banks

To achieve high memory bandwidth, shared memory is divided into 32 equally-sized memory
modules, called *banks*, which can be accessed simultaneously. Shared memory is a 1D
address space. Depending on the compute capability of a GPU, the addresses of shared
memory are mapped to different banks in different patterns. If a shared memory load or
store operation issued by a warp does not access more than one memory location per bank,
the operation can be serviced by one memory transaction. Otherwise, the operation is
serviced by multiple memory transactions, thereby decreasing memory bandwidth utilization.

#### Bank Conflict

When multiple addresses in a shared memory request fall into the same memory bank, a
*bank conflict* occurs, causing the request to be replayed. The hardware splits a request
with a bank conflict into as many separate conflict-free transactions as necessary.

Three typical situations occur when a request to shared memory is issued by a warp:

+ *Parallel access*: multiple addresses accessed across multiple banks.
+ *Serial access*: multiple addresses accessed within the same bank.
+ *Broadcast access*: a single address read in a single bank.

#### Access Mode

Memory bank width varies for devices depending on compute capability. There are two
different bank widths:

+ 4 bytes for device of compute capability 2.x
+ 8 bytes for device of compute capability 3.x

A bank conflict does not occur when two threads from the same warp access the same address.
In that case, for read access, the word is broadcast to the requesting threads, and for
write accesses, the word is written by only one of the threads.

### 5.1.3 Configuring the Amount of Shared Memory

Each SM has 64 KB of on-chip memory. The shared memory and L1 cache share this hardware
resource. CUDA provides two methods for configuring the size of L1 cache and shared memory:

+ Per-device configuration.
+ Per-kernel configuration.

### 5.1.4 Synchronization

CUDA provides several runtime functions to perform intra-block synchronization. In general,
there are two basic approaches to synchronization:

+ Barriers.
+ Memory fences

#### Explicit Barrier

In CUDA, it is only possible to perform a barrier among threads in the same thread block.
You can specify a barrier point in a kernel by calling the following intrinsic function:

```c++
void __syncthreads();
```

#### Memory Fence

*Memory fence* functions ensure that any memory write before the fence is visible to
other threads after the fence.

## 5.2 Constant Memory

Constant memory is a special-purpose memory used for data that is read-only and accessed
uniformly by threads in a warp.

Constant memory resides in device DRAM (like global memory) and has a dedicated on-chip cache.
Like the L1 cache and shared memory, reading from the per-SM constant cache has a much lower
latency than reading directly from constant memory.
