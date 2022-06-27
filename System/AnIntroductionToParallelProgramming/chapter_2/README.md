# 2. Parallel hardware and parallel software

To write *efficient* parallel programs, we often need some
knowledge of the underlying hardware and system software. It's also
very useful to have some knowledge of different types of parallel
software, so in this chapter we'll take a brief look at a
few topics in hardware and software.

## 2.1 Some background

Parallel hardware and software have grown out of conventional
**serial** hardware and software that runs a single job at a time.

### 2.1.1 The von Neumann architecture

The separation of memory and CPU is often called the *von Neumann bottleneck*,
since the interconnect determines the rate at which instructions and
data can be accessed.

To address the von Neumann bottleneck and, more generally, improve
computer performance, computer engineers and computer scientists
have experimented with many modifications to the basic von
Neumann architecture.

## 2.2 Modifications to the von Neumann model

### 2.2.1 The basics of caching

Caching is one of the most widely used methods of addressing
the von Neumann bottleneck. In general, a *cache* is a collection of memory
locations that can be accessed in less time than some other memory
locations. In our setting, when we talk about caches we'll usually
mean a *CPU cache*, which is a collection of memory locations that the
CPU can access more quickly than it can access main memory.

Once we have a cache, an obvious problem is deciding which data and
instructions should be stored in the cache. The universally used principle
is based on the idea that programs tend to use data and instructions
that are physically close to
recently used data and instruction.

To exploit the principle of locality, the system uses an
effectively *wider* interconnect to access data and instructions.
That is, a memory access will effectively operate on blocks
of data and instructions instead of individual instructions
and individual data items. These blocks are called *cache blocks* or *cache lines*.
A typical cache line stores 8 to 16 times as much information
as a single memory location.

When a cache is checked for information and the information is available,
it's called a *cache hit* or just a *hit*. If not available, it's called
a *cache miss* or a *miss*.

When the CPU attempts to read data or instructions and
there's a cache read miss, it will read from memory the cache
line that contains the needed information and store it in the cache.
This may stall the processor while it waits for the slower memory.

When the CPU writes data to a cache, the value in the cache and the
value in main memory are different or *inconsistent*. There are
two basic approaches to dealing with the inconsistency. In *write-through*
caches, the lines is written to main memory when it is written to
the cache. In *write-back* caches, the data isn't written to main memory when it
is written to the cache. In *write-back* caches, the data
isn't written immediately. Rather, the updated data in the
cache is marked *dirty*, and when the cache line is replaced
by a new cache line from memory, the dirty line is written to memory.

### 2.2.2 Cache mappings

Another issue in cache design is deciding where lines should be
stored. At one extreme is a *fully associative* cache, in which a new line can be placed at any
location in the cache. At the other extreme is a *direct mapped* cache,
in which cache line has a unique location in the cache to which it
will be assigned. Intermediate schemes are called *n-way set associative*.
In these schemes, each cache line can be placed in one of n
different locations in the cache.

### 2.2.3 Virtual memory

Using a page table has the potential to significantly increase each
program's overall run-time. To address this issue, processors have a
special address translation cache, called a *translation-lookaside buffer*, or TLB.

### 2.2.4 Instruction-level parallelism

*Instruction-level parallelism* attempts to improve processor performance
by having multiple processor components or *functional units* simultaneously
executing instructions.

#### 2.2.4.1 Pipelining

Basic principle. I omit detail here.

#### 2.2.4.2 Hardware multithreading

*Hardware multithreading* provides a means for systems to continue
doing useful work the the task being currently executed has stalled.

In *fine-grained* multithreading, the processor switches between
threads after each instruction, skipping threads that are stalled.
*Coarse-grained* multithreading attempts to avoid this problem by only
switching threads that are stalled waiting for a time-consuming operation
to complete.

## 2.3 Parallel hardware

### 2.3.1 Classifications of parallel computers

We'll make use of two, independent classifications of parallel computers. The first,
Flyyn's taxonomy, classifies a parallel computer according to the number of
instruction streams and the number of data streams it can simultaneously manage.
A classical von Neumann system is therefore a *single instruction stream, single data stream*,
or SISD system.

### 2.3.2 SIMD systems

SIMD systems operate on multiple data streams by applying the same instruction
to multiple data items.

#### 2.3.2.1 Vector processors

The key characteristic of the vector processor is that they can operate
on arrays or *vectors* of data, while conventional CPUs operate on individual
data elements or *scalars*. Typical recent systems have the following characteristics:

+ *Vector registers*. These are registers capable of storing a vector of operands and
operating simultaneously on their contents.
+ *Vectorized and pipelined functional units*. Note that the same operation
is applied to each element in the vector.
+ *Vector instructions*. These are instructions that operate on vectors rather
+ *Interleaved memory*. The memory system consists of multiple "banks" that can
be accessed more or less independently. After accessing one bank, there will
be a delay before it can be reaccessed, but a different bank can be
accessed much sooner.
+ *Strided memory access and hardware scatter/gather*. In *strided memory access*,
the program accesses elements of a vector located at fixed intervals.
Scatter/gather is writing (scatter) or reading (gather) elements of
a vector located at irregular intervals.

#### 2.3.2.2 Graphics processing units

Real-time graphics application programming interfaces use points, lines, and
triangles to internally represent the surface of an object. The use a
*graphics processing pipeline* to convert the internal representation into
an array of pixels that can be sent to a computer screen. Several of
the stages of this pipeline are programmable. The behavior of the
programmable stages is specified by functions called *shader functions*.

It should be stressed that GPUs are not pure SIMD systems. Although
the datapaths on a given core can use SIMD parallelism, current
generation GPUs can run more than one instruction stream on a single core.

### 2.3.3 MIMD systems

MIMD systems support multiple simultaneous instruction streams
operating on multiple data streams. Thus MIMD systems typically consist of
a collection of fully independent processing units or cores, each of which
has its own control unit and its own datapath. Futhermore, unlike SIMD
systems, MIMD systems are usually *asynchronous*.

#### 2.3.3.1 Shared-memory systems

The most widely available shared-memory systems use one or more *multicore* processors.

In shared-memory systems with multiple multicore processors, the interconnect
can either connect all the processors directly to main memory, or each
processor can have a direct connection to a block of main memory,
and the processors can access each other's blocks of main memory through
special hardware built into the processors.

In the first type of system, the time to access all the memory locations will
be same for all the cores, while in the second type, a memory location to which
a core is directly connected, can be accessed more quickly than a memory
location that must be accessed through another chip. Thus, the first type of system
is called a *uniform memory access*, or UMA, system, while the second type is
called a *nonuniform memory access*, or NUMA, system.

![A UMA multicore system](https://s2.loli.net/2022/06/27/Bjvd2H1J7WGQc5F.png)
![A NUMA multicore system](https://s2.loli.net/2022/06/27/o2vMpsr8lqJmHxL.png)

#### 2.3.3.2 Distributed-memory systems

The most widely available distributed-memory systems are called *clusters*.
They are composed of a collection of commodity systems.

### 2.3.4 Interconnection networks

The interconnect plays a decisive role in the performance of both
distributed- and shared-memory systems. A slow interconnect will seriously degrade
the overall performance of all but the simplest parallel program.

#### 2.3.4.1 Shared-memory interconnects

In the past, it was common for shared memory systems to use a *bus* to
connect processors and memory. Originally, a *bus* was a collection of parallel
communication wires together with some hardware that controls access to the bus.
The key characteristic of a bus is that the communication wires
are shared by the devices that are connected to it. Buses have the virtue of
low cost and flexibility; multiple devices can be connected to a bus
with little additional cost. However, since the communication wires
are shared, as the number of devices connected to the bus increases, the
likelihood that there will be contention for use of the bus increases,
and the expected performance of the bus decreases. Therefore if we connect
a large number of processors to a bus, we would expect that the
processors would frequently have to wait for access to main memory. So, as the
size of shared-memory systems has increased, buses are being replaced
by *switched* interconnects.

As the name suggests, *switched* interconnects use switches to control the
routing of data among the connected devices. A *crossbar* is a relatively
simple and powerful switched interconnect.

Crossbars allow simultaneous communication among different devices, so they
are much faster than buses however more expensive.
