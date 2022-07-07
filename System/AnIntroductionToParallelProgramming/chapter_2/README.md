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

The below diagram (a) shows a simple crossbar. The lines are bidirectional
communication links, the squares are cores or memory modules. amd
the circles are switches.

The individual switches can assume one of the two configurations shown
in (b). With these switches and at least as many memory modules
as processors, there will only be a conflict between two cores attempting
to access memory if the two cores attempt to simultaneously access
the same memory module.

![Crossbar](https://s2.loli.net/2022/07/05/9pT2oGvmBRjgUsz.png)

Crossbars allow simultaneous communication among different devices, so they
are much faster than buses however more expensive.

#### 2.3.4.2 Distributed-memory interconnects

Distributed-memory interconnects are often divided into two groups:
direct interconnects and indirect interconnects. In a *direct interconnect*
each switch is directly connected to a processor-memory pair, and the
switches are connected to each other. *Indirect interconnects* provide an
alternative to direct interconnects. In an indirect interconnect, the switches
may not be directly connected to a processor. They're often shown with unidirectional
links and a collection of processors, each of which has an outgoing
and an incoming link, and a switching network.

### 2.3.5 Cache coherence

Suppose we have a shared-memory system with two cores, each of which
has its own private data cache. As long as the two cores only read
shared dta, there is no problem.However, things would become complicated if not.

![A shared-memory system with two cores and two caches](https://s2.loli.net/2022/07/07/293vhoMWaUL4iDp.png)

For example, suppose that `x` is a shared variable that has been initialized
to 2, `y0` is private and owned by core 0, and `y1` and `z1` are private
and owned by core 1. Now suppose the following statements are
executed at the indicated times.

| **Time** |          **Core 0**          |          **Core 1**          |
|:--------:|:----------------------------:|:----------------------------:|
|     0    |            y0 = x;           |          y1 = 3 * x          |
|     1    |            x = 7;            | statement(s) not involving x |
|     2    | statement(s) not involving x |          z1 = 4 * x          |

It's not clear what value `z1` will get. This unpredictable behavior will
occur regardless of whether the system is using a write-through or a write-back
policy. If it's using a write-through policy, the main memory will be updated
by the assignment `x = 7`. However, this will have no effect on the value
in the cache of core 1. If the system is using a write-back policy,
the new value of `x` in the cache of core 0 probably won't be available
to core 1 when it updates `z1`.

Clearly, this is a problem. The programmer doesn't have direct control
over when the caches are updated. The caches we described for single
processor systems provide no mechanism for ensuring that when the
caches of multiple processors store the same variable, an update
by one processor to the cached variable is "seen" by the other
processors. This is called the *cache coherence* problem.

#### 2.3.5.1 Snooping cache coherence

There are two main approaches to ensuring cache coherence:

+ *Spooning cache coherence*.
+ *directory-based cache coherence*.

The idea behind snooping comes from bus-based systems. Thus when core 0
updates the copy of `x` stored in its cache, if it also broadcasts this
information across the bus, and if core 1 is "snooping" the bus, it
will see that `x` has been updated, and it can mark its copy of `x`
as invalid. The principal difference between the description and
the actual snooping protocol is that the broadcast only informs
the other cores that the *cache line* contaning `x` has been updated,
not that `x` has been updated.

#### 2.3.5.2 Directory-based cache coherence

Unfortunately, in large networks, broadcasts are expensive, and snooping cache
coherence requires a broadcast every time a variable is updated. So snooping
cache coherence isn't scalable, because for larger systems it will
cause performance to degrade.

*Directory-based cache coherence* protocols attempt to solve this problem
through the use of a data structure called a *directory*. The directory
stores the status of each cache line. Typically, this data structure
is distributed.

## 2.4 Parallel software

First, we need introduce some terminology. Typically when we run our
shared-memory programs, we'll start a single process and fork multiple
threads. So when we discuss shared-memory programs, we'll talk about
*threads* carrying out tasks. On the other hand, when we run distributed-memory
programs, we'll start multiple processes, and we'll talk about *processes*
carrying out tasks. When the discussion applies equally well to
shared-memory and distributed-memory systems, we'll talk about
*processes/threads* carrying out tasks.

### 2.4.1 Coordinating the processes/threads

In a very few cases, obtaining excellent parallel performance is trivial. For
example, suppose we have two arrays and we want to add them.

```c
double x[n], y[n];
for(int i = 0; i < n; ++i) {
  x[i] += y[i];
}
```

To parallelize this, we only need to assign elements of the arrays to
the processes/threads, we might make process/thread 0 responsible for elements
$0, \dots, n / p - 1$ and so on.

So for this example, the programmer only needs to

+ Divide the work among the processes/threads in such as way that
  + Each process/thread gets roughly the same amount of work.
  + The amount of communication required is minimized.
+ Arrange for the processes/threads to synchronize.
+ Arrange for communication among the processes/threads.

### 2.4.2 Shared-memory

As we noted earlier, in shared-memory programs, variables can be
*shared* or *private*.

#### 2.4.2.1 Dynamic and static threads

In many environments, shared-memory programs use *dynamic threads*. In this
paradigm, there is often a master thread and at any given instant
a collection of worker threads. The master thread typically waits for
work requests over a network and when a new request arrives, it forks
a worker thread, the thread carries out the request, and when the thread
completes the work, it terminates and joins the master thread. This
paradigm makes efficient use of system resources, since the resources
required by a thread are only being used while the thread is actually running.

An alternative to the dynamic paradigm is the *static thread* paradigm. In
this paradigm, all of the threads are forked after any needed setup by
the master thread and the threads run until all the work is completed.
After the threads join the master thread, the master thread may do some
cleanup, and then it also terminates. In terms of resource usage,
this may be less efficient. However, forking and joining threads
can be fairly time-consuming operations. So if the necessary resources are available,
the static thread paradigm has the potential for better performance
than the dynamic paradigm.

#### 2.4.2.2 Nondeterminism

In any MIMD system in which the processors execute asynchronously it is
likely that there will be *nondeterminism*.

### 2.4.3 Distributed-memory

In distributed-memory programs, the cores can directly access only
their own, private memories. There are several APIs that are used. However,
by far the most widely used is message-passing.

The first thing to note regarding distributed-memory APIs is that
they can be used with shared-memory hardware. It's perfectly feasible
for programmers to logically partition shared-memory into private address
spaces for the various threads, and a library or compiler can
implement the communication that's needed.

#### 2.4.3.1 Message-passing

A message-passing API provides (at a minimum) a send a receive function.
Processes typically identify each other by ranks in the range
$0, 1, \dots, p - 1$, where $p$ is the number of processes.
For example, process 1 might send a message to process 0 with
the following code:

```c
char message[100];

my_rank = Get_rank();
if(my_rank == 1) {
  sprintf(message, "Greetings from process 1");
  Send(message, MSG_CHAR, 100, 0);
} else if(my_rank == 0) {
  Receive(message, MSG_CHAR, 100, 1);
  printf("Process 0 > Received: %s\n", message);
}
```

The most widely used API for message-passing is the *Message-Passing Interface* or MPI.

#### 2.4.3.2 Ond-sided communication

In *one-sided communication*, or *remote memory access*, a single process
calls a function, which updates either local memory with a value
from another process or remote memory with a value from the calling process.

### 2.4.4 GPU programming

The memory for the CPU host and the GPU memory are usually separate. So
the code that runs on the host typically allocates and initializes storage
on both the CPU and GPU. It will start the program on the GPU, and it is
responsible for the output of the results of the GPU program. Thus GPU
programming is really *heterogeneous* programming, since it involves programming
two different types of processors.

The GPU itself will have one or more processors. Each of these processors
is capable of running hundreds or thousands of threads.

The threads running on a processor are typically divided into groups:
the threads within a group use the SIMD model, and two threads in
different groups can run independently.

Another issue in GPU programming that's different from CPU programming
is how the threads are
scheduled to execute. GPUs use a hardware scheduler, and this hardware
scheduler uses very little overhead. However, the scheduler will choose
to execute an instruction when all the threads in SIMD group are ready.

## 2.5 Input and output

### 2.5.1 MIMD systems

We'll make these assumptions and following these rules when our
parallel programs need to do I/O:

+ In distributed-memory programs, only process 0 will access `stdin`. In
shared-memory programs, only the master thread or thread 0 will access `stdin`.
+ In both distributed-memory and shared-memory programs, all the
processes/threads can access `stdout` and `stderr`.
+ However, because of the nondeterministic order of output to `stdout`,
in most cases only a single process/thread will be used for all
output to `stdout`. The principal exception will be output for
debugging a program. In this situation, we'll often have multiple
processes/threads writing to `stdout`.

### 2.5.2 GPUs

In most cases, the host code in our GPU programs will carry out all I/O.

## 2.6 Performance

### 2.6.1 Speedup and efficiency in MIMD systems

Usually the best our parallel program can do is to divide the work
equally among the cores while at the same time introducing no
additional work for the cores. If we call the serial run-time $T_{serial}$ and
our parallel run-time $T_{parallel}$, then it's usually the case that
the best possible run-time of our parallel program is $T_{parallel} = T_{serial} / p$.
When this happens, we say that our parallel program has *linear speedup*.

In practice, we usually don't get perfect linear speedup, because
the use of multiple processes/threads almost invariably introduces
some overhead. For example, shared-memory programs will almost always
have critical sections, which will require that we use some mutual exclusion
mechanism, such as a mutex. The calls to the mutex functions
are the overhead that's not present in the serial program, and the use
of the mutex forces the parallel program to serialize execution of the
critical section. Distributed-memory programs will almost always
need to transmit data across the network, which is usually much
slower than local memory access. Thus it will be *unusual* for us
to find that our parallel programs get linear speedup. Furthermore,
it's likely tha the overheads will increase as we increase the
number of processes or threads.

So we define the *speedup* of a parallel program to be

$$
S = \frac{T_{serial}}{T_{parallel}}
$$

And we have $S = p$ for the linear speedup. And we define $S /p$
the *efficiency* of the parallel program. If we substitute the
formula for $S$, we see that the efficiency is

$$
E = \frac{S}{p} = \frac{T_{serial}}{p \cdot T_{parallel}}
$$

If the serial run-time has been taken on the same type of core that
the parallel system is using, we can think of efficiency as the
average utilization of the parallel cores on solving the problem.
That is, the efficiency can be thought of as the fraction of the
parallel run-time that's spent, on average, by each core working
on solving the original problem. The remainder of the parallel run-time
is the parallel overhead. This can be seen by simply multiplying
the efficiency and the parallel run-time.

$$
E \cdot T_{parallel} = \frac{T_{serial}}{p}
$$

Many parallel programs are developed by explicitly dividing the work
of the serial program among the processes/threads and adding in the
necessary "parallel overhead", such as mutex exclusion or communication.
Therefore if $T_{overhead}$ denotes this parallel overhead, it's often
the case that

$$
T_{parallel} = T_{serial} / p + T_{overhead}
$$

### 2.6.2 Amdahl's law

*Amdahl's law* says unless virtually all of a serial program is parallelized,
the possible speedup is going to be very limited—regardless of
the number of cores available.

For example, we're able to parallelize 90% of a serial program.
Furthermore, suppose that the parallelization is "perfect", we let
$T_{serial} = 20$, and we could have

$$
S = \frac{T_{serial}}{0.9 \times T_{serial} / p + 0.1 \times T_{serial}} = \frac{20}{18 /p + 2} <= 10
$$

Even though we could make $p$ so large, we'll never get a speedup
better than 10.

More generally, if a fraction $r$ of our serial program remains
un-parallelized the Amdahl's law syas we can't get a speedup better than
$1 / r$.

### 2.6.3 Scalability in MIMD systems

In discussion of MIMD parallel program performance, scalability has a somewhat
more formal definition. Suppose we run a parallel program with a fixed
number of processes/threads and a fixed input size, and we obtain
efficiency $E$. Suppose we now increase the number of processes/threads that are
used by the program. If we find a corresponding rate of increase in the problem
size so that the program always has efficiency $E$, the the program is *scalable*.

## 2.7 Parallel program design

So we've got a serial program. How do we parallelize it?

1. *Partitioning*. Divide the computation to be performed and the
data operated on by the computation into small tasks.
2. *Communication*. Determine what communication needs to be carried out.
3. *Agglomeration or aggregation*. Combine tasks and communications identified
in the first step into larger tasks.
4. *Mapping*. Assign the composite tasks identified in the previous
step to processes/threads.

This is sometimes called *Foster's methodology*.
