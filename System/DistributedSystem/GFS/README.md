# The Google File System

## Introduction

The developers have reexamined traditional choices and explored radically
different points in the design space.

+ Component failures are the norm rather than the exception.
Therefore, constant monitoring, error detection, fault tolerance,
and automatic recovery must be integral to the system.
+ Files are huge by traditional standards. As a result, design
assumptions and parameters such as I/O operation and block sizes
have to be revisited.
+ Most files are mutate by appending new data rather than overwriting
existing data.
+ Co-designing the applications and the file system API benefits
the overall system by increasing our flexibility.

## Design Overview

### Assumptions

There are some assumptions in designing a file system.

+ The system is built from many inexpensive commodity components that
often fail. It must constantly monitor itself and detect, tolerate,
and recover promptly from component failures on a routine basis.
+ The system stores a modest number of large files. We expect
a few million files, each typically 100MB or larger in size.
+ The workload primarily consist of two kinds of reads: large
streaming reads and small random reads. In large streaming reads,
individual operations typically read hundreds of KBs, more commonly 1MB
or more. Successive operations from the same client often read
through a contiguous region of a file. A small random read typically
reads a few KBs at some arbitrary offset. Performance-conscious
applications often batch and sort their small needs to advance steadily
through the file rather than go back and forth.
+ The workloads also have many large, sequential writes that
append data to files. Typical operation sizes are similar to those
for reads.
+ The system must efficiently implement well-defined semantics
for multiple clients that concurrently append to the same file.
Our files are often used as producer-consumer queues or for many-way
merging. Hundreds of producers, running one per machine, will
concurrently append to a file. Atomicity with minimal synchronization overhead
is essential.
+ HIgh sustained bandwidth is more important than low latency.

### Interface

+ `create`
+ `delete`
+ `open`
+ `close`
+ `read`
+ `write`
+ `snapshot`: creates a copy of file or a directory tree at low cost
+ `record append`: allows multiple clients to append data to
the same file concurrently while guaranteeing the atomicity of
each individual client's append.

### Architecture

A GFS cluster consists of a single *master* and multiple *chunkservers*
and is accessed by multiple *clients*, as shown below.Each of these is
typically a commodity Linux machine running a user-level server
process. It is easy to run both a chunkserver and a client on the same
machine, as long as machine resources permit and the lower reliability
caused by running possibly flaky application code is acceptable.

![GFS Architecture](https://s2.loli.net/2022/09/06/g4IuJfheRQ8lx3s.png)

Files are divided into fixed-size *chunks*. Each chunk is identified
by an immutable and globally unique 64 bit *chunk handle* assigned
by the master at the time of chunk creation. Chunkservers store
chunks on local disks as Linux files and read or write chunk data
specified by a chunk handle and byte range. For reliability,
each chunk is replicated on multiple chunkservers. By default, we store
three replicas, though users can designate different replication levels
for different regions of the file namespace.

The master maintains all file system metadata. This includes the namespace,
access control information, the mapping from files to chunks, and
the current locations of chunks. It also controls system-wide activities
such as chunk lease management, garbage collection of orphaned chunks,
and chunk migration between chunkservers. The master periodically
communicates with each chunkserver in *HeartBeat* messages to
give it instructions and collect its state.

### Single Master

Having a single master vastly simplifies our design and enables the
master to make sophisticated chunk placement and replication
decisions using global knowledge. However, we must minimize its
involvement in reads and writes so that it does not become a bottleneck.
Clients never read and write file data through the master. Instead,
a client asks the master which chunkservers it should contact.

We give the interactions for a simple read operation.

1. Using the fixed chunk size, the client translates the file name
and byte offset specified by the application into a chunk index
within the file.
2. Sends the master a request containing the file name and chunk index.
3. Master replies with the corresponding chunk handle and locations of the replicas.
4. The client caches this information.
5. The client sends a request to one of the replicas.

### Chunk Size

Chunk size is one of the key design parameters. GFS has chosen 64MB.

A large chunk size offers several important advantages.

+ It reduces clients' need to interact with the master because
reads and writes on the same chunk require only one initial request
to the master. The reduction is especially significant for our
workloads because applications mostly read and write large files sequentially.
+ Since on a large chunk, a client is more likely to perform many
operations on a given chunk, it can reduce network overhead by
keeping a persistent TCP connection to the chunkserver over
an extended period of time.
+ It reduces the size of the metadata stored on the master.

### Metadata

The master stores three major types of metadata:

+ The file and chunk namespaces.
+ The mapping from files to chunks.
+ The locations of each chunk's replicas

The first two types are also kept persistent by logging mutations
to an *operation log* stored on the master's local disk and replicated on
remote machines.

#### In-Memory Data Structures

Since metadata is stored in memory, master operations are fast.
Furthermore, it is easy and efficient for the master to periodically
scan through its entire state in the background. This periodic scanning
is used to implement chunk garbage collection, re-replication
in the presence of chunkserver failures, and chunk migration to balance
load and disk space usage across chunkservers.

#### Chunk Locations

The master does not keep a persistent record of which chunkservers have
a replica of a given chunk. It simply polls chunkservers for that
information at startup. The master can keep itself up-to-date thereafter
because it controls all chunk placement and monitors chunkserver status with
regular *HeartBeat* messages.

#### Operation Log

The operation log contains a historical record of critical metadata
changes. Since the operation log is critical, we must store it reliably
and not make changes visible to clients until metadata changes are
made persistent. Therefore, we replicate it on multiple remote machines
and respond to a client operation only after flushing the corresponding log
record to disk both locally and remotely.

## System Interactions

### Leases and Mutation Order

A mutation is an operation that changes the contents or metadata of
a chunk such as a write or an append operation. Each mutation is
performed at all the chunk's replicas. We use leases to maintain
a consistent mutation order across replicas. The master grants
a chunk least to one of the replicas, which we call the *primary*. The
primary picks a serial order for all mutations to the chunk. All
replicas follow this order when applying mutations.

![Write Control and Data Flow](https://s2.loli.net/2022/09/06/M6mQUoS7Aj3FO1q.png)

1. The client asks the master which chunkserver holds the current
lease for the chunk and the locations of the other replicas.
2. The master replies with the identity of the primary and the
locations of the other (*secondary*) replicas.
3. The client pushes the data to all the replicas. A client can do
so in any order.
4. Once all the replicas have acknowledged receiving the data, the
client sends a write request to the primary. The request identifies
the data pushed earlier to all of the replicas. The primary assigns
consecutive serial numbers to all the mutations it receives,
possibly from multiple clients. It applies the mutation to its
own local state in serial number order.
5. The primary forwards the write request to all secondary replicas.
Each secondary replica applies mutations in the same serial number
order assigned by the primary.
6. The secondaries all reply to the primary indicating that
they have completed the operation.
7. The primary replies to the client. Any errors encountered at
any of the replicas are reported to the client.

### Data Flow

GFS decouples the flow of data from the flow of control to use
the network efficiently. Data is pushed linearly along a carefully
picked chain of chunkservers in a pipelined fashion.

To fully utilize each machine's network bandwidth, the data is pushed
linearly along a chain of chunkservers rather than distributed in
some other topology.

To avoid network bottlenecks and high-latency links as much as
possible, each machine forwards the data to the "closet" machine
in the network topology that has not received it.

### Atomic Record Appends

GFS provides an atomic append operation called *record append*. In
a record append, however, the client specifies only the data. GFS
appends it to the file at least once atomically at an offset
of GFS's choosing and returns that offset to the client.

The client pushes the data to all replicas of the last chunk of
the file Then, it sends its request to the primary. The primary
checks to see if appending the record to the current chunk would
cause the chunk to exceed the maximum size. If so ,it pads the chunk
to the maximum size. If so, it pads the chunk to the maximum size,
tells secondaries to do the same, and replies to the client indicating
that the operation should be retried on the next chunk. If the
record fits within the maximum size, the primary appends the data
to its replica, tells the secondaries to write the data at the
exact offset where it has, and finally replies success to the client.

### Snapshot

The snapshot operation makes a copy of a file or a directory tree
almost instantaneously, while minimizing any interruptions of
ongoing mutations.

And there are too many details. I omit
