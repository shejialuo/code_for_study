# MapReduce: Simplified Data Processing on Large Clusters

## Introduction

MapReduce is a programming model and an associated implementation for
processing and generating large data sets. Users specify a *map*
function that processes a key/value pair to generate a set of
intermediate key/value pairs, and a *reduce* function hat merges all
intermediate values
associated with the same intermediate key.

## Programming Model

The computation takes a set of *input* key/value pairs, and produces
a set of *output* key/value pairs. The user of the MapReduce library
expresses the computation as two functions: `Map` and `Reduce`.

`Map`, written by the user, takes an input pair and produces a set
of *intermediate* key/value pairs. The MapReduce library groups
together all intermediate values associated with the same intermediate key
`I` and passes them to the `Reduce` function.

The `Reduce` function, also written by the user, accepts an intermediate
key `I` and a set of values for that key. It merges together these values
to form a possibly smaller set of values.

### Types

Conceptually the map and reduce functions supplied by the user
have associated types:

+ `Map`: `(k1, v1) -> list(k2, v2)`
+ `Reduce`: `(k2, list(v2)) -> list(v2)`

## Implementation

### Execution Overview

The `Map` invocations are distributed across multiple machines by
automatically partitioning the input data into a set of $M$ splits.
The input splits can be processed in parallel by different machines.
`Reduce` invocations are distributed by partitioning the intermediate key
space into $R$ pieces using a partitioning function.

Below figure shows the overall flow of a MapReduce operation.

1. The MapReduce library in the user program first splits the
input files into $M$ pieces of typically 16MB to 64MB per piece.
It then starts up many copies of the program on a cluster of machines.
2. One of the copies of the program is special - the master. The
rest are workers that are assigned work by the master. There are
$M$ map tasks and $R$ reduce tasks to assign. The master picks idle
workers and assigns each one a map task or a reduce task.
3. A worker who is assigned a map task reads the contents of the
corresponding input split. It parses key/value pairs out of the
input data and passes each pair to the user-defined `Map` function.
The intermediate key/value pairs produced by the `Map` function
are buffered in memory.
4. Periodically, the buffered pairs are written to local disk, partitioned
into $R$ regions by the partitioning function. The locations of
these buffered pairs on the local disk are passed back to the master.
5. When a reduce worker is notified by the master about these
locations, it use remote procedure calls to read the buffered data
from the local disks of the map workers. When a reduce worker has
read all intermediate data, it sorts it by the intermediate keys
so that all occurrences of the same key are grouped together.
6. The reduce worker iterates over the sorted intermediate data
and for each unique intermediate key encountered, it passes the
key and the corresponding set of intermediate values to the user's
`Reduce` function.
7. When all map tasks and reduce tasks have been completed, the
master wakes up the user program. At this point, the `MapReduce`
call in the user program returns back to the user code.

### Master Data Structures

The master keeps several data structures. For each map task and reduce
task, it stores the state (*idle*, *in-progress*, or *completed*), and
the identity of the worker machine.

For each completed map task, the master stores the locations and sizes
of the $R$ immediate file regions produced by the map task.

### Fault Tolerance

#### Worker Failure

The master pings every worker periodically. If no response is
received from a worker in a certain amount of time, the master marks
the worker as failed. Any map tasks completed by the worker are reset back
to their initial *idle* state, and therefore become eligible for
scheduling on other workers.

Completed map tasks are re-executed on a failure because their
output is stored on the local disks of the failed machine and is
therefore inaccessible.

When a map task is executed by worker $A$ and the later executed by
worker $B$, all workers executing tasks are notified of the re-execution.

#### Master Failure

It's easy to make the master write periodic checkpoints of the master
data structures. If the master task dies, a new copy can be started
from the last checkpoint state.
