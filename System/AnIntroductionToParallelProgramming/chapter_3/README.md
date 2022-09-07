# 3. Distributed memory programming with MPI

The implementation of message-passing that we'll be using is called MPI, which
is an abbreviation of *Message-Passing Interface*. MPI is not a new
programming language. It defines a *library* of functions that can
be called from C and Fortran programs.

## 3.1 Getting started

Let's write a program similar to "hello, world" that makes some use
of MPI. Instead of having each process simply print a message, we'll
designate one process to do the output, and the other processes will
send it messages, which it will print.

In parallel programming, it's common for the processes to be
identified by nonnegative integer *ranks*. So if there are $p$
processes, the processes will have ranks $0,1,2,\dots, p - 1$.
For our parallel "hello, world", let's make process 0 the designated
process, and the other processes will send it messages. See
[mpi_hello](./mpi_hello.c)

### 3.1.1 Compilation and execution

Many systems use a command called `mpicc` for compilation:

```sh
mpicc -g -Wall -o mpi_hello mpi_hello.c
```

Typically, `mpicc` is a script that's a *wrapper* for the C compiler.
A *wrapper script* is a script whose main purpose is to run some
program. Many systems also support program startup with `mpiexec`:

```sh
mpiexec -n <number of processes> ./mpi_hello
```

### 3.1.2 MPI programs

All of the identifiers defined by MPI start with the string `MPI_`. All
of the letters in MPI-defined macros and constants are capitalized.

### 3.1.3 MPI_Init and MPI_Finalize

The call to `MPI_Init` tells the MPI system to do all of the necessary
setup. As a rule of thumb, no other MPI functions should be called
before the program calls `MPI_Init`. Its syntax is

```c
int MPI_init(
  int* argc_p,
  char*** argv_p);
```

The arguments, `argc_p` and `argv_p`, are pointers to the arguments to
`argc` and `argv`. However, when our program doesn't use these arguments,
we can just pass `NULL` for both. Like most MPI functions, `MPI_Init` returns
an `int` error code.

The call to `MPI_Finalize` tells the MPI system that we're done using
MPI, and that any resources allocated for MPI can be freed. In general,
no MPI functions should be called after the call to `MPI_Finalize`.

```c
#include <mpi.h>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  ...
  MPI_Finalize();
  return 0;
}
```

### 3.1.4 Communicators, MPI_Comm_size, and MPI_Comm_rank

In MPI a *communicator* is a collection of processes that can send
messages to each other. One of the purposes of `MPI_Init` is
to define a communicator that consists of all of the processes started
by the user when starting the program. This communicator is called
`MPI_COMM_WORLD`.

```c
int MPI_Comm_size(
  MPI_Comm comm,
  int* comm_sz_p);

int MPI_Comm_rank(
  MPI_Comm comm,
  int* my_rank_p);
```

### 3.1.5 SPMD programs

A single program is written so that different processes carry out
different actions, and this can be achieved by simply having the
processes branch on the basis of their process rank, which is called
*single program, multiple data* or *SPMD*.

### 3.1.6 MPI_Send

```c
int MPI_Send(
  void* msg_buf_p,
  int msg_size,
  MPI_Datatype msg_type,
  int dest,
  int tag,
  MPI_Comm communicator);
```

Since C types can't be passed as arguments to functions, MPI defines
a special type, `MPT_Datatype`. MPI also defines a number of
constant values for this type.

### 3.1.7 MPT_Recv

The first six arguments to `MPI_Recv` correspond to the first
six arguments of `MPI_Send`:

```c
int MPI_Send(
  void* msg_buf_p,
  int msg_size,
  MPI_Datatype msg_type,
  int dest,
  int tag,
  MPI_Comm communicator,
  MPI_Status* status_p);
```

### 3.1.8 Message matching

Suppose process $q$ calls `MPI_Send` with

```c
MPI_Send(send_buf_p, send_buf_sz, send_type, dest, send_tag, send_comm);
```

Also suppose that process $r$ calls `MPI_Recv` with

```c
MPI_Recv(recv_buf_p, recv_buf_sz, recv_type, src, recv_tag, recv_comm,&status);
```

Then the message sent by $q$ with the above call to `MPI_Send` can
be received by $r$ with the call to `MPI_Recv` if

+ `recv_comm = send_comm`
+ `recv_tag = send_tag`
+ `dest = r`
+ `src = q`

These conditions aren't quite enough for the message to be
*successfully* received, however. The parameters specified by
the first three pairs of arguments must specify compatible buffers.

+ If `recv_type=send_type` and `recv_buf_sz >= send_buf_sz`,
the the message sent by $q$ can be successfully received by $r$.

## 3.2 The trapezoidal rule in MPI

Let's write a program that implements the trapezoidal rule for
numerical integration.

### 3.2.1 The trapezoidal rule

We can use the trapezoidal rule to approximate the area between
the graph a function, $y = f(x)$. It easy to calculate the area
of the trapezoid.

$$
\frac{h}{2}[f(x_{i}) + f(x_{i + 1})]
$$

We can choose the $n$ subintervals so that they would all have the same
length, we also know that if the vertical lines bounding the
region are $x = a$ and $x = b$, then we have

$$
h = \frac{b - a}{n}
$$

Thus if we call the leftmost endpoint $x_{0}$, and the rightmost
endpoint $x_{n}$ we have that

$$
x_{0} = a, x_{1} = a + h, x_{2} = a + 2h, \dots, x_{n - 1} = a + (n - 1)h, x_{n} = b.
$$

And the sum of trapezoid areas is

$$
h[f(x_{0}) /2 + f(x_{1}) + f(x_{2}) + \cdots + f(x_{n - 1} + f(x_{n}) / 2)]
$$

The pseudocode for a serial program might look something like this:

```pseudocode
/* Input: a , b, n */
h = (b - a) / n;
approx = (f(a) + f(b)) / 2.0;
for (i = 1; i < n; ++i) {
  x_i = a + i * h;
  approx += f(x_i);
}
approx = h * approx;
```

### 3.2.2 Parallelizing the trapezoidal rule

In the partitioning phase, we usually try to identify as many tasks
as possible. For the trapezoidal rule, we might identify two types
of tasks: one type is find the area of a single trapezoid, and
the other is computing the sum of these areas.

See [mpi_trapezoidal](./mpi_trapezoidal1.c)

## 3.3 Dealing with I/O

We need to address the problem of getting input from the user.

### 3.3.1 Output

In both the "greetings" program and the trapezoidal rule program,
we've assumed that process 0 can write to `stdout`. Although the MPI
standard doesn't specify which processes have access to which I/O
devices, virtually all MPI implementations allow *all* the processes
in `MPI_COMM_WORLD` full access to `stdout` and `stderr`.

However, most MPI implementations don't provide any automatic scheduling
of access to these devices.

```c
#include <stdio.h>
#include <mpi.h>

int main() {
  int my_rank, comm_sz;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  printf("Proc %d of %d > Does anyone have a toothpick?\n", my_rank, comm_sz);
  MPI_Finalize();
}
```

### 3.3.2 Input

Unlike output, most MPI implementations only allow process 0 in
`MPI_COMM_WORLD` access to `stdin`.

## 3.4 Collective communication

If we pause for a moment and think about our trapezoidal rule program,
we can find several things that we might be able to improve on.
One of the most obvious is the "global sum" after each process
has computed its part of the integral. Each process with rank
greater than 0 is "telling process 0 what to do" and then quitting.
Sometimes it does happen that this is the best we can do in a
parallel program. But we could do better.

### 3.4.1 Tree-structured communication

We might use a binary tree structure, like below. In this diagram,
initially processes 1, 3, 5, and 7 send their values to processes 0, 2, 4, and 6,
respectively. Then processes 0, 2, 4, and 6 add the received values
to the original values, and the process is repeated twice:

+ Process 2 and 6 send their new values to processes 0 and 4, respectively.
Process 0 and 4 add the received values into their new values.
+ Process 4 sends its newest value to process 0. Process 0 adds
the received value to its newest value.

![A tree-structure global sum](https://s2.loli.net/2022/08/29/i8vSwcdCAVGRnZs.png)

This is well and good, but coding this tree-structured global
sum would take a quite a bit of work.

### 3.4.2 MPI_Reduce

With virtually limitless possibilities, it's unreasonable to
expect each MPI programmer to write an optimal global-sum function, so MPI
specifically protects programmers against this trap of endless optimization
by requiring that MPI implementations include implementations of
global sums.

Now a "global-sum function" will obviously require communication. However,
unlike the `MPI_Send` and `MPI_Recv` pair, the global-sum function may
involve more than two processes. In MPI parlance, communication
functions that involve all the processes in a communicator are
called *collective communications*. `MPI_Send` and `MPI_Recv` are
often called *point-to-point* communications.

In fact, global-sum is just a special case of an entire class
of collective communications. MPI generalized the global-sum function
with `MPI_Reduce`

```c
int MPI_Reduce(
  void* input_data_p,
  void* output_data_p,
  int count,
  MPI_Datatype datatype,
  MPI_Op operator,
  int dest_process,
  MPI_Comm comm
);
```

### 3.4.3 Collective vs. point-to-point communications

It's important to remember that collective communications differ in
several ways from point-to-point communications.

1. *All* the processes in the communicator must call the same collective function.
2. The arguments passed by each process to an MPI collective communication
must be "compatible".
3. The `output_data_p` argument is only used on `dest_process`.
4. Point-to-point communications are matched on the basis of tags
and communicators. Collective communications don't use tags.

### 3.4.4 MPI_Allreduce

Sometimes all of the processes need the result of a global sum
to complete some larger computation.

```c
int MPI_Allreduce(
  void* input_data_p,
  void* output_data_p,
  int count,
  MPI_Datatype datatype,
  MPI_Op operator,
  MPI_Comm comm
);
```

### 3.4.5 Broadcast

If we can improve the performance of the global sum in our trapezoidal
rule program by replacing a loop of receives on process 0 with
a tree structured communication, we ought to be able to do something
similar with the distribution of the input data. In fact, if
we simply "reverse" the operation, we obtain the tree-structured
communication shown below and we can use this structure to distribute
the input data. A collective communication in which data belonging
to a single process is sent to all of the processes in the communicator
is called a *broadcast*.

![A tree-structure broadcast](https://s2.loli.net/2022/09/07/MczviLTZqoKpJUt.png)

```c
int MPI_Bcast(
  void* data_p,
  int count,
  MPI_Datatype datatype,
  int source_proc,
  MPI_Comm comm
);
```

### 3.4.6 Data distributions

Suppose we want to write a function that computes a vector sum:

$$
\begin{align*}
\mathbf{x} + \mathbf{y} &= (x_{0}, x_{1}, \dots, x_{n} - 1) + (y_{0}, y_{1}, \dots, y_{n - 1}) \\
&= (x_{0} + y_{0}, x_{1} + y_{1}, \dots, x_{n - 1 + y_{n - 1}}) \\
&= (z_{0}, z_{1}, \dots, z_{n - 1}) \\
&= \mathbf{z}
\end{align*}
$$

We might specify that the tasks are just the additions of corresponding
components. Then there is no communication between the tasks,
and the problem of parallelizing vector addition boils down to
aggregating the tasks and assigning them to the cores. This is
often called a *block partition* of the vector.

An alternative to a block partition is a *cyclic partition*. In a
cyclic partition, we assign the components in a round-robin fashion.

A third alternative is a *block-cyclic partition*. The idea here
is that instead of using a cyclic distribution of individual
components, we use a cyclic distribution of *blocks* of components.

### 3.4.7 Scatter

Now suppose we want to test our vector addition function. It would
be convenient to be able to read the dimension of the vectors and then
read in the vectors $\mathbf{x}$ and $\mathbf{y}$. We already know
how to read in the dimension of the vectors: process 0 can prompt
the user, read in the value, and broadcast the value to the other
processes. However, this could be very wasteful.

We might try writing a function that reads in an entire vector
on process 0, but sends the needed components to each of the other
processes. For the communication MPI provides just such a function:

```c
int MPI_Scatter(
  void* send_buf_p,
  int send_count,
  MPI_Datatype send_type,
  void *recv_buf_p,
  int recv_count,
  MPI_Datatype recv_type,
  int src_proc,
  MPI_Comm comm
);
```

### 3.4.8 Gather

Of course, out test program will be useless unless we can see the
result of our vector addition. So we need to write a function for
printing out a distributed vector. Our function can collect all
of the components of the vector onto process 0, and the process 0
can print all of the components. The communication in this function
can be carried out by `MPI_Gather`:

```c
int MPI_Gather(
  void* send_buf_p,
  int send_count,
  MPI_Datatype send_type,
  void* recv_buf_p,
  int recv_count,
  MPI_Datatype recv_type,
  int desc_proc,
  MPI_Comm comm
);
```
