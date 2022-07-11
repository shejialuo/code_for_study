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
