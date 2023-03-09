# Interprocess Communication

## Pipes

Pipes are the oldest form of UNIX System IPC and are provided by all
UNIX systems. Pipes have two limitations.

+ They have been half duplex.
+ Pipes can be used on between processes that have a common ancestor.

A pipe is created by calling the `pipe` function. Two file descriptors are returned
through the `fd` argument: `fd[0]` is open for reading, and `fd[1]` is open for
writing. The output of `fd[1]` is the input for `fd[0]`.

```c
#include <unistd.h>
int pipe(int fd[2]);
// Return 0 if OK, -1 on error
```

A pipe in a single process is next to useless. Normally, the process that calls `pipe`
then calls `fork`, creating an IPC channel from the parent to the child, or vice versa.

What happens after the `fork` depends on which direction of data flow we want. For a
pipe from the parent to the child, the parent close the read end of the pipe, and the child
closes the write end. For a pipe from the child to the parent, the parent closes `fd[1]`,
and the child closes `fd[0]`.

When one end of a pipe is closed, two rules apply:

+ If we `read` from a pipe whose write end has been closed, `read` returns 0 to indicate
an end of file after all the data has been read.
+ If we `write` to a pipe whose read end has been closed, the signal `SIGPIPE` is generated.
If we either ignore the signal or catch it and return from the signal handler, `write`
returns -1 with `errno` set to `EPIPE`.

[pipeExample1.c](./pipeExample1.c)

[pipeExample2.c](./pipeExample2.c)

## popen and pclose Functions

Since a common operation is to create a pipe to another process to either read its output
or send it input, the standard I/O library provided the `popen` and `pclose` functions.
These two functions handle all the dirty work that we've been doing.

```c
#include <stdio.h>

FILE *popen(const char *cmdstring, const char *type);

int plcose(FILE *fp);
```

## FIFOs

FIFOs are sometimes called named pipes. Unnamed pipes can be used only between related
processes when a common ancestor has created the pipe. With FIFOs, however, unrelated
process can exchange data.

Creating a FIFO is similar to creating a file.

```c
#include <sys/stat.h>
int mkfifo(const char *path, mode_t mode);
int mkfifoat(int dfd, const char *path, mode_t mode);
```

## XSI IPC

The three types of IPC that we call XSI IPC: message queues, semaphores, and shared memory.

### Identifiers and Keys

Each *IPC structure* in the kernel is referred to by a non-negative integer *identifier*.
To send a message or to fetch a message from a message queue, for example, all we need to
know is the identifier for the queue. IPC identifiers are not small integers. Indeed,
when a given IPC structure is created and then removed, the identifier associated with
that structure continually increases until it reaches the maximum positive value for an
integer and then wraps around to 0.

The identifier is an internal name for an IPC object. Cooperating processes need an
external naming scheme to be able to rendezvous using the same IPC object. For this purpose,
an IPC object is associated with a *key* that acts as an external name.

Whenever an IPC structure is being created, a key must be specified. The data type of this
key is the primitive system data type `key_t`. There are various ways for a client and server
to rendezvous at the same IPC structure.

+ The server can create a new IPC structure by specifying a key of `IPC_PRIVATE` and store the
returned identifier somewhere (such as a file) for the client to obtain.
+ The client and the server can agree on a key by defining the key in a common header.
+ The client and the server can agree on a pathname and project ID and call the function `ftok`
to convert these two values into a key.

### Permission Structure

XSI IPC associates an `ipc_perm` structure with each IPC structure. This structure defines
the permissions and owner and includes at least the following members:

```c
struct ipc_perm {
  uid_t uid; /* owner’s effective user ID */
  gid_t gid; /* owner’s effective group ID */
  uid_t cuid; /* creator’s effective user ID */
  gid_t cgid; /* creator’s effective group ID */
  mode_t mode; /* access modes */
};
```

### Advantages and Disadvantages

A fundamental problem with XSI IPC is that IPC structures are systemwide and do not have a
reference count. Another problem with XSI IPC is that these IPC structures are not known by
names in the file system.
