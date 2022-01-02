# Threads

## Thread Concepts

A typical UNIX process can be thought of as having a single thread of control:
each process is doing only one thing at a time. With multiple threads of control,
we can design our programs to do more than one thing at a time within a single
process, with each thread handling a separate task.

A thread consists of the information necessary to represent an execution context
within a process. This includes a *thraed* ID that identifies the thread within
a process, a set of register values, a stack, a scheduling priority and policy,
a signal mask, an `errno` variable, and thread-specific data.

## Thread Identification

Every thread has a thread ID. A thread ID is represented by the `pthread_t` data
type. Implementations are allowed to use a structure to represent the `pthread_t`
data type, so portable implementations can't treat them as integers. Therefore,
a function must be used to compare two thread IDs.

```c
#include <pthread.h>
int pthread_equal(pthread_t tid1, pthread_t tid2);
// Returns: nonzero if equal, 0 otherwise
```

A thread can obtain its own thread ID by calling the `pthread_self` function.

```c
#include <pthread.h>
pthread_t pthread_self(void);
// Returns: the thread ID of the calling thread
```

## Thread Creation

Additional threads can be created by calling the `pthread_create` function.

```c
#include <pthread.h>
int pthread_create(pthread_t *restrict tidp,
                   const pthread_attr_t *restrict attr,
                   void *(*start_rtn)(void *), void *restrict arg);
// Returns 0 if OK, error number on failure.
```

The memory location pointed to by `tidp` is set to the thread ID of the newly created
thread when `pthread_create` returns successfully. The `attr` argument is used
to customize various thread attributes.

The newly created thread starts running at the address of the `start_rtn` function.
This function takes a single argument, `arg`, which is a typeless pointer. If
you nned to pass more than one argument to the `start_rtn` function, then you
need to store them in a structure.

When a thread is created, there is no guarantee which will run first.

## Thread Termination

If any thread within a process calls `exit`, `_Exit`, or `_exit`, then the entire
process terminates. Similarly, when the default action it to terminate the process,
a signal sent to a thread will terminate the entire process.

A single thraed can exit in three ways.

+ The thread can simply return from the start routine.
+ The thread can be canceled by another thread in the same process.
+ The thread canc all `pthread_exit`.

```c
#include <pthread.h>
void pthread_exit(void *rval_ptr);
```

The `rval_ptr` argument is a typeless pointer, similar to the single argument passed
to the start routine. This pointer is available to other threads in the process by
calling the `pthread_join` function.

```c
#include <pthread.h>
int pthread_join(pthread_t thread, void **rval_ptr);
// Returns 0 if OK, error number on failure.
```

The calling thread will block until the specified thread calls `pthread_exit`,
returns from its start routine, or is canceled.

By calling `pthread_join`, we automatically place the thread with which we're joining
in the detached state so that its resources can be recovered.

The typeless pointer passed to `pthread_create` and `pthread_exit` can be used to
pass more than a single value. The pointer can be used to pass the address of a
structure containing more complex information. Be careful that the memory used for
the structure is till valid when the caller has completed. If the structure was
allocated on the caller's stack, the memory contents might have changed by the
time the structure is used. If a thread allocates a structure on its stack and passes
a pointer
to this structure to `pthread_exit`, then the stack might be destroyed and its
memory reused for something else by the time rge caller of `pthread_join` tries
to use it.

One thread can request that another in the same process be canceld by calling
the `pthread_cancel` function.

```c
#include <pthread.h>
int pthread_cancel(pthread_t tid);
// Returns 0 if OK, error number on failure.
```

In the default circumstances, `pthread_cancel` will cause the thread to behave
as if had called `pthread_exit` with an argument of `PTHREAD_CANCELED`.

A thread can arrange for functions to be called when it exits.

```c
#include <pthread.h>
void pthread_cleanup_push(void (*rtn)(void *), void *arg);
void pthread_cleanup_pop(int execute).
```

The `pthread_cleanup_push` function schedules the cleanup function, `rtn`, to be
called with the single argument `arg`, when the thread performs one of the
following actions:

+ Makes a call to `pthread_exit`.
+ Responds to a cancellation request.
+ Makes a call to `pthread_cleanup_pop` with a nonzero argument.

We can detach a thread by calling `pthread_detach`.

```c
#include <pthread.h>
int pthread_detach(pthread_t tid);
// Returns 0 if OK, error number on failure.
```
