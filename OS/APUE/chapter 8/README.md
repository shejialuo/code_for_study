# Process Control

## Process Identifiers

Every process has a unique process ID, a non-negative integer. Because the process
ID is the only well-known identifier of a process that is always unique, it is
often used as a piece of other identifiers, to guarantee uniqueness.

There are some special processes, but the details differ from implementation to
implementation. Process ID 0 is usually the scheduler process and is often known
as `swapper`. It is a system process.

Process ID 1 is usually the `init` process and is invoked by the kernel at the
end of the bootstrap procedure. The program file for this process was `/etc/init`
in older versions of the UNIX System and is `/sbin/init` in newer versions.
This process is responsible for bringing up a UNIX system after the kernel has
been bootstrapped. `init` usually reads the system-dependent initialization
files: `/etc/rc*`, `/etc/inittab` and `/etc/init.d` to bring the system to a
certain state. The `init` process never dies. It is a normal user process.Process ID 1 is usually the `init` process and is invoked by the kernel at the
end of the bootstrap procedure. The program file for this process was `/etc/init`
in older versions of the UNIX System and is `/sbin/init` in newer versions.
This process is responsible for bringing up a UNIX system after the kernel has
been bootstrapped. `init` usually reads the system-dependent initialization
files: `/etc/rc*`, `/etc/inittab` and `/etc/init.d` to bring the system to a
certain state. The `init` process never dies. It is a normal user process.

In addition to the process ID, there are other identifiers for every process.
The following functions return these identifiers.

```c
#include <unistd.h>
pid_t getpid();
// Returns: process ID of calling process
pid_t getppid();
// Returns: parent process ID of calling process
uid_t getuid();
// Returns: real user ID of calling process
uid_t geteuid();
// Returns: effective user ID of calling process
gid_t getgid();
// Returns: real group ID of calling process
gid_t getegid();
// Returns: effective group ID of calling process
```

## fork Function

An existing process can create a new one by calling the `fork` function:

```c
#include <unistd.h>
pid_t fork(void);
// Returns: 0 in child, process ID of child in parent, -1 on error
```

The new process created by `fork` is called the *child process*.
This function is called once but returns twice.

Both the child and the parent continue executing with the instruction that
follows the call to `fork`. The child is a copy of the parent. For example,
the child gets a copy of the parent's data space, heap, and stack. Note that
this is a copy for the child; the parent and the child do not share these portions
of memory. The parent and the child do share the text segment.

Modern implementations don't perform a complete copy of the parent's data,
stack, and heap, since a `fork` is often followed by an `exec`. Instead, a technique
called **copy-on-write** is used. These regions are shared by the parent and the
child and have their protection changed by the kernel to read-only. If either
process tries to modify these regions, the kernel then makes a copy of that piece
of memory only.

In general, we never know whether the child starts executing before the parent,
or vice versa. The order depends on the scheduling algorithm used by the kernel.

One characteristic of `fork` is that all file descriptors that are open in the
parent are duplicated in the child as below.

![Sharing of open files between parent and child after fork](https://i.loli.net/2021/05/18/pMCYvADPGez3LyR.png)

It is important that the parent and the child share the same file offset. If both
parent and child write to the same descriptor, without any form of synchronization,
their output will be intermixed.

## vfork Function

The `vfork` function was intended to create a new process for the purpose of
executing a new program. The `vfork` function creates the new process, just like
`fork`, without copying the address space of the parent into the child, as the
child won't reference that address space; the child simply calls `exec` right
after the `vfork`. Instead, the child runs in the address space of the parent
until it calls either `exec` or `exit`.

Another difference between the two functions is that `vfork` guarantees that the
child runs first, until the child calls `exec` or `exit`. When the child calls
either of these functions, the parent resumes.

## exit Functions

What happens if the parent terminates before the child? The answer is that the
`init` process becomes the parent process of any process whose parent terminates.
In such a case, we say that the process has been inherited by `init`. What normally
happens is that whenever the terminating process is the parent of any process
that still exists.

Another condition we have to worry about is when a child terminates before its
parent. If the child completely disappeared, the parent wouldn't be able to fetch
its termination status when and if the parent was finally ready to check if the
child had terminated. The kernel keeps a small amount of information for every
terminating process calls `wait` or `waitpid`. Minimally, this information consists
of the process ID, the termination status of the process, and the amount of the
CPU time taken by the process.

In UNIX System terminology, a process that has terminated, but whose parent has
not yet waited for it, is called a *zombie*. If we write a long-running program
that forks many child processes, they become zombies unless we wait for them and
fetch their termination status.

## wait and waitpid Functions

When a process terminates, either normally or abnormally, the kernel notifies
the parent by sending the `SIGCHLD` signal to the parent. We need to be aware
that a process that calls `wait` or `waitpid` can

+ Block, if all of its children are still running
+ Return immediately with the termination status of a child, if a child has
terminated and is waiting for its termination status to be fetched

```c
#include <sys/wait.h>
pid_t wait(int *statloc);
pid_t waitpid(pid_t pid, int *statloc, int options);
// Both return: process ID if OK, 0, or -1 on error
```

The differences between these two functions are as follows:

+ The `wait` function can block the caller until a child process terminates,
  whereas `waitpid` has an option that prevents from blocking.
+ The `waitpid` function doesn't wait for the child that terminates first;
  it has a number of options that control which process it waits for.

The interpretation of the `pid` argument for `waitpid` depends on its value:

+ `pid == -1`: Waits for any child process. `waitpid` is equivalent to `wait`.
+ `pid > 0`: Waits for the child whose process ID equals `pid`.
+ `pid == 0`: Waits for any child whose process group ID equals that of the calling
  process.
+ `pid < -1`: Waits for any child whose process group ID equals the absolute value
  of `pid`.

## waitid Function

The Single UNIX Specification includes an additional function to retrieve the
exit status of a process. The `waitid` function is similar to `waitpid`, but
provides extra flexibility.

```c
#include <sys/wait.h>
int waitid(idtype_t idtype, id_t id, siginfo_t *infop, int options);
// Returns: 0 if OK, -1 on error
```

### wait3 and wait4 Functions

Most UNIX system implementations provide two additional functions: `wait3` and
`wait4`. The only feature provided by these two functions that isn't provided by
the `wait`, `waitpid` and `waitpid` functions is an additional argument that
allows the kernel to return a summary of the resources used by the terminated and
process and all its children processes.

```c
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/resource.h>
pid_t wait3(int *statloc, int options, struct rusage *rusage);
pid_t wait4(pid_t pid, int *statloc, int options, struct rusage *rusage);
// Both return: process ID if OK, 0, or −1 on error
```

## exec Functions

When a process calls one of the `exec` functions, that process is completely
replaced by the new program, and the new program starts executing at its `main`
function; `exec` merely replaces the current process with a brand-new program
form disk.

There are seven different `exec` functions:

```c
#include <unistd.h>
int execl(const char *pathname, const char *arg0, ... /* (char *)0 */ );
int execv(const char *pathname, char *const argv[]);
int execle(const char *pathname, const char *arg0, ...
/* (char *)0, char *const envp[] */ );
int execve(const char *pathname, char *const argv[], char *const envp[]);
int execlp(const char *filename, const char *arg0, ... /* (char *)0 */ );
int execvp(const char *filename, char *const argv[]);
int fexecve(int fd, char *const argv[], char *const envp[]);
// All seven return: −1 on error, no return on success
```

The first difference in these functions is that the first four take a pathname
argument, the next two take a filename argument, and the last one takes a file
descriptor argument. When a `filename` argument is specified,

+ If `filename` contains a slash, it is taken as a pathname.
+ Otherwise, the executable file is searched for in the directories specified by
  the `PATH` environment variable.

If either `execlp` or `execvp` finds an executable file using one of the path
prefixes, but the file isn't a machine executable that was generated by the
linker, the function assumes that the file is a shell script and tries to
invoke `/bin/sh` with the `filename` as input to the shell.

With `fexecve`, we avoid the issue of finding the correct executable file
altogether and rely on the caller to do this. By using a file descriptor, the
caller can verify the file is in fact the intended file and execute it without a
race.

The next difference concerns the passing of the argument list (`l` stands for
list and `v` stands for vector). The functions `execl`, `execlp`, and `execle`
require each of the command-line arguments to the new program to be specified as
separate arguments. We mark the end of the arguments with a null pointer. For
the other four functions, we have to build an array of pointers to the arguments,
and the address of this array is the argument to these three functions.

The final difference is the passing of the environment list to the new program.
The three functions whose names end in an `e` allow us to pass a pointer to an
array of pointers to the environment strings.

In many UNIX system implementations, only one of these seven functions, `execve`,
is a system call within the kernel. The other six are just library functions
that eventually invoke this system call. We can illustrate the relationship among
these seven functions below.

![Relationship of the seven exec functions](https://i.loli.net/2021/05/24/bM8HLGijFn59I6S.png)

## Changing User IDs and Group IDs

We can set the real user ID and effective user ID with the `setuid` function.
Similarly, we can set the real group ID and the effective group ID with the
`setgid` function:

```c
#include <unistd.h>
int setuid(uid_t uid);
int setgid(gid_t gid);
// Both return: 0 if OK, -1 on error
```

There are rules for who can change the IDs. Let's only the user ID for now.
(Everything we describe for the user ID also applies to the group ID.)

+ If the process has superuser privileges, the `setuid` function sets the real
  user ID, effective user ID and saved set-user-ID to `uid`.
+ If the `uid` equals either the real user ID or the saved set-user-ID, `setuid`
  sets only the effective user ID to `uid`.
+ If neither of these two conditions is true, `errno` is set to `EPERM` and `-1`
  is returned.

We can make few statements about the three user IDs that the kernel maintains.

+ Only a superuser can change the real user ID. Normally, the real user
ID is set by the `login` program when we log in and never changes.
+ The effective user ID is set by the `exec` functions only if the set-user-ID bit
  is set for the program file. If the set-user-ID bit is not set, the `exec`
  functions leave the effective user ID as its current value. Naturally, we
  can't set the effective user ID to any random value.
+ The saved set-user-ID is copied from the effective user ID by `exec`.

We can summary in the following picture.

![Ways to change the three user IDs](https://i.loli.net/2021/10/10/jWlIYb179LhpSrk.png)

Historically, BSD supported the swapping of the real user ID and the
effective user ID with the `setreuid` function.

```c
#include <unistd.h>
int setreuid(uid_t ruid, uid_t euid);
int setregid(gid_t rgid, gid_t egid);
// Both return: 0 if OK, −1 on error
```

We can apply a value of -1 for any of the arguments to indicate that
the corresponding ID should remain unchanged.

The rule is simple: an unprivileged user can always swap between the
real user ID and the effective user ID. This allows a set-user-ID
program to swap to the user's normal permissions and swap back again
later for set-user-ID operations.

POSIX.1 includes the two functions `seteuid` and `setegid`. These
functions are similar to `setuid` and `setgid`, but only the effective
user ID and effective group ID is changed.

```c
#include <unistd.h>
int seteuid(uid_t uid);
int setegid(gid_t gid);
// Both return: 0 if OK, −1 on error
```

An unprivileged user can set its effective user ID to either its real
user ID or its saved set-user-ID. For a privileged user, only the
effective user ID is set to `uid`.

## Interpreter Files

All contemporary UNIX systems support interpreter files. These files are text
files that begin with a line of the form.

```sh
#! pathname [optional-argument]
```

The space between the exclamation point and the `pathname` is optional.

The recognition of these files is done within the kernel as part of
processing the `exec` system call. The actual file that gets executed by
the kernel as part of processing the `exec` system call.

## system Function

It is convenient to execute a command string from within a program. ISO C defines
the `system` function, but its operation is strongly dependent.

```c
#include <stdlib.h>
int system(const char *cmdstring);
```

`system` is implemented by calling `fork`, `exec` and `waitpid`.
