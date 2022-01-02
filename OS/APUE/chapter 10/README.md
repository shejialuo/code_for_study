# Signals

## Signal Concepts

First, every signal has a name. These names all begin with the three characters
`SIG`. Signal names are all defined by positive integer constants (the signal number)
in the header `<signal.h>`.

No signal has a signal number of 0. Numerous conditions can generate a signal:

+ The terminal-generated signals occur when users press certain terminal keys.
  Pressing the DELETE key on the terminal (or Control-C on many systems) normally
  causes the interrupt signal (`SIGINT`) to be generated. This is how to stop a
  runaway program.
+ Hardware exceptions generate signals: divide by 0, invalid memory reference,
  and the like. These conditions are usually detected by the hardware, and the kernel
  is *notified*. The kernel generates the appropriate signal for the process.
+ The `kill` function allows a process to send any signal to another process or
  process group.
+ The `kill` command allows us to send signals to other processes. This program
  is just an interface to the `kill` function.
+ Software conditions can generate signals when a process should be notified of
  various events.

We can tell the kernel to do one of three things when a signal occurs. We call
this the *disposition* of the signal, or the *action* of the signal:

+ Ignore the signal. This works for most signals, but two signals can never be
  ignored: `SIGKILL` and `SIGSTOP`.
+ Catch the signal. To do this, we tell the kernel to call a function of ours
  whenever the signal occurs.
+ Let the default action apply.

## signal Function

The simplest interface to the signal features of the UNIX System is the `signal`
function.

```c
#include <signal.h>
void (*signal(int signo, void (*func)(int)))(int);
// Returns: previous disposition of signal if OK, SIG_ERR on error
```

The `signo` argument is just the name of the signal.

The value of func is:

+ the constant `SIG_IGN`, telling the system to ignore the signal.
+ the constant `SIG_DFL`, default value.
+ the address of a function to be called when the signal occurs.

The prototype for the `signal` function states that the function requires two
arguments and returns a pointer to a function that returns nothing.

## Unreliable Signals

In earlier versions of the UNIX System, signals were unreliable. By this we mean
that signals could get lost: a signal could occur and the process would never
know about it. Also, a process had little control over a signal: a process could
catch the signal or ignore it. Sometimes, we would like to tell the kernel to block
a signal: don't ignore it, just remember if it occurs, and tell us later when
we're ready.

One problem with these early versions was that the action for a signal was reset
to its default each time the signal occurred. The classic example from programming
books that described these earlier systems concerns how to handle the interrupt signal:

```c
int sig_int();

signal(SIGINT, sig_int);

sig_int() {
  signal(SIGINT, sig_int); /* reestablish handler for next time */
}
```

Another problem with these earlier systems was that the process was unable to turn
a signal off when it didn't want the signal to occur.

```c
int sig_int();
int sig_int_flag;

main() {
  signal(SIGINT, sig_int);
  while(sig_int_flag == 0)
    pause();
}

sig_int() {
  signal(SIGINT, sig_int);
  sig_int_flag = 1;
}
```

However, there is a window of time when things can go wrong. If the signal occurs
after the test of `sig_int_flag` but before the call to `pause`, the process
could go to sleep forever ((assuming that the signal is never generated again).

## Interrupted System Calls

A characteristic of earlier UNIX Systems was that if a process caught a signal
while the process was blocked in a "slow" system call, the system call was interrupted.
The system call returned an error and `errno` was set to`EINTR`.

To support this feature, the system calls are divided into two categories: the
"slow" system calls and all the others. The slow system calls are those can block
forever:

+ Reads that can block the caller forever if data isn't present with certain file
  types (pipe, terminal devices, and network devices)
+ Writes that can block the caller forever if data can't be accepted immediately
  by these same file types
+ Opens on certain file types that block the caller until some condition occurs
+ The `pause` function and the `wait` function
+ Certain `ioctl` operations
+ Some of the IPC functions

The notable exception to these slow system calls is anything related to disk I/O.

## Reentrant Functions

The Single UNIX Specification specifies the functions that are guaranteed to be
safe to call from within a single handler. These functions are reentrant and are
called *async-signal safe* by the Single UNIX Specification. Besides being reentrant,
they block any signals during operation if delivery of a signal might cause inconsistencies.

```c
static void my_alarm(int signo) {
  struct passwd *rootptr;

  printf("in signal handler\n");
  if ((rootptr = getpwnam("root")) == NULL)
    printf("error\n");
  alarm(1);
}

int main(int argc, char *argv[]) {
  struct passwd *ptr;
  signal(SIGALRM, my_alarm);

  alarm(1);

  for ( ; ; ) {
    if ((ptr = getpwnam("shejialuo")) == NULL)
      printf("getpwnam error\n");
    if (strcmp(ptr->pw_name, "shejialuo") != 0)
      printf("return value corrupted!, pw_name = %s\n", ptr->pw_name);
    }
}
```

When this program was run, these results were random. If we call a nonreentrant
function from a signal handler, the results are unpredictable.

## SIGCLD Semantics

SIGCLD consists of the following behavior:

+ If the process specifically sets its disposition to `SIG_IGN`, children of the
  calling process will not generate zombie processes.
+ If we set the disposition of `SIGCLD` to be caught, the kernel immediately
  checks whether any child processes are ready to be waited for and, if so, calls
  the `SIGCLD` handler.

## Reliable-Signal Terminology and Semantics

We need to define some of the terms used throughout our discussion of signals.

+ A signal is *generated* for a process when the event that causes the signal
  occurs. When the signal is generated, the kernel usually sets a flag of some form
  in the process table.
+ A signal is *delivered* to a process when the action for a signal is taken.
  During the time between the generation of a signal and its delivery, the signal
  is said to be *pending*.
+ A process has the option of *blocking* the delivery of a signal. The system
  determines what to do with a blocked signal when signal is delivered, not when
  it's generated. This allows the process to change the action for the signal before
  it's delivered.
+ Each process has a *signal mask* that defines the set of signals currently
  blocked from delivery to that process.

## kill and raise Functions

The `kill` function sends a signal to a process or a group of processes. The `raise`
function allows a process to send a signal to itself.

```c
#include <signal.h>
int kill(pid_t pid, int signo);
int raise(int signo);
// Both returns: 0 if OK, -1 on error
```

There are four different conditions for the `pid` argument to `kill`.

+ `pid > 0`: The signal is sent to the process whose process ID is `pid`.
+ `pid == 0`: The signal is sent to all processes whose process group ID equals
  the process group ID of the sender and for which the sender has permission to send
  the signal.
+ `pid < 0`: The signal is sent to all processes whose process group ID equals
  the absolute value of `pid` and for which the sender has permission to send the signal.
+ `pid == -1`: The signal is sent to all processes on the system for which the
  sender has permission to send the signal.

## alarm and pause Functions

The `alarm` function allows us to set a timer that will expire at a specified
time in the future. When the timer expires, the `SIGALRM` signal is generated.
If we ignore or don't catch the signal, its default action is to terminate the process.

```c
#include <unistd.h>
unsigned int alarm(unsigned int seconds);
// Returns: 0 or number of seconds until previously set alarm
```

The `pause` function suspends the calling process until a signal is caught.

A common use for `alarm`, in addition to implementing the `sleep` function, is
to put an upper time limit on operations that can block.

```c
static void sig_alrm(int);

int main(int argc, char *argv[]) {
  int n;
  char line[MAXLINE];

  if (signal(SIGALRM, sig_alrm) == SIG_ERR)
    printf("signal(SIGALRM) error\n");
  alarm(10);
  if ((n = read(STDIN_FILENO, line, MAXLNE)) < 0)
    printf("read error");
  alarm(0);
  write(STDOUT_FILENO, line ,n);
  exit(0);
}

static void sig_alrm(int signo) {}
```

This sequence of code is common in UNIX applications, but this program has two problems.

+ A race condition between the first call to `alarm` and the call to `read`.
+ If system calls are automatically restarted, the `read` is not interrupted when
  the `SIGALRM` signal handler returns.

Let's redo the preceding example using `longjmp`.

```c
static void sig_alrm(int signo);
static jmp_buf env_alrm;

int main(int argc, char *argv[]) {
  int n;
  char line[MAXLINE];

  if (signal(SIGALRM, sig_alrm) == SIG_ERR)
    printf("signal(SIGALRM) error\n");
  if (setjmp(env_alrm) != 0)
    exit(0);
  alarm(10);
  if ((n = read(STDIN_FILENO, line, 3)) < 0)
    printf("read error");
  alarm(0);
  write(STDOUT_FILENO, line ,n);
  exit(0);
}

static void sig_alrm(int signo) {
  longjmp(env_alrm, 1);
}
```

## Signal Sets

We need a singal set to represent multiple signals. POSIX.1 defines the data type
`sigset_t` to contain a signal set and the following five functions to manipulate
signal sets.

```c
#include <signal.h>
int sigemptyset(sigset_t *set);
int sigfillset(sigset_t *set);
int sigaddset(sigset_t *set, int signo);
int sigdelset(sigset_t *set, int signo);
// All four return 0 if OK, -1 on error
int sigismember(const sigset_t *set, int signo);
// Returns 1 if true, 0 if false, -1 on error
```

## sigprocmask Function

A process can examine its signal mask, change its signal mask, or perform both
operations in one step by calling the following function.

```c
#include <signal.h>
int sigprocmask(int how, const sigset_t * set,
                sigset_t *restrict oset);
// Returns: 0 if OK, -1 on error
```

First, if `oset` is a non-null pointer, the current signal mask for the process
is returned through `oset`.

Second, if `set` is a non-null pointer, the `how` argument indicates how the
current mask is modified:

+ `SIG_BLOCK`: The new signal mask for the process is the union of its current
  signal mask and the signal set pointed to by `set`.
+ `SIG_UNBLOCK`: The new signal mask for the process is the intersection of its
  current signal mask and the complement of the signal set pointed to by `set`.
+ `SIG_SETMASK`: The new signal mask for the process is replaced by the value of
  the signal set pointed to by `set`.

## sigpending Function

The `sigpending` function returns the set of signals that are blocked from delivery
and currently pending for the calling process.

```c
#include <signal.h>
int sigpending(sigset_t *set);
// Returns 0 if OK, -1 on error
```

For example:

```c
int main(int argc, char *argv[]) {
  sigset_t newmask, oldmask, pendmask;

  if(signal(SIGINT,sig_quit) == SIG_ERR)
    printf("can't catch SIGQUIT\n");

  sigemptyset(&newmask);
  sigaddset(&newmask, SIGQUIT);

  if(sigprocmask(SIG_BLOCK, &newmask, &oldmask) < 0)
    printf("SIG_BLOCK error");

  sleep(5);

  if(sigpending(&pendmask) < 0)
    printf("sigpending error\n");
  if(sigismember(&pendmask, SIGQUIT))
    printf("SIGQUIT pending\n");

  if (sigprocmask(SIG_SETMASK, &oldmask, NULL) < 0)
    printf("SIG_SETMASK error\n");
    printf("SIGQUIT unblocked\n");

  sleep(5);
  exit(0);
}

static void sig_quit(int signo) {
  printf("caught SIGQUIT\n");
  if(signal(SIGINT,sig_quit) == SIG_ERR)
    printf("can't reset SIGQUIT\n");
}
```

## sigaction Function

The `sigaction` function allows us to examine or modify the action associated with
a particular signal.

```c
#include <signal.h>
int sigaction(int signo, const struct sigaction *restrict act,
              struct sigaction *restrict oact);
// Returns: 0 if OK, -1 on error
```

This function uses the following structure:

```c
struct sigaction {
  void (*sa_handler)(int); // handler
  sigset_t sa_mask;
  int sa_flags;
  void (*sa_sigaction)(int, siginfo_t *, void *);
};
```

## abort Function

The `abort` function causes abnormal program termination.

```c
#include <stdlib.h>
void abort(void);
// This functions never returns
```

## Job-Control Signals

POSIX.1 consider six to be job-control signals:

+ `SIGCHLD`: Child process has stopped or terminated.
+ `SIGCONT`: Continue process, if stopped.
+ `SIGSTOP`: Stop signal.
+ `SIGTSTP`: Interactive stop signal
+ `SIGTTIN`: Read from controlling terminal by background process group member.
+ `SIGTTOU`: Write to controlling terminal by background process group member.

## Signal Names and Numbers

In this section, we describe how to map between signal numbers and names. Some
systems provide the array:

```c
extern char *sys_siglist[];
```

The array index is the signal number, giving a pointer to the character string
name of the signal.

To print the character string corresponding to a signal number in a portable manner,
we can use the `psignal` function.

```c
#include <signal.h>
void psignal(int signo, const char *msg);
```
