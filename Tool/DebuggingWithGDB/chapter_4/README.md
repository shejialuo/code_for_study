# Chapter 4

When you run a program under GDB, you must first generate debugging
information when you compile it.

## 4.1 Compiling for Debugging

In order to debug a program effectively, you need to generate debugging
information when you compile it. This debugging information is stored in
the object file.

## 4.2 Starting your Program

Use the `run` command to start your program under GDB. If you are running your
program in an execution environment that supports processes, `run` creates
an inferior process and makes that process run your program.

The execution of a program is affected by certain information it receives
from its superior. `GDB` provides ways to specify this information, which you
must do *before* starting your program. This information is divided into
four categories:

+ The *arguments*
+ The *environment*
+ The *working directory*
+ The *standard input and output*

## 4.3 Your Program's Arguments

+ `set args`: Specify he arguments to be used the next time your program is run.
+ `show args`: Show the arguments to give your program when it is started.

## 4.4 Debugging an Already-running Process

To use `attach`, your program must be running in an environment which
supports processes. When you have finished debugging the attached
process, you can use the `detach` command to release it from GDB control.

## 4.5 Killing the Child Process

`kill` command is useful if you wish to debug a core dump instead of
a running process. GDB ignores any core dump file while your program
is running.

## 4.6 Debugging Multiple Inferiors Connections and Programs

GDB lets you run and debug multiple programs in a single session.
GDB represents the state of each program execution with an object
called an *inferior*. An inferior typically corresponds to a process,
but is more general and applies also to targets that do not have processes.
Inferiors may be created before a process runs, and may be retained after
a process exits.

To find out what inferiors exists any moment, use `info inferiors`. To get
information about the current inferior, use `inferior`. To find out what
open target connections exist at any moment, use `info connections`.

You can get multiple executables into a debugging session via the `add-inferior`
and `cline-inferior` commands. And use `remove-inferiors infno` to remove the
inferior or inferiors infno. It is not possible to remove an inferior that is
running with this command. For those, use the `kill` or `detach` command first.

## 4.7 Debugging Programs With Multiple Threads

GDB provides these facilities for debugging multi-threaded programs:

+ automatic notification of new threads.
+ `thread thread-id`, a command to switch among threads.
+ `info threads`, a command to inquire about existing threads.
+ `thread apply [thread-id-list | all] args`, a command to apply a command
to a list of threads.
+ thread-specific breakpoints.

For debugging purposes, GDB associates its own thread number, always a single integer
with each thread of an inferior. This number is unique between all threads of an
inferior, but not unique between threads of different inferiors.

You can refer to a given thread in an inferior using the qualified *inferior-num.thread-num*
syntax, also known as *qualified thread ID*.
