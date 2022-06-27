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
