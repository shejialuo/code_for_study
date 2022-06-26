# Chapter 5

Inside GDB, your program may stop for any of several reasons, such as signals, a
breakpoint, or reaching a new line after a GDB command such as `step`.

## 5.1 Breakpoints, Watchpoints, and Catchpoints

A *breakpoint* makes your program stop whenever a certain point in the program
is reached. For each breakpoint, you can add conditions to control in finer
detail whether your program stops.

A *watchpoint* is a special breakpoint that stops your program when the value of
an expression changes. The expression may be a value of a variable, or it could
involve values of one or more variables combined by operators, such as `a + b`.
This is sometimes called *data breakpoints*. You must use a different command
to set watchpoints.

A *catchpoint* is another special breakpoint that stops your program when a certain
kind of event occurs, such as the throwing of a C++ exception or the loading of
a library. You use a different command to set a catchpoint.

GDB assigns a number to each breakpoint, watchpoint, or catchpoint when you create
it. these numbers are successive integers starting with one.

### 5.1.1 Setting Breakpoints

It is possible that a breakpoint corresponds to several locations in your program.
Examples of this situation are:

+ Multiple functions in the program may have the same value.
+ A C++ constructor.
+ A C++ template function.
+ An inlined function

In all those cases, GDB will insert a breakpoint at all the relevant locations.

You cannot delete the individual locations from a breakpoint. However, each
location can be invidiously enabled or disabled.

### 5.1.2 Setting Watchpoints

You can use a watchpoint to stop execution whenever the value of an expression
changes, without having to predicate a particular place where this may happen.

GDB does software watchponting by single-stepping your program and testing
the variable's value each time, which is hundreds of times slower than normal
execution.

On some systems, GDB includes support for hardware watchpoints, which do
not slow down the running of your program.

GDB automatically deletes watchpoints that watch local variables, or
expressions that involve such variables, when they go out of scope.

### 5.1.3 Setting Catchpoints

You can use *catchpoints* to cause the debugger to stop for certain kinds
of program events, such as C++ exceptions or the loading of a shared library.
Use the `catch` command to set a catchpoint.

### 5.1.4 Deleting Breakpoints

Withe the `clear` command you can delete breakpoints according to where they
are in your program. With the `delete` command you can delete individual
breakpoints, watchpoints, or catchpoints by specifying their breakpoint numbers.

### 5.1.5 Disabling Breakpoints

You disable and enable breakpoints, watchpoints, and catchpoints with the `enable`
and `disable` commands, optionally specifying one or more breakpoint numbers as arguments.

### 5.1.6 Dynamic Printf

The dynamic printf command `dprintf` combines a breakpoint with
formatted printing of your program's data to give you the effect of
inserting `printf` calls into your program on the fly, without having to
recompile it.

### 5.1.7 How to save breakpoints to a file

To save breakpoint definitions to a file use the `save breakpoints` command.

## 5.2 Continuing

Continuing means resuming program execution until your program
completes normally. In contrast, *stepping* means executing just one
more "step" of your program, where "step" may mean either one line of
source code, or one machine instruction.

## 5.3 Skipping Over Functions and Files

The program you are debugging may contain some functions which are
uninteresting to debug. The `skip` command lets you tell GDB to skip a function,
all functions in a file or a particular function in a particular file
when stepping.

## 5.4 Signals

GDB has the ability to detect any occurrence of a signal in your program.
You can tell GDB in advance what to do for each kind of signal.
