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



