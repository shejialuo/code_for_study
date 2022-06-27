# Chapter 8

When your program has stopped, the first thing you need to know is
where it stopped and how it got there.

Each time your program performs a function call, information about the
call is generated. That information includes the location of the call in
your program, the arguments of the call, and the local variables of the
function being called. The information is saved in a block of data called
a *stack frame*.

## 8.1 Stack Frames

When your program is started, the stack has only one frame, that of the
function `main`. This is called the *initial* frame or the *outermost* frame.

## 8.2 Backtrace

A backtrace is a summary of how your program got where it is. It
shows one line per frame, for many frames, starting with the currently
executing frame, followed by its caller, and on up the stack.

## 8.3 Selecting a Frame

Most commands for examining the stack and other data in your program work
on whichever stack frame is selected at the moment.
