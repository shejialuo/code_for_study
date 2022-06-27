# Chapter 6

When you are debugging a program, it is not unusual to realize that you
have gone too far, and some event of interest has already happened.
If the target environment supports it, GDB can allow you to "rewind" the
program by running it backward.

A target environment that supports reverse execution should be able
to "undo" the changes in machine state that have taken place as
the program was executing normally. Variables, registers should
revert to their previous values.
