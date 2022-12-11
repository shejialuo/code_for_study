# EXE: Automatically Generating Inputs of Death

## Introduction

Instead of running code on manually or randomly constructed input, EXE
(Execution generated Executions) runs it on *symbolic* input that is
initially allowed to be "anything". As checked code runs, if it tries
to operate on symbolic expressions, EXE replaces the operation with
its corresponding input-constraint; it runs all other operations as usual.
When code conditionally checks a symbolic expression, EXE forks execution,
constraining the expression to be true on the true branch and false on the
other. When a path terminates or hits a bug, EXE automatically generates
a test case that will run this path by solving the path's constraints for
concrete values using its co-designed constraint solver, STP.

## EXE Overview

We illustrate EXE's main features by walking the reader through the simple code
example.

```c
#include <assert.h>
int main() {
  unsigned i, t, a[4] = {1, 3, 5, 2};
  make_symbolic(&i);
  if (i >= 4)
    exit(0);
  char *p = (char*)a + i * 4;
  *p = *p - 1;

  t = a[*p];
  t = t/ a[i];

  if (t == 2)
    assert(i == 1);
  else
    assert(i == 3);
}
```

To check their code with EXE, programmers only need to mark which memory locations
should be treated as holding *symbolic data* whose values are initially entirely
unconstrained. In the example, the call `make_symbolic(&i)` marks `i` as symbolic.
They then compiled their code using the EXE compiler, `exe-cc`.

As the program runs, EXE executes each feasible path, tracking all constraints. When
a program path terminates, EXE calls STP to solve the path's constraints for concrete
values. A path terminates when

+ it calls `exit()`
+ it crashes
+ an assertion fails
+ EXE detects an error

Constraint solution are literally the concrete bit values for an input that will cause
the given path to execute. When generated in response to an error, they provide a concrete
attack that can be launched against the tested system.

The EXE compiler has three main jobs. First it inserts checks around every assignment,
expression, and branch in the tested program to determine if its operands are concrete
or symbolic. An operand is defined to be concrete if and only if all its constituent bits
are concrete. If all operands are concrete, the operation is executed just as in the
instrumented program. If any operand is symbolic, the operation is not performed, but
instead passed to the EXE runtime system, which adds it as a constraint for the current
path. For the example' expression `p = (char*)a + i * 4`, EXE checks if the operands `a` and
`i` on the right hand side of the assignment are concrete. If so, it executes the expression,
assigning the result to `p`. However, since `i` is symbolic, EXE instead adds the constraint that
*p* equals *(char\*a) + i \* 4*. Note that because `i` can be one of four values, `p` simultaneously
refers to four different locations.

Second `exe-cc` inserts code to fork a program execution when it reaches a symbolic branch point, so
that it can explore each possibility. Consider the `if (i>=4)`. Since `i` is symbolic, so is this
expression. Thus, EXE forks execution and on the true path asserts that `i>=4` is true, and on the
false path that it is not. Each time it adds a branch constraint, EXE queries STP to check that there
exists at least one solution for the current path's constraints. If not, the path is impossible and
EXE stops executing it.

Third, `exe-cc` inserts code that calls to check if a symbolic expression could have any possible
value that could cause either a null or out-of-bounds memory reference or a division or modulo by
zero.
