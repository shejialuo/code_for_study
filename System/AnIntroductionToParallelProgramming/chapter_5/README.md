# 5. Shared-memory programming with OpenMP

OpenMP is designed for systems in which each thread or process
can potentially have access to all available memory, and when we're
programming with OpenMP, we view our system as a collection of
autonomous cores or CPUs, all of which have access to main memory.

## 5.1 Getting started

OpenMP provides what's known as a "directive-based" shared-memory
API. In C and C++, this means that there are special preprocessor
instructions known as `pragma`s.

[openMP_hello_world.c](./openMP_hello_world.c)

### 5.1.1 Compiling and running OpenMP programs

```sh
gcc −g −Wall −fopenmp −o
omp_hello
omp_hello . c
. / omp_hello 4
```

### 5.1.2 The program

OpenMP `pragma`s always begin with `# prgama omp`. Our first directive
is a `parallel` directive, it specifies that the *structured block*
of code that follows should be executed by multiple threads.
A structured block is a C statement or a compound C statement
with one point of entry and one point of exit.

We'll usually specify the number of threads on the command line,
so we'll modify our `parallel` directives with the `num_threads` clause.
A *clause* in OpenMP is just some text that modifies a directive.
The `num_threads` clause can be added to a `parallel` directive.
It allows the programmer to specify the number of threads that
should execute the block.

What actually happens when the program gets to the `parallel` directive?
Prior to the `parallel` directive, the program is using a single
thread, the process started when the program start execution.
WHen the program reaches the `parallel` directive, the original
thread continues executing and `thread_count` are started.
In OpenMP parlance, the collection of threads executing the
`parallel` block is called a *team*. OpenMP thread terminology
includes the following:

+ *master*: the first thread of execution, or thread 0.
+ *parent*: thread that encountered a `parallel` directive and
started a team of threads. In many cases, the parent is also the master thread.
+ *child*: each thread started by the parent is considered a *child* thread.

When the block of code is completed, when the threads return from
the call to `Hello`, there is an *implicit barrier*.

## 5.2 The trapezoidal rule

### 5.2.1 A first OpenMP version

[first_version_trapezoidal_rule.c](./first_version_trapezoidal_rule.c)

## 5.3 Scope of variables

In OpenMp, the *scope* of a variable refers to the set of threads
that can access the variable in a `parallel` block. A variable that
can be accessed by all the threads in the team has *shared* scope,
while a variable that can only be accessed by a single thread
has a *private* scope.

## 5.4 The reduction clause

A *reduction operator* is an associative binary operation, and a
*reduction* is a computation that repeatedly applies the same reduction
operator to a sequence of operands to get a single result. Furthermore,
all of the intermediate results of the operation should be stored
in the same variable: the *reduction variable*.

In OpenMP it may be possible to specify that the result of a reduction
is a reduction variable. To do this, a `reduction` clause can be
added to a `parallel` directive.

```c
global_result = 0.0;
# pragma omp parallel num_threads(thread_count) reduction(+: global_result)
global_result += Local_trap(double a, double b , int n);
```

The syntax of the `reduction` clause is:

```c
reduction(<operator>: <variable list>)
```

## 5.5 The parallel for directive

As an alternative to our explicit parallelization of the trapezoidal rule,
OpenMP provides the `parallel for` directive. Using it, we can
parallelize the serial trapezoidal rule by simply placing a
directive immediately before the `for` loop:

```c
h = (b - a) / n;
approx = (f(a) + f(b)) / 2.0;
# pragma omp parallel for num_threads(thread_count) reduction(+: approx)
for(i = 1; i <= n - 1; i++)
  approx += f(a + i *h);
approx = h * approx;
```

Like the `parallel` directive, the `parallel for` directive forks
a team of threads to execute the following structured block. However,
the structured block following the `parallel for` directive must be
a `for` loop. Furthermore, with the `parallel for` directive
the system parallelizes the `for` loop by dividing the iterations of
the loop among the threads.

In a `for` loop that has been parallelized with a `parallel for`
directive, the default partitioning of the iterations among the
threads is up to the system. However, most system use roughly a
block partitioning.

### 5.5.1 Caveats

There are several caveats associated with the use of the `parallel for` directive.
First, OpenMP will only parallelize `for` loops. However, OpenMP
will only parallelize `for` loops for which the number of iterations
can be determined.

### 5.5.2 Data dependencies

Consider the following code, which computes the first $n$ Fibonacci numbers:

```c
fibo[0] = fibo[1] = 1;
# pragam omp parallel for num_threads(thread_count)
for(i = 2; i < m; i++)
  fibo[i] = fibo[i - 1] + fibo[i - 2]
```

The dependence of the computation of `fibo[n]` on the computation of `fibo[n - 1]`
is called a *data dependence*, also sometimes called a *loop-carried dependence*.

### 5.5.3 Estimating pi

One way to get a numerical approximation to $\pi$ is to use
many terms in the formula

$$
\pi = 4\sum_{k = 0} ^ {\infty} \frac{(-1)^{k}}{2k+1}
$$

We can implement this formula in serial code with

```c
double factor = 1.0;
double sum = 0.0;
for (k = 0; k < n; k++) {
  sum += factor/ (2 * k + 1);
  factor = -factor;
}
pi_approx = 4.0∗ sum;
```

However, it's pretty clear that the update to `factor` is an
instance of a loop-carried dependence. So we could change the code
to eliminate the loop dependence.

```c
factor = (k % 2 == 0) ? 1.0 : -1.0;
sum += factor/(2*k + 1);
```

However, we still need to ensure that each thread has its own
copy of `factor`. That is, to make our code correct, we need to
also ensure that `factor` has private scope. We can do this by adding
a `private` clause to the `parallel for` directive.

```c
double sum = 0.0;
# pragma omp parallel for num_threads(thread_count) \
    reduction(+:sum) private(factor)
  for (k = 0; k < n; k++) {
    k % 2 == 0 ? factor = 1.0 : factor = -1.0;
    sum += factor/ (2*k+1);
  }
```

The `private` clause specifies that for each variable listed inside
the parentheses, a private copy is to be created for each thread.

### 5.5.4 More on scope

Our problem with the variable `factor` is common one. We usually need
to think about the scope of each variable in a `parallel` block
or a `parallel for` block. Therefore, rather than letting OpenMP
decide on the scope of each variable, it's a very good practice for
us a s programmers to specify the scope of each variable in a block.
In fact, OpenMP provides a clause that will explicitly requires
us to do this: the `default` clause.

```c
# pragma omp parallel for num_threads(thread_count) \
  default(none) reduction(+:sum) private(k, factor) \
  shared(n)
```

## 5.6 Scheduling loops

In OpenMP, assigning iterations to threads is called *scheduling*,
and the `schedule` clause can be used to assign iterations in either
a `parallel for` or a `for` directive.

In general, the `schedule` clause has the form:

```c
schedule(<type> [, <chunksize>])
```

The `type` can be any one of the following:

+ `static`. The iterations can be assigned to the threads
before the loop is executed.
+ `dynamic` or `guided`. The iterations are assigned to the threads
while the loop is executing, so after a thread completes its
current set of iterations, it can request more than the run-time system.
+ `auto`. The compiler and/or the run-time system determine the schedule.
+ `runtime`. The schedule is determined at run-time based on an environment variable.

The `chunksize` is a positive integer. In OpenMP parlance, a *chunk*
of iterations is a block of iterations that would be executed consecutively
in the serial loop. The number of iterations in the block is the
`chunksize`. Only `static`, `dynamic`, and `guided` schedules can
have a `chunksize`.

## 5.7 Tasking

Tasking allows developers to specify independent units of computation
with the `task` directive;

```c
#pragma omp task
```

When a thread reaches a block of code with this directive, a new
task is generated by the OpenMP run-time that will be scheduled for
execution.
