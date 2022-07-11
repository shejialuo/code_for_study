# 4. Shared-memory programming with Pthreads

In this chapter we'll be using POSIX threads for most of our shared-memory
functions.

## 4.1 Matrix-vector multiplication

Let's take a look at writing a Pthreads matrix-vector multiplication program.
Recall that if $A = (a_{ij})$ is an $m \times n$ matrix and
$\mathbf{x} = (x_{0}, x_{1}, \dots, x_{n - 1}) ^ {T}$ is an $n$-dimensional
column vector, then the matrix-vector product $A\mathbf{x} = \mathbf{y}$
is an $m$-dimensional column vector, $\mathbf{y} = (y_{0}, y_{1}, \dots, y_{m - 1})^{T}$,
in which the $i$th component $y_{i}$ is obtained by finding the dot
product of the $i$th row of $A$ with $\mathbf{x}$:

$$
y_{i} = \sum_{j = 0} ^ {n - 1} a_{ij}x_{j}
$$

Thus pseudocode for a *serial* program for matrix-vector multiplication
might look like this:

```c
for(i = 0; i < m; i++) {
  y[i] = 0.0;
  for(j = 0; j < n; j++) {
    j[i] + A[i][j] * x[j];
  }
}
```

we want to parallelize this by dividing the work among the threads.
One possibility is to divide the iterations of the outer loop among
the threads. If we do this, each thread will compute some of the
components of $y$. For example, suppose that $m = n = 6$ and the number
of threads, `thread_count` or $t$, is three. Then the computation could
be divided among the threads as follows:

| **Thread** | **Components of y** |
|:----------:|:-------------------:|
|      0     |      y[0], y[1]     |
|      1     |      y[2], y[3]     |
|      2     |      y[4], y[5]     |

The thread will need to access every element of row `i` of `A` and
every element of `x`. We see that each thread needs to access every
component of `x`, which each thread only needs to access its assigned
rows of `A` and assigned components of `x`, while each thread only
needs to access its assigned rows of `A` and assigned components of
`y`. This suggests, at a minimum, `x` should be shared. Let's also
make `A` and `y` shared.

Having made these decisions, we only need to write the code that
each thread will use for deciding which components of `y` it will
compute. To simplify the code, let's assume that both $m$ and $n$ are
evenly divisible by $t$.

So we can get

$$
\begin{align*}
first component&: q \times \frac{m}{t} \\
last component&: (q + 1) \times \frac{m}{t} - 1
\end{align*}
$$

```c
void *Pth_mat_vect(void* rank) {
  long my_rank = (long) rank;
  int i, j;
  int local_m = m / thread_count;
  int my_first_row = my_rank * local_m;
  int my_last_row = (my_rank + 1) * local_m - 1;

  for(i = my_first_row; i <= my_last_row; i++) {
    y[i] = 0.0;
    for(j = 0; j  < n; j++)
      y[i] += A[i][j] * x[j];
  }
  return NULL;
}
```

## 4.2 Critical sections

Let's try to estimate the value of $\pi$. There are lots of different
formulas we could uses. One of the simplest is

$$
\pi = 4\left(1 - \frac{1}{3} + \frac{1}{5} + \cdots + (-1)^n \frac{1}{2n + 1} + \cdots \right)
$$

The following *serial* code uses this formula:

```c
double factor = 1.0;
double sum = 0.0;
for(i = 0; i < n; i++, factor = - factor) {
  sum += factor(2 * i  + 1);
}
double pi = 4.0 * sum;
```

We can try to parallelize this in the same way we parallelized the
matrix-vector multiplication program: divide up the iterators in the `for` loop
among the threads and make `sum` a shared variable. To simplify the computations,
let's assume that the number of threads, `thread_count` or $t$,
evenly divides the number of terms in the sum, $n$. Then if $\bar{n} = n / t$,
thread 0 can add the first $\bar{n}$ terms.

Well, Here come the critical sections. I have already
learned a lot
about the thread, so I omit a lot of detail here.
