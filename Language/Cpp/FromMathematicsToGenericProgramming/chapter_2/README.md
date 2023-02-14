# Chapter 2 The First Algorithm

Let's begin by looking at the fast multiplication algorithm, which
is still an important computational technique today.

## 2.1 Egyptian Multiplication

We define multiplication by $1$ like this:

$$
la = a
$$

Next we have the case where we want to compute a product of one more
thing than we already computed:

$$
(n + 1)a = na + a
$$

One way to multiply $n$ by $a$ is to add instances of $a$ together $n$
times. However, this could be extremely tedious for large numbers,
since $n - 1$ additions are required.

[multiply0.cpp](./multiply0.cpp)

The algorithm described by Ahmes which many modern authors refer to as
the "Russian Peasant Algorithm" relies on the following insight.

$$
\begin{align*}
4a &= ((a + a) + a) + a \\
   &= (a + a) + (a + a)
\end{align*}
$$

It allows us to compute $a + a$ only once and reduce the number of additions.

The idea is to keep halving $n$ and doubling $a$, constructing a sum of
power-of-2 multiples.

[multiply1.cpp](./multiply1.cpp)

## 2.2 Improving the Algorithm

Our `multiply1` function works well, but it also does recursive calls. Since
function calls are expensive, we want to transform the program to avoid this
expense.

One principle we're going to take advantage of is this: *It is often easier to*
*do more work than less*. Specifically, we are going to compute

$$
r + na
$$

[mult_acc0.cpp](./mult_acc0.cpp)

We can improve this further by simplifying the recursion. Notice that two
recursive differ only in their first argument. Instead of having two
recursive calls for the odd and even cases, we'll just modify the value of
$r$ before we recurse, like this:

[mult_acc1.cpp](./mult_acc1.cpp)

Now our function is *tail-recursive*. We'll take advantage of this fact shortly.
We make two observations:

+ $n$ is rarely 1.
+ If $n$ is even, there's no point checking to see if it's 1.

[mult_acc2.cpp](./mult_acc2.cpp)

*Definition 2.1* A strictly tail-recursive procedure is one in which all the
tail-recursive calls are done with the formal parameters of the procedure being
the corresponding arguments.

[mult_acc3.cpp](./mult_acc3.cpp)

Now it is easy to convert this to an iterative program by replacing the tail
recursion with a `while(true)` construct:

[mult_acc4.cpp](./mult_acc4.cpp)
