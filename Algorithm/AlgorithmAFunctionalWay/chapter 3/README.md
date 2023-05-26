# Chapter 3 The efficiency of functional programs

## 3.1 Reduction order

### 3.1.1 Examples

Functional programs are evaluated by reducing expressions to values. A *reduction strategy*
consists of choosing the next expression to reduce, subject to the constraint that an expression
must only be evaluated after all the sub-expressions on which it depends have already been
evaluated.

An important property of functional programs states that any reduction strategy, providing that it
terminates, produces the same result called the *normal form*.

[normal form](./NormalForm.hs)

We consider another example:

```hs
cancel x y = x
f x = f (x + 1)
```

The expression `cancel 2 (f 3)` can be evaluated in at least two ways (strict or lazy).

```txt
cancel 2 (f 3) => cancel 2 (f 3)       cancel 2 (f 3) = 2
               => cancel 2 (f 4)
               => ...
```

### 3.1.2 Controlling evaluation order in Haskell

The default evaluation strategy in Haskell is lazy evaluation. However, it is possible to
evaluate a given expression strictly use predefined operator `$!` which expects
two arguments `f` and `x`. This operator forces the evaluation of `x` before applying the
function `f` to it.

```hs
double $! (5 * 4)
```

### 3.1.3 Lazy data structures

In a lazy language, algebraic type constructors are also lazy. For example, consider the
following declaration of a list:

```hs
data List a = Cons a (List a) | Nil

map f Nil = Nil
map f (Cons x xs) = Cons (f x) (map f xs)

head (Cons x xs) = x
```

The following sequence shows an example of evaluating an application of the `head` function
in a lazy manner:

```txt
head (map double (Cons 1 (Cons 2 (Cons 3 Nil))))
  => head (Cons (double 1) (map double (Cons 2 (Cons 3 Nil))))
  => double 1
  => add 1 1
  => 2
```

However, it is also possible to define a strict constructor by preceding the relevant component
with the symbol `!`.

```hs
data List a = Cons !a (List a) | Nil
```

Applied to the previous example, the evaluation sequence is:

```txt
head (map double (Cons 1 (Cons 2 (Cons 3 Nil))))
  => head (Cons (double 1) (map double (Cons 2 (Cons 3 Nil))))
  => head (Cons (add 1 1) (map double (Cons 2 (Cons 3 Nil))))
  => head (Cons 2 (map double (Cons 2 (Cons 3 Nil))))
  => 2
```

It is also possible to define a list structure which forces the tail of the list to be evaluated:

```hs
data List a = Cons a !(List a) | Nil
```

```txt
head (map double (Cons 1 (Cons 2 (Cons 3 Nil))))
  => head (Cons (double 1) (map double (Cons 2 (Cons 3 Nil))))
  => ...
  => head (Cons (double 1) (Cons (double 2) (Cons (double 3) Nil)))
  => double 1
  => add 1 1
  => 2
```

## 3.2 Analyzing the efficiency of programs

### 3.2.1 Graph reduction

Graph reduction is based on a special data structure called the *heap*, which contains all the
objects manipulated by the program. These objects can be constants such as integers and lists, or
unevaluated *closures* which represent a function applied to some arguments. An object is referenced
through a *pointer*, which is representative of its location in the heap. For example, the heap
containing the expression `double (5 * 4)` is shown below.

![Example of reducing an expression](https://s2.loli.net/2022/12/06/VC3tqhTdJIoOKyu.png)

The heap contains two closures which correspond to applications of the functions `double` and `(*)`.
Each closure has pointers to the function name and its arguments, which can be constants or other
closures.

### 3.2.2 Time efficiency analysis

We need to find a quantitative measure that will be representative of the 'time'. Most approaches
simply define it as *the number of function applications* required to compute a given expression.

### 3.2.3 Step-counting analysis

We will use the number of function applications (or steps) as a measure. The analysis proceeds in
three successive phases:

1. In the first phase, for each function $f$ we derive a *step-counting* version $T_{f}$. By
definition, the number of calls required to compute $f$ applied to some arguments under a strict
regime is equal to $T_{f}$ applied to the same arguments.
2. The second phase consists of finding for recursive functions the structural property that
complexity depends on. This is called the *size*.
3. A closed expression expressed in terms of the size of the inputs is derived from the corresponding
step-counting version.

#### Transformation rules

Each expression $e$ in the program has a cost debited $T(e)$ and each function $f$ has a step-counting
version denoted $T_{f}$. Then

$$
\{f \ a_{1}a_{2} \dots a_{n} = e\} \rightarrow {T_{f} \ a_{1}a_{2} \dots a_{n} = 1 + T(e)}
$$

The cost of computing a constant $c$, or a variable $v$, is assumed to be 0. The cost of evaluating
a conditional expression is the cost of evaluating the predicate plus the cost of evaluating either
the 'then' or 'else' clause depending on the value of the predicate.

$$
\begin{align*}
T(c) &\rightarrow 0 \\
T(v) &\rightarrow 0 \\
T (\text{if } a \text{ then } b \text{ else } c) &\rightarrow T(a) +
(\text{if } a \text{ then } T(b) \text{ else } T(c))
\end{align*}
$$

### 3.2.4 Space efficiency analysis

### 3.2.5 Space leaks

## 3.3 Program transformation
