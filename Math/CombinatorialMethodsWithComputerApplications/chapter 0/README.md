# Chapter 0 Introduction to Combinatorics

*Combinatorial mathematics*, or more briefly, *combinatorics*, refers to the body
of mathematics developed for solving problems concerned with *discrete sets*.

Most combinatorics problems have one of three fundamental objectives: counting or
calculating a sum, constructing a configuration involving two or more discrete sets
and optimization.

## 0.1 Objectives of Combinatorics

### Combinatorial Enumeration

*Combinatorial enumeration* is concerned with the theory and methods of discrete
measurement. Summing the values of a function over a finite or countable set is
the prototypical discrete measurement. The word *counting* is frequently used as
a synonym for combinatorial enumeration.

### Incidence Structures

An *incidence structure* is a combinatorial configuration that involves two or
more discrete sets.

### Optimization

In the present context, we mean by *combinatorial optimization* any discrete problem
concerned with finding a maximum or a minimum. In some other contexts, it has a
special meaning of finding the maximum value of a function on a region of a
Euclidean space or of functions that could possibly be so represented.

## 0.2 Ordering and Selection

We begin with the analysis and solution of a sample counting problem involving ordering
and selection.

### A Counting Problem

DEFINITION: An **ordering of a set** $S$ of $n$ objects is bijection from the set
$\{1,2,\dots,n\}$ to the set $S$. It serves as a formal model for an arrangement
for the $n$ objects into a row.

DEFINITION: An **(unordered) selection** from a set $S$ is a subset of $S$.

We give an example here.

#### Example 0.1

In how many ways it possible to arrange two of the letters
$A,B,C,D,E$ and two of the digits $0,1,2,3$ into a row of four characters,
such that no two digits are adjacent.

#### Solution 0.1

It is easy to determine that there are 10 possible selections
of two of the five letters and 6 possible selections of two of the four digits.
Thus there are 60 possible selections of a combination of four symbols that
meets the given requirement.

An arrangement of four such symbols into a row meets the requirement if it has
any of the three forms $LDLD,DLDL$ and $DLLD$.

Thus, the answer is $4 \times 3\times 60 = 720$.

### Sequences and Generating Functions

A some what more general version of Example 0.1 supposes that $x_{n}$ is the
number of ways to form an arrangement of four symbols when there are $n$
letters, but still only four digits. We have just calculated that $x_{5} = 720$.
Similar analysis yields the values
$$
x_{0} = 0 \ x_{1} = 0 \ x_{2} = 72 \ x_{3} = 216 \ x_{4} = 432 \ x_{5} = 720 \ \dots
$$

The sequence over all non-negative integers $n$ is called a *counting sequence*
for this problem.

We could make it into a *generating function*:

$$
0 + 0z + 72z^{2} + 216z^{3} + 432z^{4} + 720z^{5}
$$

### Recurrences

A sequence can be specified by giving some of its initial values and a
*recurrence* that says how each later entry can be calculated from
earlier entries.

## 0.3 Some Rules For Counting

### Rules of Sum and Product

DEFINITION: **Rules of Sum**: let $U$ and $V$ be disjoint sets. Then

$$
|U \cup V| = |U| + |V|
$$

DEFINITION: **Rules of Product**: let $U$ and $V$ be sets. Then

$$
|U \times V| = |U| \cdot |V|
$$

### Rule of Quotient

DEFINITION: A **partition** of a set $U$ is a collection of mutually exclusive
subsets $U_{1},\dots, U_{p}$ called **cells of the partition**, whose union is
$U$.

DEFINITION: **Rule of Quotient**: Let $\mathcal{P}$ be a partition of a set $U$
into cells, each of the same cardinality $k$. Then the number of cells equals
the quotient

$$
\frac{|U|}{k}
$$

### When to Subtract

There are some common circumstances when calculating the cardinality if a set is
achieved using a subtraction operation. One is when set $X$ to be counted is a
subset of a largest set $U$ and it looks easier to calculate the size of $U$ and
of the complement $U-X$ than the size of the set $X$ directly.

Another circumstances where subtraction is used is in calculating the size of a union
of overlapping subsets.

### Pigeonhole Principle

DEFINITION: **Pigeonhole Principle**: Let $f: U \to V$ be a function with finite
domain and finite codomain. Let any two of the following three conditions hold:

+ $f$ is one to one.
+ $f$ is onto.
+ $|U| = |V|$.

Then the third condition also holds.

A generalized version of the Pigeonhole Principle asserts that when there are $p$
pigeons and $h$ pigeonholes, there is a pigeonhole with at least
$$
\left\lceil\frac{p}{h}\right\rceil
$$
pigeons.

### Empty Sums and Empty Products

DEFINITION: A sum over an empty set of numbers is called an **empty sum**. Its
value is taken to be 0, the additive identity of the number system.

DEFINITION: A product over an empty set of numbers is called an **empty product**.
Its value is taken to be 1, the multiplicative identity of the number system.

### Multisets

Informally, a multiset is often described as a "set in which the same element may
occur more than once".

DEFINITION: A **multiset** is a pair $(S,\iota)$ in which $S$ is a set and
$\iota: S \to \Z^{+}$ is a function that assigns to each element $s \in S$ a number
$\iota(s)$ called its **multiplicity**.

The Rule of Quotient implies that the number of ways to arrange the element of a
finite multiset $(S,\iota)$ is

$$
\frac{(\sum_{s \in S} \iota(s))!}{\prod_{s\in S}(\iota(s)!)}
$$

DEFINITION: The **cardinality of a multiset** $(S, \iota)$ is taken to be the sum
$$
\sum_{s \in S} \iota(s)
$$
of the multiplicities of its elements. It is denoted $|(S, \iota)|$.

## 0.4 Counting Selections

This section gives models for several different kinds of selection from a set $S$
and methods for counting the number of possible selections.

### Ordered Selections

DEFINITION: An **ordered selection** of $k$ objects from a set of $n$ objects is
a function from the set
$$
\{1,2,\dots,k\}
$$
to the set $S$. It serves as a formal model for an arrangement of $k$ objects from
$S$ into a row, or of a repetition-free list of length $k$ of objects from $S$.

**Proposition 0.4.1.** Let $P(n,k)$ be the number of possible ordered selections
of $k$ objects from a set $S$ of $n$ objects. Then

$$
P(n,k) = A_{n}^{k}
$$

### Unordered Selections

**Proposition 0.4.2.** The number of unordered selections of $k$ objects from a set
$S$ of $n$ objects is given by the rule

$$
C_{n}^{k} = A_{n}^{k} / k!
$$

### Selections with Repetitions Allowed

DEFINITION: An **ordered selection with unlimited repetition** of $k$ objects
from a set $S$ of size $n$ is a finite sequence
$$
x_{1}, x_{2}, \dots, x_{n}
$$
of $k$ objects, each of which is an element of $S$.

**Proposition 0.4.3.** The number of ordered selections of $k$ objects from a
set $S$ of $n$ objects is $n^{k}$.

## 0.5 Permutations

DEFINITION: A **permutation** of a set $S$ is a bijection (a one-to-one, onto
function) from $S$ to itself.

### 2-Line Repetition of Permutations

DEFINITION: The **2-line representation of a permutation** $\pi$ of a set $S$ is
a 2-line array that lists the objects of $S$ in its top row.

DEFINITION: The **inverse of a permutation** $\pi$ on a set $S$ is the permutation
$\pi^{-1}$ that restores each object of $S$ to its position before the application
of $\pi$.

### Composition of Permutations

DEFINITION: The **composition of permutations** $\pi$ and $\tau$ is the permutation
$\pi \circ \tau$ resulting from first applying $\pi$ and the applying $\tau$. Thus,
$(\pi \circ \tau)(x) = \tau(\pi(x))$.

### Cyclic Permutations

A **cyclic permutation** is a permutation whose successive application would take
each object of the permuted set successively through the positions of all the
other objects.

DEFINITION: A permutation of the form

$$
\begin{pmatrix}
x & \pi(x) & \pi^{2}(x) & \cdots & \pi^{p -2} & \pi^{p -1}\\
 \pi(x) & \pi^{2}(x) & \cdots & \pi^{p -2} & \pi^{p -1} & x
\end{pmatrix}
$$

This is said to be **cyclic permutation** of *period* $p$.

### Disjoint Cycle Representation

A fundamental way of understanding a permutation $\pi$ of a finite set $S$
is in terms of the cyclic permutations it induces on various subsets of $S$.
Its structure is understood in terms of the lengths of these cycles of objects.

**Proposition 0.5.1.** Let $\pi$ be a permutation on a finite set $S$ and let
$x \in S$. The the sequence $x \pi(x) \pi^{2}(x) \pi^{3}(x) \dots$
eventually contains an entry $\pi^{j}(x)$ such that $\pi^{j}(x) = x$,
and the sequence is periodic with period $j$.

**Proof**: Since the set $S$ is finite, the sequence must eventually contain
some entry $\pi^{j}(x)$ that matches a previous entry. Suppose that $\pi^{i}(x)$
is the previous entry such that

$$
\pi^{j}(x) = \pi^{i}(x)
$$

Then

$$
\begin{align*}
  \pi^{j - i} &= \pi^{-i}(\pi^{j}(x)) \\
              &= \pi^{-i}(\pi^{i}(x)) \\
              &= \pi^{0}(x) = x
\end{align*}
$$

Choose an arbitrary element $x_{1} \in S$. Let $k_{1}$ be the smallest integer
such that $\pi^{k_{1}}(x_{1}) = x_{1}$. Let $T_{1}$ be the subset

$$
T_{1} = \{x_{1},\pi(x_{1}),\dots, \pi^{k_{1}- 1}(x_{1})\}
$$

Then the restriction $\pi|_{T_{1}}$ of the permutation $\pi$ to the
subset $T_{1}$ is the cyclic permutation

$$
\pi|_{T_{1}} = \left(x_{1},\pi(x_{1}),\dots, \pi^{k_{1}- 1}(x_{1})\right)
$$

**Proposition 0.5.2** Let $\pi$ be a permutation on a finite set $S$ and let $x \in S$. Let

$$
T = \{\pi^{i}(x) | i \in \N \}
$$

Let $y \in S - T$ and let

$$
T' = \{\pi^{j}(y) | j \in \N\}
$$

Then the subsets $T$ and $T'$ are disjoint.

DEFINITION: A **disjoint cycle representation** of a permutation $\pi$ on a set
$S$ is as a composition of cyclic permutations on subsets of $S$
that constitute a partition of $S$, one cyclic permutation for
each subset in the partition.

## 0.6 Graphs

One widely studied combinatorial structure is called a *graph*. Intuitively, a graph
is a configuration comprising a discrete set of points in space and
a discrete set of curves, each of which runs either between two points or from
a point back to the same point. Formally, it is based on two abstract sets.

DEFINITION: A graph $G = (V,E)$ is a mathematical structure consisting of
two finite sets $V$ and $E$, called *vertices* and *edges*. Each edge has
a set of one or two vertices associated to it, which are called its *endpoints*.
