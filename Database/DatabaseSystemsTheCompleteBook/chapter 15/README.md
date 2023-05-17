# Chapter 15 Query Execution

The broad topic of query processing will be covered in this chapter. The
*query processor* is the group of components of a DBMS that turns user queries
and data-modification commands into a sequence of database operations and
executes those operations.

Below suggests the division of topics. In this chapter, we concentrate
on query execution, that is, the algorithms that manipulate the data of the
database. We focus on the operations of the extended relational algebra.

![The major parts of the query processor](https://s2.loli.net/2022/04/07/Lw3IzYiuSd4VbX8.png)

To set the context for query execution, we offer a very brief outline
of query compilation. Query compilation is divided into the three
major steps shown below.

![Outline of query compilation](https://s2.loli.net/2022/04/07/GORpsJE2dViwqDY.png)

1. *Parsing*. A *parse tree* for the query is constructed.
2. *Query Rewrite*. The parse tree is converted to an initial query plan,
which is usually an algebraic representation of the query. This initial
plan is then transformed into an equivalent plan that is expected to
require less time to execute.
3. *Physical Plan Generation*. The abstract query plan from 2, often歌词each of the operators of the logical
plan, and by selecting an order of execution for these operators.
The physical plan is represented by an expression tree.

Parts 2 and 3 are often called the *query optimizer*, and these are
teh hard parts of query compilation. To select the best query plan
we need to decide:

+ Which of the algebraically equivalent forms of a query leads to the
most efficient algorithm for answering the query?
+ For each operation of the selected form, what algorithm should we use
to implement that operation?
+ How should the operations pass data from one to the other, in a pipelined
fashion, in main-memory buffers, or via the disk?

Each of these choices depends on the metadata about the database. Typical
metadata that is available to the query optimizer includes: the size
of each relation; statistics such as the approximate number and frequency
of different values for an attribute; the existence of certain indexes;
and the layout of data on disk.

## 1. Introduction to Physical-Query-Plan Operators

Physical query plans are built from operators, each of which implements
one step of the plan. Often, the physical operators are particular implementations
for one of the operations of
relational algebra. However, we also need physical operators for other tasks
that do not involve an operation of relational algebra. For example,
we often need to "scan" a table, that is, bring into main memory each tuple
of some relation.

### 1.1 Scanning Tables

Perhaps the most basic thing we can do in a physical query plan is
to read the entire contents of a relation $R$. A variation of this
operator involves a simple predicate, where we read only those
tuples of the relation $R$ that satisfy the predicate. There are
two basic approaches to locating the tuples of a relation $R$.

+ In many cases, the relation $R$ is stored in an area of secondary memory,
with its tuples arranged in blocks. The blocks containing the tuples of
$R$ are known to the system, and it is possible to get the blocks
one by one. This operation is called *table-scan*.
+ If there is an index on any attribute of $R$, we may
be able to use this
index to get all the tuples of $R$. This operation is called *index-scan*.

### 1.2 Sorting While Scanning Tables

The physical-query-plan operator *sort-scan* takes a relation $R$ and a specification
of the attributes on which the sort is to be made, and produces $R$ in
that sorted order. There are several ways that sort-scan can be implemented.
If relation $R$ must be sorted by attribute $a$, and there is a B-tree index
on $a$, then a scan of the index allows us to produce $R$ in the desired
order. If $R$ is small enough to fit in main memory, then we can
retrieve its tuples using a table scan or index scan, and then
use a main-memory sorting algorithm. If $R$ is too large to fit in main memory,
then we can use a multiway merge-sort.

### 1.3 The Computation Model for Physical Operators

A query generally consists of several operations of relational algebra,
and the corresponding physical query plan is composed of several physical
Since choosing physical query plan wisely is an essential of a good query
processor, we must be able to estimate the "cost" of each operator we use.
We shall use the number of disk I/O's as our measure of cost for an operation.

When passing algorithms for the same operations, we shall make an assumption:

+ We assume that the arguments of any operator are found on disk, but the
result of the operator is left in main memory.

## 1.4 Parameters for Measuring Costs

We need a parameter to represent the portion of main memory that the operator
uses, and require other parameters to measure the size of its arguments. Assume
that the main memory is divided into buffers, whose size is the same as the
size of disk blocks. Then $M$ will denote the number of main-memory buffers
available to an execution of a particular operator.

Next, let us consider the parameters that measure the cost of accessing argument
relations. These parameters, measuring size and distribution of the data in a
relation, are often computed periodically to help the query optimizer choose
physical operators.

We shall make the simplifying assumption that data is accessed one block at
a time from disk. There are three parameter families, $B$, $T$ and $V$:

+ When describing the size of a relation $R$, we must often are concerned
with the number of blocks that are needed to hold all the tuples of $R$.
This number of blocks will be denoted $B(R)$, or just $B$. Usually, we assume
that $R$ is *clustered*; that is, it is stored in $B$ blocks or in approximately
$B$ blocks.
+ Sometimes we also need to know the number of tuples in $R$, and we denote
this quantity by $T(R)$.
+ Finally, we shall sometimes want to refer to the number of distinct values
that appear in a column of a relation. If $R$ is a relation, and one of its
attributes is $a$, then $V(R, a)$ is the number of distinct values of the
column for $a$ in $R$.

## 1.5 I/O Cost for Scan Operators

If relation $R$ is clustered, then the number of disk I/O's for the table-scan
operator is approximately $B$. Likewise, if $R$ fits in main-memory, then we can
implement sort-scan by reading $R$ into emory and performing an in-memory sort,
again requiring only $B$ disk I/O's.

However, if $R$ is not clustered, the the number of required disk I/O's is generally
much higher. If $R$ is distributed among tuples of other relations, then a table-scan
for $R$ may require reading as many blocks as there are tuples of $R$; that is, the
I/O cost is $T$. Similarly, if we want to sort $R$, but $R$ fits in memory, then $T$
disk I/O's are what we need to get all of $R$ into memory.

Generally, an index on a relation $R$ occupies many fewer than $B(R)$ blocks. Therefore,
a scan of the entire $R$, Therefore, a scan of the entire $R$, which takes at least
$B$ disk I/O will require significantly more I/O's than does examining the entire index.

## 1.6 Iterators for Implementation of Physical Operator

Many physical operators can be implemented as an *iterator*, which is a group of three
methods that allows a consumer of the result of the physical operator to get the
result one tuple at a time:

1. `Open()` starts the process of getting tuples, but does not get a tuple. It initializes
any data structures needed to perform the operation.
2. `GetNext()` returns the next tuple in the result and adjusts data structures as
necessary to allow subsequent tuples to be obtained.
3. `Close()` ends the iteration after all tuples have been obtained.

## 2. One-Pass Algorithms

There are many algorithms for operators have been proposed, they largely fall into
three classes:

+ Sorting-based methods
+ Hash-based methods.
+ Index-based methods.

In addition, we can divide algorithms for operators into three "degrees" of difficulty
and cost:

+ Some methods involve reading the data only once from disk. These are the *one-pass*
algorithms.
+ Some methods work for data that is too large to fit in available main memory but not
for the largest imaginable data sets. These *two-pass* algorithms are characterized by
reading data a first time from disk, processing it in some way, writing all, and then
reading it a second time for further processing during the second pass.
+ Some methods work without a limit on the size of the data. These methods use three
or more passes to do their jobs.

We shall classify operators into three broad groups:

1. *Tuple-at-a-time, unary operations*. These operations (selection and projection) do
not require an entire relation, or even a large part of it, in memory at once. Thus, we
can read a block at a time, use one main-memory buffer, and produce our output.
2. *Full-relation, unary operations*. These one-argument operations require seeing all
or most of the tuples in memory at once, so one-pass algorithms are limited to relations
that are approximately of size $M$ or less. The operations of this class are $\gamma$
(the group operator) and $\delta$ (the duplicate-elimination operator).
3. *Full-relation, binary operations*.

### 2.1 One-Pass Algorithms for Tuple-at-a-Time Operations

The tuple-at-a-time operations $\sigma(R)$ and $\pi(R)$ have obvious algorithms,
regardless of whether the relation fits in main memory. We read the blocks of $R$ one
at a time into an input buffer, perform the operation on each tuple, and move
the selected tuples or the projected tuples to the output buffer as below. And we
require only that $M \geq 1$ for the input buffer

![A selection or projection being performed on a relation R](https://s2.loli.net/2023/04/07/8sfJNZ5KTVoedXQ.png)

### 2.2 One-Pass Algorithms for Unary, Full-Relation Operations

#### Duplicate Elimination

To eliminate duplicates, we can read each block of $R$ one at a time, but for each
tuple we need to make a decision as to whether:

1. It is the first time we have seen this tuple where we copy it to the output.
2. We have seen the tuple before, we must not output this tuple.

We could use a hash table with a large number of buckets, or some form of balanced
binary search tree. Each of these structures ha some space overhead in addition to
the space needed to store the tuples; for instance, a main-memory hash table needs
a bucket array and space for pointers to link the tuples in a bucket. However, the
overhead tends to be small compared with the space needed to store the tuples, and
we shall in this chapter neglect this overhead.

On this assumption, we may store in the $M - 1$ available buffers of main memory
as many tuples as will fit in $M - 1$ blocks of $R$. If we want one copy of each
distinct tuple of $R$ to fit in main memory, then $B(\delta(R))$ must be no larger
than $M - 1$. Since we expect $M$ to be much larger than 1, a simpler approximation
to this rule, and the one we shall generally use, is:

+ $B(\delta(R)) \leq M$

However, we cannot in general compute the size of $\delta(R)$ without computing
$\delta(R)$ itself. We should underestimate the size.

#### Grouping

A grouping operation $\gamma_{L}$ gives us zero or more grouping attributes and
presumably one or more aggregated attributes. If we create in main memory one
entry for each group then we can scan the tuples of $R$, one block at a time.

+ For a `MIN(a)` or `MAX(a)` aggregate, record the minimum or maximum value,
respectively.
+ For any `COUNT` aggregation, add one for each tuple of the group that is seen.
+ For `SUM(a)`, add the value of attribute `a` to the accumulated sum for its
group, provided `a` is not `NULL`.
+ `AVG(a)` is the hard case. We must maintain two accumulations.

The number of disk I/O's needed for this one-pass algorithm is $B$, as must be
the case for any one-pass algorithm for a unary operator.

### 2.3 One-Pass Algorithms for Binary Operations

Bag union can be computed by a very simple one-pass algorithm. To compute $R \cup_{B} S$,
we copy each tuple of $R$ to the output and then copy every tuple of $S$. The number of
disk I/O's is $B(R) + B(S)$. $M = 1$ suffices regardless how large $R$ and $S$ are.

Other binary operations require reading the smaller of the operands $R$ and $S$ into
main memory and building a suitable data structure so tuples can be both inserted
quickly and found quickly. As before, a hash table or balanced tree suffices. Thus, the
approximate requirement for a binary operation on relations $R$ and $S$ to be performed
in one pass is:

+ $min(B(R), B(S)) \leq M$

#### Set Union

We read $S$ into $M - 1$ buffers of main memory and build a search structure whose
search key is the entire tuple. All these tuples are also copied to the output. We
then read each block of $R$ into the $M$th buffer, one at a time. For each tuple
$t$ of $R$, we see if $t$ is in $S$m and if not, we copy $t$ to the output. If $t$
is also in $S$, we skip $t$.

#### Set Intersection

Read $S$ into $M - 1$ buffers and build a search tree with full tuples as the search
key. Read each block of $R$, and for each tuple $t$ of $R$, see if $t$ is also in $S$.
If so, copy $t$ to the output, and if not, ignore $t$.

#### Set Difference

Since difference is not commutative, we must distinguish between $R −S$ and
$S − R$, continuing to assume that $R$ is the larger relation. In each case, read
$S$ into $M − 1$ buffers and build a search structure with full tuples as the search
key.

To compute $R - S$, we read each block of $R$ and examine each tuple $t$ on
that block. If $t$ is in $S$, then ignore $t$; if it is not in $S$ then copy $t$ to
the output.

To compute $S − R$, we again read the blocks of $R$ and examine each tuple
$t$ in turn. If $t$ is in $S$, then we delete $t$ from the copy of $S$ in main memory,
while if $t$ is not in $S$ we do nothing. After considering each tuple of $R$, we copy
to the output those tuples of $S$ that remain.

#### Bag Intersection

We read $S$ into $M - 1$ buffers, but we associate with each distinct
tuple a *count* which initially measures the number of times this
tuple occurs in $S$. Multiple copies of a tuple $t$ are not stored individually.
Rather we store one copy of $t$ and associate with it a count equal to
the number of times $t$ occurs.

This structure could take slightly more space than $B(S)$ blocks if
there were few duplicates. Thus, we shall continue to assume that
$B(S) \leq M$ is sufficient for a one-pass algorithm to work.

Next, we read each block of $R$, and for each tuple $t$ of $R$ we
see whether $t$ occurs in $S$. If not we ignore $t$. However, if
$t$ appears in $S$, and the count associated with $t$ is still positive,
then we output $t$ and decrement the count by 1 until it becomes 0.

#### Bag Difference

It's just like Set Difference but with a count.

#### Product

Read $S$ into $M - 1$ buffers of main memory. Then read each block
of $R$, and for each tuple $t$ of $R$ concatenate $t$ with each
tuple of $S$ in main memory.

## 3. Nested-Loop Joins

### 3.1 Tuple-Based Nested-Loop Join

The simplest variation of nested-loop join has loops that range over
individual tuples of the relations involved. In this algorithm,
which we call *tuple-based nested-loop join*, we compute the join
$R(X, Y) \bowtie S(Y,Z)$ as follows:

```pseudocode
FOR each tuple s in S DO
  FOR each tuple r in R DO
    IF R and S to join to make a tuple t THEN
      output t;
```

If we are careless about how we buffer the blocks of relations $R$
and $S$, then this algorithm could require as many as $T(R)T(S)$ disk I/O's.
However, there are many situations where this algorithm can be modified
to have much lower cost.

+ Use an index on the join attribute or attributes of $R$ to find
the tuples of $R$ that match a given tuple of $S$.
+ Look much more carefully at the way tuples of $R$ and $S$
are divided among block, and use as much of the memory as it can to
reduce the number of disk I/O's.

### 3.2 Block-Based Nested-Loop Join Algorithm

We can improve on the tuple-based nested-loop join if we compute
$R \bowtie S$ by:

1. Organizing access to both argument relations by blocks.
2. Using as much main memory as we can to store tuples belonging
to the relation $S$, the relation of the outer loop.

```pseudocode
FOR each chunk of M-1 blocks of S DO BEGIN
  read these blocks into main-memory buffers;
  organize their tuples into a search structure whose
    search key is the common attributes of R and S
  FOR each block b of R DO BEGIN
    read b into main memory;
    FOR each tuple t of b DO BEGIN
      find the tuples of S in main memory that join with t;
      output the join of t with ech of these tuples.
    END;
  END;
END;
```

### 3.3 Analysis of Nested-Loop Join

Assuming $S$ is the smaller relation, the number of chunks is
$B(S) / (M  - 1)$. At each iteration, we read $M - 1$ blocks of
$S$ and $B(R)$ blocks of $R$. The number of disk I/O' is thus
$B(S)(M - 1 + B(R)) / (M - 1)$, or $B(S) +(B(S)B(R)) / (M - 1)$.

Assuming all of $M$, $B(S)$, and $B(R)$ are large, but $M$ is the
smallest of these, an approximation to the above formula is $B(S)B(R) / M$.

Although nested-loop join is generally not the most efficient join algorithm
possible, we should note that in some early relational DBMS, it was
the only method available.

## 4. Two-Pass Algorithms Based on Sorting

We concentrate on *two-pass algorithms*, where data from the operand
relations is read into main memory, processed in some way, written
out to disk again, and the reread form disk to complete the operation.
We can naturally extend this idea to any number of passes, where
the data is read many times into main memory. However, we concentrate
on two-pass algorithms because:

+ Two passes are usually enough, even for very large relations.
+ Generalizing to more than two passes is not hard.

We begin with an implementation of the sorting operator $\tau$ that illustrates
the general approach: divide a relation $R$ for which $B(R) > M$
into chucks of size $M$, sort them, and then process the sorted
sublists in some fashion that requires only one block of each
sorted sublist in main memory at any one time.

### 4.1 Two-Phase, Multiway Merge-Sort

It's possible to sort very large relations in two passes using an
algorithm called *Two-Phase, Multiway Merge-Sort* (TPMMS). Suppose
we have $M$ main-memory buffers to use for the sort. TPMMS sorts
a relation $R$ as follows:

+ *Phase 1*: Repeatedly fill the $M$ buffers with new tuples from
$R$ and sort them, using any main-memory sorting algorithm.
Write out each *sorted sublist* to secondary storage.
+ *Phase 2*: Merge the sorted sublists. For this phase to work, there
can be at most $M - 1$ sorted sublists, which limits the size of $R$.
We allocate one input block to each sorted sublist and one block to the ouput.

In order for TPMMS to work, there must be no more than $M - 1$ sublsts.
Suppose $R$ fits on $B$ blocks. SInce each sublist consists of
$M$ blocks, the number of sublists is $B / M$. We thus require $B / M \leq M -1$,
or $B \leq M(M - 1)$ (or about $B \leq M^{2}$)

The algorithm requires us to read $B$ blocks in the first pass, and
another $B$ disk I/O's to write the sorted sublists. The sorted
sublists are each read again in the second pass, resulting in a total
of $3B$ disk I/O's.

### 4.2 Duplicate Elimination Using Sorting

To perform the $\delta(R)$ operation in two passes, we sort the tuples
of $R$ in sublists as in 2PMMS. In the second pass, we use the available
main memory to hold one block for each sorted sublist and one output
block. However, instead of sorting on the second pass, we repeatedly select
the first unconsidered tuple $t$ among all the sorted sublists.
We write one copy of $t$ to the output and eliminate from the
input blocks all occurrences of $t$.

### 4.3 Grouping and Aggregation Using Sorting

The two-pass algorithm for $\gamma_{L}(R)$ is quite similar to the
algorithm for $\delta(R)$ or 2PMMS. We summarize it as follows:

1. Read the tuples of $R$ into memory, $M$ blocks at a time. Sort the
tuples in each set of $M$ blocks, using the grouping attributes
of $L$ as the sort key. Write each sorted sublist to disk.
2. Use one main-memory buffer for each sublist, and initially load
the first block of each sublist into its buffer.
3. Repeatedly find the least value of the sort key (grouping attributes)
present among the first available tuples in the buffers. This value,
$v$, becomes the next group, for which we:
    1. Prepare to compute all the aggregates on list $L$ for this group.
    2. Examine each of the tuples with sort key $v$, and accumulate
    the needed aggregates.
    3. If a buffer becomes empty, replace it with the next block from
    the same sublist.

### 4.4 A Sort-Based Union Algorithm

When bag-union is wanted, there is no need to consider a two-pass
algorithm for $U_{B}$. However, the one-pass algorithm for $U_{S}$
only works when at least one relation is smaller than the available
main memory, so we must consider a two-pass algorithm for set union.

To compute $R \cup_{S} S$, we modify 2PMMS as follows:

1. In the first phase, create sorted sublists from both $R$ and $S$.
2. Use one main-memory buffer for each sublist of $R$ and $S$. Initialize
each with the first block from the corresponding sublist.
3. Repeatedly find the first remaining tuple $t$ among all the
buffers. Copy $t$ to the output, and remove from the buffers all
copies of $t$. Manage empty input buffers and a full output buffer
as for 2PMMS.

The IO cost is $3(B(R) + B(S))$. And $M$ should satisfy $B(R) + B(S) \leq M^2$,

### 4.5 Sort-Based Intersection and Difference

It's the same IDEA as section 4.4. Omit detail here.

### 4.6 A Simple Sort-Based Join Algorithm

When taking a join, the number of tuples from the two relations that
share a common value of the join attributes, and therefore need to
be in main memory simultaneously, can exceed what fits in memory. The
extreme example when there is only one value of the join attribute(s),
and every tuple of one relation joins with every tuple of the other relation.
In this situation, there is really no choice but to take a nested-loop
join of the two sets of tuples with a common value in the join-attribute(s).

To avoid facing this situation, we can try to reduce main-memory use
for other aspects of the algorithm, and thus make available a
large number of buffers to hold the tuples with a given join-attribute value.

Given relations $R(X, Y)$ and $S(Y, Z)$ to join, and given $M$ blocks
of main memory for buffers, we do the following:

1. Sort $R$, using 2PMNS, with $Y$ as the sort key.
2. Sort $S$ similarly.
3. Merge the sorted $R$ and $S$. We use only two buffers: one for
the current block of $R$ and the other for the current block of $S$.
The following steps are done repeatedly:
    1. Find the least value $y$ of the join attributes $Y$ that is currently
    at the front of the blocks for $R$ and $S$.
    2. If $y$ does not appear at the front of the other relation, then
    remove tuples with sort key $y$.
    3. Otherwise, identify all the tuples from both relations having sort key
    $y$. If necessary, read blocks from the sorted $R$ and/or $S$,
    until we are sure there are no more $y$'s in either relation. As many as
    $M$ buffers are available for this purpose.
    4. Output all the tuples that can be formed by joining tuples
    from $R$ and $S$ that have a common $Y$-value $y$.
    5. If either relation has no more unconsidered tuples in main
    memory, reload the buffer for that relation.

### 4.7 Analysis of Simple Sort-Join

First, we need to sort the $R$ and $S$ we should write the results
to the disk, so we will do four disk I/O's per block: $4(B(R) + B(S))$.

When we merge the sorted $R$ and $S$ to find the joined tuples, we read each
block of $R$ and $S$ a fifth time, using another 1500 disk I/O's.

+ The simple sort-join uses $5(B(R) + B(S))$ disk I/O's.
+ It require $B(R) \leq M^{2}$ and $B(S) \leq M^2$ to work
+ It also requires that the tuples with a common value for the join attributes fit
in $M$ blocks

### 4.8 A More Efficient Sort-Based Join

If we do not have to worry about very large numbers of tuples with a
common value for the join attribute(s), then we can save two disk I/O's
per block by combing the second phase of the sorts with the join itself.
We call this algorithm *sort-join*. To compute $R(X,Y) \bowtie S(Y,Z)$
using $M$ main-memory buffers:

1. Create sorted sublists of size $M$, using $Y$ as the sort key, for
both $R$ and $S$.
2. Bring the first block of each sublist into a buffer; we assume there
are no more than $M$ sublists in all.
3. Repeatedly find the least $Y$-value $y$ among the first available
tuples of all the sublists.

+ Disk I/O's cost is $3(B(R) + B(S))$.
+ $B(R) + B(S) \leq M^2$.

## 5. Two-Pass Algorithms Based on Hashing

There is a family of hash-based algorithms that attack the same problems.
The essential idea behind all these algorithms is as follows. If the
data is too big to store in main-memory buffers, hash all the tuples
of the argument or arguments using an appropriate hash key.

### 5.1 Partitioning Relations by Hashing

To begin, let us review the way we would take a relation $R$
and, using $M$ buffers, partition $R$ into $M - 1$ buckets of
roughly equal size. We shall assume that $h$ is the hash function, and that
$h$ takes complete tuples of $R$ as its argument. We associate
one buffer with each bucket.

```pseudocode
initialize M - 1 buckets using M -1 empty buffers;
FOR each block b of relation R DO BEGIN
  read block b into the Mth buffer;
  FOR each tuple t in b DO BEGIN
    IF the buffer for bucket h(t) has no room for t THEN
      BEGIN
        copy the buffer to disk;
        initialize a new empty block in that buffer;
      END;
    copy t to the buffer for bucket
    END;
  END;
END;
FOR each bucket DO
  IF the buffer for this bucket is not empty THEN
    write the buffer to disk;
``````pseudocode
initialize M - 1 buckets using M -1 empty buffers;
FOR each block b of relation R DO BEGIN
  read block b into the Mth buffer;
  FOR each tuple t in b DO BEGIN
    IF the buffer for bucket h(t) has no room for t THEN
      BEGIN
        copy the buffer to disk;
        initialize a new empty block in that buffer;
      END;
    copy t to the buffer for bucket
    END;
  END;
END;
FOR each bucket DO
  IF the buffer for this bucket is not empty THEN
    write the buffer to disk;
```

### 5.2 A Hash-Based Algorithm for Duplicate Elimination

We hash $R$ to $M - 1$ buckets. Note that two copies of the same tuple
$t$ will hash to the same bucket. Thus, we can examine one bucket at
a time, perform $\delta$ on that bucket in isolation, and take as
the answer the union of $\delta(R_{i})$, where $R_{i}$ is the portion
of $R$ that hashes to the $i$th bucket.

This method will work as long as the individual $R_{i}$'s are sufficiently
small to fit in main memory and thus allow a one-pass algorithm. Since we
may assume the hash function $h$ partitions $R$ into equal-sized buckets,
each $R_{i}$ will be approximately $B(R) / (M -1)$ blocks in size.
If that number of blocks is no larger than $M$, then the two-pass
will work. Thus, a conservative estimate is $B(R) \leq M^{2}$.

The number of disk I/O's is also similar to that of the sort-based
algorithm. We read each block of $R$ once as we hash its tuples,
and we write each block of each bucket to disk. We then read each
block of each bucket again in the one-pass algorithm that focuses on
that bucket. Thus, the total number of disk I/O's is $3B(R)$.

### 5.3 Hash-Based Grouping and Aggregation

To perform the $\gamma_{L}(R)$ operation, we again start by hashing all
the tuples of $R$ to $M - 1$ buckets. However, in order to make sure that
all tuples of the same group wind up in the same bucket, we must
choose a hash function that depends only on the grouping attributes
of the list $L$.

Having partitioned $R$ into buckets, we can then use the one-pass algorithm
for $\gamma$ to process each bucket in turn.

+ $B(R) \leq M^{2}$
+ Disk IO: $3B(R)$

### 5.4 Hash-Based Union, Intersection, and Difference

When the operation is binary, we must make sure that we use the same hash
function to hash tuples of both arguments. For example, to compute $R \cup_{S} S$,
we has both $R$ and $S$ to $M - 1$ buckets, say $R_{1}, R_{2}, \dots, R_{M-1}$
and $S_{1}, S_{2}, \dots, S_{M - 1}$. We then take the set-union of $R_{i}$
with $S_{i}$ for all $i$, and output the result. Note that if
a tuple $t$ appears in both $R$ and $S$, then for some $i$ we shall find
$t$ in both $R_{i}$ and $S_{i}$. Thus, we can the union of these two
buckets, we shall output only one copy of $t$, and there is no
possibility of introducing duplicates into the result.

To take the intersection or difference of $R$ and $S$, we create the
$2(M -1)$ buckets exactly as for set-union and apply the appropriate
one-pass algorithm to each pair of corresponding buckets.

Note that all these one-pass algorithms require $B(R) + B(S)$ disk IO's. Thus,
the total disk IO's are $3(B(R) + B(S))$. And we require that
$\min(B(R), B(S)) \leq M^{2}$.

### 5.5 The Hash-Join Algorithm

To compute $R(X, Y) \bowtie S(Y, Z)$ using a two-pass, hash-based
algorithm. We must use as the hash key just the join attributes.
Then we can be sure that if tuples of $R$ and $S$ join, they will
wind up in corresponding buckets $R_{i}$ and $S_{i}$ for some $i$.

+ Hash join requires $3(B(R) + B(S))$ disk I/O’s to perform its task.
+ It will work as long as approximately $\min(B(R), B(S)) \leq M^{2}$.

## 6. Index-Based Algorithms

Index-based algorithms are especially useful for the selection operator,
but algorithms for join and other binary operators also use indexes to
very good advantage.

### 6.1 Clustering and Nonclustering Indexes

We may also speak of *clustering indexes*, which are indexes on an attribute
or attributes such that all the tuples with a fixed value for
the search key of this index appear on roughly as few blocks as
can hold them.

### 6.2 Index-Based Selection

Suppose that the condition $C$ is of the form $a = v$, where $a$ is an
attribute for which an index exists, and $v$ is a value. Then one
can search the index with value $v$ and get pointers to exactly those
tuples of $R$ that have $a$-value $v$.

If the index on $R.a$ is a clustering index, then the number of disk I/O's to
retrieve the set $\sigma_{a = v}(R)$ will be average $B(R) / V(R,a)$

When the index on $R.a$ is nonclustering. To a first approximation, each tuple
retrieve will be on a different block, and we must access $T(R) / V(R,a)$ tuples.

### 6.3 Joining by Using an Index

Let us examine the natural join $R(X, Y) \bowtie S(Y, Z)$.

Four our first index-based join algorithm, suppose that $S$ has an index
on the attributes $Y$. Then one way to compute the join is to examine
each block of $R$, and within each block consider each tuple $t$. Let $t_{Y}$
be the component or components of $t$ corresponding to the attribute(s) $Y$.
Use the index to find all those tuples of $S$ that have $t_{Y}$ in their $Y$-components.

### 6.4 Joins Using a Sorted Index

When the index is B-tree, we have a number of other opportunities to
use the index. Perhaps the simplest is when we want to compute $R(X, Y) \bowtie S(Y, Z)$,
and we have such an index on $Y$ for either $R$ or $S$. We can then
perform an ordinary sort-join, but we do not have to perform the
intermediate step of sorting one of the relations on $Y$.
