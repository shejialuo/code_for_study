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

