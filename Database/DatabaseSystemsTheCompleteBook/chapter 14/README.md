# Chapter 14 Index Structures

In this chapter, we shall introduce the most common form of index in database
systems: the B-tree. We shall also discuss hash tables in secondary storage.

## 1. Index-Structure Basics

In this section, we introduce concepts that apply to all index structures.
Storage structures consist of *files*, which are similar to the files
used by operating systems. A *data file* may be used to store a relation,
for example. The data file may have one or more *index files*. Each index file
associates values of the search key with pointers to data-file
records that have that value for the attributes of the search key.

Indexes can be "dense", meaning there is an entry in the index file
for every record of the data file. They can be "parse", meaning that
only some of the data records are represented in the index, often one index entry
per block of the data file. Indexes can also be "primary" or "secondary".
A primary index determines the location of the records of the data file,
while a secondary index does not.

### 1.1 Sequential Files

A *sequential file* is created by sorting the tuples of a relation by
their primary key. The tuples are then distributed among blocks,
in this order.

![A dense index(left) on a sequential data file (right)](https://s2.loli.net/2022/03/24/RFzwrg57ya2mSDb.png)

### 1.2 Dense Indexes

If records are sorted, we can build on them a *dense index*, which is
a sequence of blocks holding only the keys of the records and pointers
to the records themselves. The index blocks of the dense index maintain
these keys in the same sorted order as in the file itself. Since keys
and pointers presumably take much less space than complete
records, we expect to use many fewer blocks for the index than for the
file itself. The index is especially advantageous when it, but not
the data file, can fit in main memory. Then, by using the index,
we can find any record given its search key, with only one disk I/O per lookup.

The dense index supports queries that ask for records with a given
search key value. There are several factors that make the index-based search
more efficient than it seems.

+ The number of index blocks is usually small compared with the number
of data blocks.
+ Since keys are sorted, we can use binary search.
+ The index may be small enough to be kept permanently in main
memory buffers.

### 1.3 Sparse Indexes

A sparse index typically has only one key-pointer pair per block
of the data file. It thus uses less space than a dense index,
at the expense of somewhat more time to find a record given its key.
You can only use a sparse index if the data file is sorted by the
*search key*, while a dense index can be used for any search key.

![A sparse index on a sequential file](https://s2.loli.net/2022/10/31/9irbV1EXdPHLscI.png)

### 1.4 Multiple Levels of Index

An index file can cover many blocks. Even if we use binary search
to find the desired index entry, we still may need to do many disk
I/O's to get to the record we want. By putting an index on the index, we
can make use of the first level of index more efficient.

![Adding a second level of sparse index](https://s2.loli.net/2022/10/31/wLCkxMQPlgjBdEG.png)

### 1.5 Secondary Indexes

A secondary index serves the purpose of any index: it is a data structure
that facilitates finding records given a value for one or more fields. However,
the secondary index does not determine the placement of records in
the data file. Rather, the secondary index tells us the current locations
of records; that location
may have been decided by a primary index on some other field. An import
consequence of the distinction between the primary and secondary
indexes is that:

+ Secondary indexes are always dense. It makes no sense to talk of a sparse,
secondary index. Since the secondary index does not influence location, we
could not use it to predict the location of any record whose key
was not mentioned in the index file explicitly

![A secondary index](https://s2.loli.net/2022/10/31/J1dra2ql5Vy9HWD.png)

However, the keys in the index file are sorted. The result is that the pointers
in one index block can go to many different data blocks, instead of one or a
few consecutive blocks. For example, to retrieve all the records with search
key 20, we are not only have to look at two index blocks, but we are sent
by their pointers to three different data blocks. Thus, using a secondary index
may result in many more disk I/O's than if we get the same number of records
via a primary index.

### 1.6 Indirection in Secondary Indexes

There is some wasted space, perhaps a significant amount of wastage, in the
structure suggested by above figure. If a search-key value appears $n$ times
in the data file, then the value is written $n$ times in the index file. It
would be better if we could write the key value once for all the pointers
to data records with that value.

A convenient way to avoid repeating values is to use a level of indirection,
called *buckets*, between the secondary index file and the data file.

![Saving space by using indirection in a secondary index](https://s2.loli.net/2022/10/31/ZGwxb5CEgHV6myQ.png)

There is an important advantage to using indirection with secondary indexes: often,
we can use the pointers in the buckets to answer queries without ever looking at
most of the records in the data file. Specifically, when there are several conditions
to a query, and each condition has a secondary index to help it, we can find the
bucket pointers that satisfy all the conditions by intersecting sets of pointers
in memory, and retrieving only the records pointed to by the surviving pointers. We
thus save the I/O cost of retrieving records that satisfy some, but not all.

For example, suppose we have secondary indexes with indirect buckets on both `studioName`
and `year`, and we are asked the query.

```sql
SELECT title
FROM movie
WHERE studioName = 'Disney' AND year = 2005;
```

Using the index on `studioName`,we find the pointers to all records for Disney movies,
but we do not yet bring any of those records from disk to memory. Instead, using the
index on `year`, we find the pointers to all movies of 2005. We then intersect the two
sets of pointers, getting exactly the movies that were made by Disney in 2005. Finally,
we retrieve from disk all data blocks holding one or more of these movies, thus
retrieving the minimum possible number of blocks.

## 2. B-Trees

There is a more general structure that is commonly used in commercial systems. This
family of data structures is called *B-trees*, and the particular variant that
is most often used is known as *B+ tree*. In essence:

+ B-trees automatically maintain as many levels of index as is appropriate for
the size of the file being indexed.
+ B-trees manage the space on the blocks they use so that every block is between
half used and completely full.

### 2.1 The structure of B-trees

A B-tree organizes its blocks into a tree that is *balanced*, meaning that all
paths from the root to a leaf have the same length. Typically, there are three
layers in a B-tree: the root, an intermediate layer, and leaves, but any number
of layers is possible.

There is a parameter $n$ associated with each B-tree index, and this parameter
determines the layout of all blocks of the B-tree. Each block will have space
for $n$ search-key values and $n + 1$ pointers.

There are several important rules about what can appear in the blocks of a B-tree:

+ The keys in leaf nodes are copies of keys from the data file. These keys are
distributed among the leaves in sorted order, from left to right.
+ At the root, there are at least two used pointers. All pointers to B-tree blocks
at the level below.
+ At a leaf, the last pointer points to the next leaf block to the right. Among the
other $n$ pointers in a leaf block, at least $(n + 1) / 2$ of these pointers are
used and point to data records; unused pointers are null and do not point anywhere.
The $i$th pointer, if it is used, points to a record with the $i$th key.
+ At an interior node, all $n + 1$ pointers can be used to point to B-tree blocks
at the next lower level. At least $(n + 1) / 2$ of them are actually used. If $j$
pointers are used, then there will be $j - 1$ keys, says $K_{1}, K_{2}, \dots, K_{j - 1}$.
+ All used pointers and their keys appear at the beginning of the block, with the
exception of the $(n + 1)$st pointer in a leaf.

### 2.2 Applications of B-trees

The B-tree is a powerful tool for building indexes. Here are some examples:

+ The search key of the B-tree is the primary key for the data file, and the index is dense.
That is, there is one key-pointer pair in a leaf for every record of the data file.
+ The data file is sorted by its primary key, and the B-tree is a sparse index with one
key-pointer pair at a leaf for each block of the data file.
+ The data file is sorted by attribute that is not a key.

### 2.3 Lookup in B-Trees

Suppose we have a B-tree index and we want to find a record with search key value $K$. We
search for $K$ recursively, starting at the root and ending at a leaf. The search procedure
is:

*Basis*: If we are at a leaf, look among the keys there. If the $i$th key is $K$, then the
$i$th pointer will take us to the desired record.

*Induction*: If we are at an interior node with keys $K_{1}, K_{2}, \dots, K_{n}$, do linear
search.

### 2.4 Range Queries

B-trees are useful not only for queries in which a single value of the search key is sought,
but for queries in which a range of values are asked for. Typically, *range queries* have
a term in the `WHERE`-clause that compares the search key with a value or values, using one
of the comparison operations:

```sql
SELECT * FROM R WHERE R.k >= 10 AND R.k <= 25
```

### 2.5 Insertion Into B-Trees

The insertion is recursive:

+ We try to find a place for the new key in the appropriate leaf, and we put it there
if there is room.
+ If there is no room in the proper leaf, we split the leaf into two and divide the keys
between the two new nodes.
+ The splitting of nodes at one level appears to the level above as if a new key-pointer
pair needs to be inserted at that higher level. We may thus *recursively* apply this
strategy to insert at the next level: if there is room, insert it; if not; split the
parent node and continue up the tree.
+ As an exception, if we try to insert into the root, and there is no room, then we split
the root into two nodes and create a new root at the next higher level.

## 2.6 Deletion From B-Trees

If we are to delete a record with a given key $K$, we must first locate that record and its
key-pointer pair in a leaf of the B-tree.

If the B-tree nde from which a deletion occurred still has at least the minimum number of
keys and pointers, then there is nothing more to be done. However, it is possible that the
node was right at the minimum occupancy before the deletion, so after deletion the constraint
on the number of keys is violated. We then need to do one of two things for a node $N$ whose
contents are subminimum.

## 3. Hash Tables

There are a number of data structures involving a hash table that are useful
as indexes. In such a structure there is a *hash function* $h$ that takes a
search key as an argument and computes from it an integer in the range $0$ to
$N - 1$, where $B$ is the number of *buckets*. A *bucket array*, which is
an array indexed from $0$ to $B - 1$, holds the headers of $B$ linked lists,
one for each bucket of the array.

### 3.1 Secondary-Storage Hash Tables

A hash table that holds a very large number of records, so many that they must
be kept mainly in secondary storage, differs from the main-memory version in
small but important ways.

+ The bucket array consists of blocks.
+ If a bucket has too many records, a chain of overflow blocks can be added to
the bucket to hold many records.

### 3.2 Insertion Into a Hash Table

When a new record with search key $K$ must be inserted, we compute $h(K)$. If
the bucket numbered $h(K)$ has space, then we insert the record into the block
for this bucket, or into one of the overflow blocks on its chain if there is
no room in the first block.

### 3.4 Hash-Table Deletion

Just like insertion.

### 3.4 Efficiency of Hash Table Indexes

Ideally, there are enough buckets that most of them fit on one block. If so,
then the typical lookup takes only one disk I/O, and insertion or deletion from
the file takes only two disk I/O's.

However, if the file grows, then we shall eventually reach a situation where
there are many blocks in teh chain for a typical bucket. If so, we need to
search long lists of blocks, taking at least one disk I/O per block.

The hash tables we have examined so far are called *static hash tables*, because
$B$, the number of buckets, never changes. However, there are several kinds of
*dynamic hash tables*, where $B$ is allowed to vary so it approximates the
number of records divided by the number of records that can fit on a block.

+ Extensible hashing.
+ Linear hashing.

### 3.5 Extensible Hash Tables

Our first approach to dynamic hashing is called *extensible hash tables*. The major
additions to the simpler static hash table structure are:

1. There is a level of indirection for the buckets. That is, an array of pointers to
blocks represents the buckets.
2. The array of pointers can grow. Its length is always a power of 2.
3. However, there does not have to be a data block for each bucket; certain buckets
can share a block.
4. The hash function $h$ computes for each key a sequence of $k$ bits for some large $k$.
However, the bucket numbers will at all times use some smaller number of bits, say
$i$ bits. The bucket array will have $2^{i}$ entries when $i$ is the number of bits used.

### 3.6 Insertion Into Extensible Hash Tables

Insertion into an extensible hash table begins like insertion into a static hash
table. To insert a record with search key $K$, we compute $h(K)$m take the first $i$
bits of this bit sequence, and go to the entry of the bucket array indexed by these
$i$ bits.

We follow the pointer in this entry of the bucket array and arrive at a block $B$. If
there is no room, then there are two possibilities, depending on the number of $j$,
which indicates how many bits of the hash value are used to determine membership in
block $B$.

1. If $j < i$, the nothing needs to be done to the bucket array.
    1. Split block $B$ into the two.
    2. Distribute records in $B$ to the two blocks, based on the value of their $(j+1)$st
    bits.
    3. Put $j + 1$ in each block's nud to indicate the number of bits used to determine
    membership.
    4. Adjust the pointers in the bucket array so entries that formerly pointed to $B$
    now point either to $B$ or the new block, depending on their $(j+1)$st bit.
2. If $j = i$, then we must first increment $i$ by 1. We double the length of the bucket
array, so it now has $2^{i + 1}$ entries. Suppose $w$ is a sequence of $i$ bits indexing
one of the entries in the previous bucket array. In the new bucket array, the entries
indexed by both $w0$ and $w1$ each point to the same block that the $w$ entry used to
point to.
