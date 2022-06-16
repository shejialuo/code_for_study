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

![A sparse index on a sequential file]()

### 1.4 Multiple Levels of Index

An index file can cover many blocks. Even if we use binary search
to find the desired index entry, we still may need to do many disk
I/O's to get to the record we want. By putting an index on the index, we
can make use of the first level of index more efficient.

![Adding a second level of sparse index]()

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

![A secondary index]()

### 1.6 Applications of Secondary Indexes

## 2. B-Trees
