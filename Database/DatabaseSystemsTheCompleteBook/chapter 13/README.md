# Chapter 13 The Secondary Storage Management

This chapter summarizes what we need to know about how a typical computer system
manage storage.

## 1. The Memory Hierarchy

### 1.1 The Memory Hierarchy

A schematic of the memory hierarchy is shown below.

![The memory hierarchy](https://s2.loli.net/2022/03/12/frjGknBsy4HwAUK.png)

Here are brief descriptions of the levels, from the lowest, or fastest-smallest level, up.

1. *Cache*.
2. *Main Memory*.
3. *Secondary Storage*.
4. *Tertiary Storage*

### 1.2 Transfer of Data Between Levels

Normally, data moves between adjacent levels of the hierarchy. At the secondary
and tertiary levels, accessing the desired data or finding the desired place
to store data takes a great deal of time, so each level is organized to transfer
large amounts of data to or from the level below.

Especially important for understanding the operation of a database system is
the fact that the disk is organized into *disk blocks* of perhaps 4-86 kilobytes.
Entire blocks are moved to or from a continuous section of main memory called a *buffer*.
Thus, a key technique for speeding up database operations is to arrange data
so that when one piece of a disk block is needed, it is likely that other data on
the same data will also be needed at about the same time.

### 1.3 Volatile and Nonvolatile Storage

An additional distinction among storage devices is whether they are
*volatile* or *nonvolatile*. A volatile device "forgets" what is
stored in it when the power goes off. A nonvolatile device is expected to keep
its contents intact even for long periods when the device is turned off or there
is a power failure.

### 1.4 Virtual Memory

Typical software executes in *virtual-memory*, an address space that is typically
32 bits.

## 2. Disks

The use of secondary storage is one of the important characteristics of a DBMS,
and secondary storage is almost exclusively based on magnetic disks.

### 2.1 Mechanics of Disks

The two principal moving pieces of a disk drive are shown below; they are
a *disk assembly* and a *head assembly*. The disk assembly consists of one
or more circular *platters* are covered with a thin layer of magnetic material,
on which bits are stored.

The disk is organized into *tracks*, which are concentric circles on
a single platter. The trackers that are at a fixed radius from the center,
among all the surfaces, form one *cylinder*.

![A typical disk](https://s2.loli.net/2022/03/14/pqHKQ9mVzwv7DAl.png)

Tracks are organized into *sectors*, which are segments of the
circle separated by *gaps* that are not magnetized to represent either 0 or 1.
The sector is an indivisible unit, as far as reading and writing the disk
is concerned. It is also indivisible as far as errors are concerned.
Should a portion of the magnetic layer be corrupted in some way, so that it cannot
store information, then the entire sector containing this portion cannot be used.
Blocks are logical units of data that are transferred between disk
and main memory; blocks consist of one or more sectors.

![Top view of a disk surface](https://s2.loli.net/2022/03/14/tVpAwB1lDNKLcvI.png)

The head assembly, holds the *disk heads*. For each surface there is one head, riding extremely
close to the surface but never touching it. A head reads the magnetism passing
under it, and can also alter the magnetism to write information on the disk.
The heads are each attached to an arm, and the arms for all the surfaces
move in and out together, being part of the rigid head assembly

### 2.2 The Disk Controller

One or more disk drives are controlled by a *disk controller*, which is a
small processor capable of:

+ Controlling the mechanical actuator that moves the head assembly, to
position the heads at a particular radius
+ Selecting a sector from among all those in the cylinder at which the
heads are positioned.
+ Transferring bits between the desired sector and the computer's main memory.
+ Possibly, buffering an entire track or more in local memory
of the disk controller, hoping that many sectors of this track
will be read soon.

### 2.3 Disk Access Characteristics

*Accessing* a block requires three steps, and each step has
an associated delay.

+ The disk controller positions the head assembly at the cylinder
containing the track on which the block is located. The time to do so is the *seek time*.
+ The disk controller waits while the first sector of the block moves under the head.
This time is called the *rotational latency*.
+ All the sectors and the gaps between them pass under the head, while
the disk controller reads or writes data in these sectors. This
delay is called the *transfer time*.

## 3. Accelerating Access to Secondary Storage

There are several things we can do to decrease the average time a disk
access takes, and thus improve the *throughput*. We begin this section
by arguing that the "I/O model" is the right one for measuring the time
database operations take. Then, we consider a number of techniques for
speeding up typical database accesses to disk:

1. Place blocks that are accessed together on the same cylinder, so
we can often avoid seek time, and possibly rotational latency as well.
2. Divide the data among several smaller disks rather than large one.
Having more head assemblies that can go after blocks independently
can increase the number of block accesses per uint time.
3. "Mirror" a disk: making two or more copies of the data on different disks.
In addition to saving the data in case one of the disk fails, this
strategy, like diving the data among several disks, let us access
several blocks at once.
4. Use a disk-scheduling algorithm.
5. Prefetch blocks to main memory in anticipation of their later use.

### 3.1 The I/O Model of Computation

Let us imagine a simple computer running a DBMS and trying to serve
a number of users who are performing queries and database modifications.
For the moment, assume our computer has one processor, one disk controller,
and one disk. The database itself is much too large to fit in main
memory. Key parts of the database may be buffered in main memory,
but generally, each piece of the database that one of the users
accesses will have to be retrieved initially from disk. The following
rule, which defines the *I/O model of computation*, can thus be assumed.

*Dominance of I/O cost*: The time taken to perform a disk access is
much larger than the time likely to be used manipulating that data
in main memory. Thus, the number of block accesses(*Disk I/O*) is a good
approximation to the time needed by the algorithm and should be minimized.

### 3.2 Organizing Data by Cylinders

Since seek time represents about half the time it takes to access
a block, it makes sense to store data that is likely to be accessed
together, such as relations, on a single cylinder, or on as many
adjacent cylinders as are needed. In fact, if we choose to read
all the blocks on a single track or on a cylinder consecutively,
then we can neglect all but the first seek time and the first rotational latency.

### 3.3 Using Multiple Disks

We can often improve the performance of our system if we replace one disk,
with many heads locked together, by several disks with their
independent heads.

### 3.4 Mirroring Disks

There are situations where it makes sense to have two or more disks
hold identical copies of data. If we have $n$ disks, each holding
the same data, the the rate at which we can read blocks goes up by
a factor of $n$

### 3.5 Disk Scheduling and the Elevator Algorithm

A simple and effective way to schedule large numbers of block
requests is known as *elevator algorithm*. We think of the disk
head as making sweeps across the disk, from innermost to outermost cylinder and
then back gain, just as an elevator makes vertical sweeps from the
bottom to top of a building and back again. As heads pass a cylinder,
they stop if there are one or more requests for blocks for that cylinder.
All theses blocks are read or written, as requested. The heads then
proceed in the same direction they were traveling until the next cylinder
with blocks to access is encountered. When the heads reach a position
where there are no requests ahead of them in their direction
of travel, they reverse direction.

### 3.6 Prefetching and Large-Scale Buffering

Our final suggestion for speeding up some secondary-memory algorithms
is called *prefetching* or sometimes *double buffering*.

## 4. Disk Failures

### 4.1 Intermittent Failures

An intermittent failure occurs if we try to read a sector, but the
correct content of that sector is not delivered to the disk controller.
If the controller has a way to tell that the sector is good or bad
then the controller can reissue the read request when bad data is read,
until the sector is returned correctly, or some preset limit,
like 100 tries, is reached.

### 4.2 Checksums

A simple form of checksum is based on the *parity* of all the bits
in the sector. If there is an odd number of 1's among a collection of bits,
we say the bits have *odd* parity and add a parity bit that is 1.
Similarly, if there is an even number of 1's among the bits, then
we say the bits have *even* parity and add parity bit 0. As a result:

+ The number of 1's among a collection of bits and their parity bit is always even.

### 4.3 Stable Storage

Checksum do not help us correct the error. To deal with the problems,
we can implement a policy known as *stable storage* on a disk or
on several disks. The general idea is that sectors are paired,
and each pair represents on sector-contents $X$. We shall refer
to the pair of sectors represents $X$ as the "left" and "right" copies, $X_{L}$ and $X_{R}$.

The stable-storage writing policy is:

+ Write the value of $X$ into $X_{L}$. Check that the value has status
"good"; If not, repeat the write. If after a set number of write
attempts, we have to successfully written $X$ into $X_{L}$, assume
that there is media failure in this sector. A fix-up such as
substituting a spare sector for $X_{L}$ must be adopted.
+ Repeat for $X_{R}$.

The stable-storage reading policy is to alternate trying to read
$X_{L}$ and $X_{R}$, until a good value is returned. Only if no
good value is returned after some large, prechosen number of tries,
is $X$ truly unreadable.

## 5. Arranging Data on Disk

A data element such as a tuple or object is represented by a *record*,
which consists of consecutive bytes in some disk block. Collections
such as relations are usually represented by placing the records that
represent their data elements in one or more blocks.

It is normal for a disk block to hold only elements of one relation,
although there are organizations where blocks hold tuples of
several relations.

### 5.1 Fixed-Length Records

The simplest sort of record consists of fixed-length *fields*, one for
each attribute of the represented tuple. Many machines allow more
efficient reading and writing main memory when data begins at an
address that is a multiple of 4 or 8; some even require us to do so.
Thus, it is common to begin all fields at a multiple of 4 or 8, as appropriate.
Space not used by the previous field is wasted. Note that, even
though records are kept in secondary, not main, memory, they are
manipulated in main memory. Thus it is necessary to lay out the
record so it can be moved to main memory and accessed efficiently there.

Often, the record begins with a *header*, a fixed-length region where
information about the record itself is kept. For example, we may want
to keep in the record:

1. A pointer to the schema for the data stored in the record. This information
helps us find the fields of the record.
2. The length of the record. This information helps us skip
over records without consulting the schema.
3. Timestamps indicating the time the record was last modified, or
last read. This information may be useful for implementing
database transactions.
4. Pointers to the fields of the record. This information can
substitute for schema information, and it will be important when
we consider variable-length fields.

We consider the following `MovieStar` schema.

```sql
CREATE TABLE MovieStar(
  name CHAR(30) PRIMARY KEY,
  address VARCHAR(255),
  gender CHAR(1),
  birthdate DATE
);
```

![Layout of records for tuples of the MovieStar relation](https://s2.loli.net/2022/08/17/bc2N4eOjU6ZHXmg.png)

### 5.2 Packing Fixed-Length Records into Blocks

Records representing tuples of a relation are stored in blocks of
the disk and moved into main memory when we need to access or update them.
The layout of a block that holds records is suggested below.

![A typical block holding records](https://s2.loli.net/2022/08/17/HzcoOnYBFwZNalh.png)

In addition to the records, there is a *block header* holding information
such as:

1. Links to one or more other blocks that are part of a network
of blocks for creating index to the tuples of a relation.
2. Information about the role played by this block in such a network.
3. Information about which relation the tuples of this block belong to.
4. A "directory" giving the offset of each record in the block.
5. Timestamp(s) indicating tht time of the block's last modification and/or access.

By far the simplest case is when the block holds tuples from
one relation, and the records for those tuples have a fixed format.
In that case, following the header, we pack as many records as
we can into the block and leave the remaining space unused.

## 6. Representing Block and Record Addresses

When in main memory, the address of a block is the virtual-memory address
of its first byte, and the address of a record within that block is the virtual-memory
address of the first byte of that record. However, in secondary storage,
the block is not part of the application's virtual-memory address
space.

In this section, we shall begin with a discussion of address spaces,
especially as they pertain to the common "client-server" architecture
for DBMS.

### 6.1 Addresses in Client-Server Systems

Commonly, a database system consists of a *server* process that
provides data from secondary storage to one or more *client* processes
that are applications using the data.

The server's data lives in a *database address space*. The addresses of
this space refer to blocks, and possibly to offsets within the block.
There are several ways that addresses in this address space can be represented:

+ *Physical Addresses*. These are byte strings that let us determine
the place within the secondary storage system where the block or
record an be found.
  + The host to which the storage is attached.
  + An identifier for the disk or other device on which the block is located.
  + The number of the cylinder of the disk.
  + The number of the track within the cylinder.
  + The number of the block within the track.
  + The offset of the beginning of record within the block.
+ *Logical Addresses*. Each block or record has a "logical address",
which is an arbitrary string of bytes of some fixed length. A *map table*,
stored on disk in a known location, relates logical to physical addresses.

### 6.2 Logical and Structured Addresses

One might wonder what the purpose of logical addresses could be. All
the information needed for a physical address is found in the map table,
and following logical pointers to records requires consulting the
map table and the going to the physical address. However, the level
of indirection involved in the map table allows us considerable flexibility.
For example, many data organizations require us to move records around,
either within a block or from block to block. If we use a map table, then all pointers to the record refer to this
map table, and all wa have to do when we move or delete the record is
to change the entry for that record in the table.

Many combinations of logical and physical addresses are possible as well,
yielding *structured* address schemes. For instance, one could us a physical
address for the block, and add the kye value for the record being referred to.

A similar, and very useful, combination of physical and logical addresses
is to keep in each block an *offset table* that holds the offsets of the
records within the block, as suggested below. Notice that the table
grows from the front end of the block, while the records are
placed starting at the end of the block. This strategy is useful
when the records need not be of equal length.

![A block with a table of offsets telling us the position of each record](https://s2.loli.net/2022/08/23/V7nHBFKa6YCdjGr.png)

The address of a record is now the physical address of its block
plus the offset of the entry in the block's offset table for that
block. This level of indirection within the block offers many of
the advantages of logical addresses, without the need for a global map table.

+ We can move the record around within the block, and all we have to do
is change the record's entry in the offset table
+ We can even allow the record to move to another block, if the offset
table entries are large enough to hold a *forwarding address* for the record.
+ We have an option, should the record be deleted, of leaving in its
offset-table entry a *tombstone*, a special value that indicates the
record has been deleted.

### 6.3 Pointer Swizzling

To avoid the cost of translating repeatedly from database addresses to
memory addresses, several techniques have been developed that are
collectively known as *pointer swizzling*. The general idea is that
when we move a block from secondary to main memory, pointers within
the block may be "swizzled", that is, translated from the database
address space to the virtual address space. Thus, a pointer actually consists of:

1. A bit indicating whether the pointer is currently a database address
or a (swizzled) memory address.
2. The database or memory pointer, as appropriate.

Below shows a simple situation in which the Block 1 has a record
with pointers to a second record on the same block and to a record
on another block. The figure also shows what might happen when Block 1
is copied to memory. The first pointer, which points within Block 1,
can be swizzled so it points directly to the memory address of the
target record.

However, if Block 2 is not in memory at this time, then we cannot swizzle
the second pointer; it must remain unswizzled, pointing to the
database address of its target. Should Block 2 be brought to memory
later, it becomes theoretically possible to swizzle the second pointer
of Block 1.

![Structure of a pointer when swizzling is used](https://s2.loli.net/2022/08/23/isWYG94XlHydIPV.png)

#### Automatic Swizzling

There are several strategies we can use to determine when to swizzle
pointers. If we use *automatic swizzling*, then as soon as a block
is brought into memory, we locate all its pointers and addresses
and enter them into the translation table if they are not already there.
These pointers include both pointers *from* records in the block
to elsewhere and the addresses of the block itself and/or its records,
if these are addressable items. We need some mechanism to
locate the pointers within the block. For example:

1. If the block holds records with a known schema, the schema will
tell us where in the records the pointers are found.
2. If the block is used for an index structure, then the block
will hold pointers at known locations.
3. We may keep within the block header a list of where the pointers are.

#### Swizzling on Demand

Another approach is to leave all pointers unswizzled when the
block is first brought into memory.

#### No Swizzling

Of course it is possible never to swizzle pointers.

### 6.4 Returning Blocks to Disk

When a block is moved from memory to disk, any pointers within that
block must be "unswizzled". However, we do not want each unswizzling operation
to require a search of the entire translation table. So we need
to change the data structure for translation table.

### 6.5 Pinned Records and Blocks

A block in memory is said to be *pinned* if it cannot at the moment
be written back to disk safely. A bit telling whether or not a block
is pinned can be located in the header of the block. There are many
reasons why a block could be pinned, including requirements of
a recovery system. Pointer swizzling introduces an important reason
why certain blocks must be pinned.

If a block $B_{1}$ has within it a swizzled pointer to some data item
in block $B_{2}$, then we must be very careful about moving block $B_{2}$
back to disk and reusing its main-memory buffer. The reason is that,
should we follow the pointer in $B_{1}$, it will lead us to the buffer,
which no longer holds $B_{2}$; in effect, the pointer has become dangling.

When we write a block back to disk, we not only need to "unswizzle" any
pointers in that block. We also need to make sure it is not pinned. Thus
the translation table must record, for each database address whose
data item is in memory, the places in memory where swizzled pointers
to that item exist. Two possible approaches are:

+ Keep the list of references to a memory address as a linked list attached
to the entry for that address in the translation table.
+ If memory addresses are significantly shorter than database addresses,
we can create the linked list in the space used for the pointers themselves.
  + The swizzled pointer.
  + Another pointer that forms part of a linked list of all occurrences of this pointer.

## 7. Variable-Length Data and Records

Until now, we have made the simplifying assumptions that records
have a fixed schema, and that the schema is a list of fixed-length fields.
However, in practice, we also wish to represent:

1. *Data items whose size varies*.
2. *Repeating fields*.
3. *Variable-format records*.
4. *Enormous fields*

### 7.1 Records With Variable-Length Fields

If one or more fields of a record have variable length, then the
record must contain enough information to let us find any field of
the record. A simple but effective scheme is to put all fixed-length
fields ahead of the variable-length fields. We then place in
the record header:

+ The length of the record.
+ Pointers to the beginnings of all the variable-length fields
other than first (which we know must immediately follow the
fixed-length fields).

Suppose we have movie-star records with name, address, gender,
and birthdate. We shall assume that the gender and birthdate are
fixed-length fields, taking 4 and 12 bytes, respectively.
However, both name and address will be represented by character strings
of whatever length is appropriate. Below suggests what a typical movie-star
record would look like. Note that no pointer to the beginning of the name
is needed.

![A MovieStar record with name and address implemented as variable-length character strings](https://s2.loli.net/2022/08/23/TIfz8BEQa3Ct2Xe.png)

### 7.2 Records With Repeating Fields

A similar situation occurs if a record contains a variable number
of occurrences of a field $F$, but the field itself is of fixed length.
It is sufficient to group all occurrences of field $F$ together and
put in the record header a pointer to the first. We can locate
all the occurrences of the field $F$ as follows. Let the number of bytes
devoted to one instance of field $F$ be $L$. We then add to the offset
for the field $F$ all integer multiples of $L$, starting at $0$,
then $L$, $2L$, $3L$, and so on.

Suppose we redesign our movie-star records to hold only the name
and address and pointers to all the movies of the start (which are variable-length).
Below shows how this type of record could be represented.

![A record with a repeating group of references to movies](https://s2.loli.net/2022/08/23/Crc6SvVKUN487Zt.png)

An alternative representation is to keep the record of fixed length,
and put the variable-length portion (fields of variable length or fields that repeat
an indefinite number of times) on a separate block. In the record itself we keep:

+ Pointers to the place where each repeating fields begins.
+ How many repetitions they are, or where the repetitions end.

Below shows the layout of a record for the problem above.

![Storing variable-length fields separately from the record](https://s2.loli.net/2022/08/23/P8pvUceLKIo6WnV.png)

There are advantages and disadvantages to using indirection for the
variable-length components of a record:

+ Keeping the record itself fixed-length allows records to be searched
more efficiently, minimizes the overhead in block headers, and allows
records to be moved within or among blocks with minimum effort.
+ On the other hand, storing variable-length components on another
block increases the number of disk I/O's needed to examine all components
of a record.

### 7.3 Variable-Format Records

An even more complex situation occurs when records do not have a fixed schema.

The simplest representation of a variable-format records is a sequence
of *tagged fields*, each of which consists of the value of the field
preceded by information about the role of this field, such as:

1. The attribute or field name.
2. The type of the field. If it is not apparent from the field name.
3. The length of the field.

![A record with tagged fields](https://s2.loli.net/2022/08/23/1MNumZOf8YnjGsU.png)

### 7.4 Records That Do Not Fit in a Block

Today, DBMS's frequently are used to manage datatypes with large values;
often values do not fit in one block. Typical example are video or audio "clips."

Often, these large values have a variable length, but even if the length
is fixed for all values of the type, we need special techniques to represent
values that are larger than blocks. In this section we shall consider
a technique called "spanned records".

Spanned records also are useful in situations where records are
smaller than blocks, put packing whole records into blocks wastes significant amounts of space.

The portion of a record that appears in one block is called a
*record fragment*. A record with two or more fragments is called *spanned*,
and records that do not cross a block boundary are *unspanned*.

If records can be spanned, then every record and record fragment requires
some extra header information:

+ Each record or fragment header must contain a bit telling whether
or not it is a fragment.
+ If it is a fragment, then it needs bits telling whether it is the
first or last fragment for its record.
+ If there is a next and/or previous fragment for the same record,
then the fragment needs pointers to these other fragments.

![Storing spanned records across blocks](https://s2.loli.net/2022/08/23/DFCQs2v8xiOGNqS.png)

### 7.5 BLOBs

Now, let us consider the representation of truly large values for
records or fields of records. Such values are often called *binary, large objects*, or BLOBs.
When a field has a BLOB as value, we must rethink at least two issues.

A BLOB must be stored on a sequence of blocks. Often we prefer
that these blocks are allocated consecutively on a cylinder or cylinders
of the disk, so the BLOB may be retrieved efficiently.

Moreover, it is possible that the BLOB needs to be retrieved so quickly,
that storing it one disk does not allow us to retrieve it fast
enough. Then, it is necessary to *stripe* the BLOB across several disks.
Thus, several blocks of the BLOB can be retrieved simultaneously,
increasing the retrieval rate.

## 8. Record Modification

### 8.1 Insertion

Let us consider insertion of new records into a relation. If the
records of a relation are kept in no particular order, we can
just find a block with some empty space, or get a new block if
there is none, and put the record here.

There is more of a problem when the tuples must be kept in some fixed
order, such as sorted by their primary key. If we need to insert a new record,
we first locate the appropriate block for that record. Suppose
first that there is space in the block to put the new record. Suppose first
that there is space in the block to put the new record. Since records
must be kept in order, we may have to slide records around in the block
to make space available at the proper point. If we need to slide
records, then the block organization shown below is useful.

![An offset table lets us slide records within a block to make room for new records](https://s2.loli.net/2022/08/23/2jclCpG9fN7kbnS.png)

If we can find room, that will be OK. However, there may be no
room in the block for the new record, in which case we have to
find room outside the block. There are two major approaches to
solving this problem.

+ *Find space on a nearby block*.
+ *Creating an overflow block*.

### 8.2 Deletion

When we delete a record, we may be able to reclaim its space. If we
use an offset table and records can slide around the block.

If we cannot slide records, we should maintain an available-space list
in the block header. Then we shall know where, and how large, the available
regions are, when a new record is inserted into the block.

There is one additional complication involved in deletion, which
we must remember regardless of what scheme we use for reorganizing blocks.
There may be pointers to the deleted record, and if so, we don't want
these pointers to dangle or wind up pointing to a new record that is ot
in the place of the deleted record. The usual technique, is
to place a *tombstone* in place of the record.

### 8.3 Update

When a fixed-length record is updated, there is no effect on the storage
system, because we know it can occupy exactly the same space
it did before the update. However, when a variable-length record
is updated, we have all the problems associated with both insertion and
deletion, except that it is never necessary to create a tombstone for the old
version of the record.
