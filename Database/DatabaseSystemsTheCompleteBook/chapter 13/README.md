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
we can often avoid seek time, and possibly rotational