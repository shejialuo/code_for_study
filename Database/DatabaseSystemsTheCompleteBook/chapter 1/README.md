# Chapter 1 The Worlds of Database Systems

## 1. Overview of a Database Management System

In the figure below we see an outline of a complete DBMS. Single boxes represent
system component, while double boxes represent in-memory data structures. The
solid lines indicate control and data flow, while dashed lines indicate data flow
only.

![Database management system components](https://i.loli.net/2021/09/19/9pKXeGxYhc2dgsO.png)

First, at the top, we suggest that there are two distinct sources of commands to
the DBMS:

+ Conventional users and application programs that ask for data or modify data.
+ A database administrator: a person or persons responsible for the structure or
schema of the database.

### 1.1 Data-Definition Language Commands

The second kind of command is the simpler to process, and we show its trail
begging at the upper right side above. It is entered by the DBA. These schema-altering
data-definition language commands are parsed by a DDL processor and passed to
the execution engine, which then goes through the index/file/record manager to
alter the *metadata*.

### 1.2 Overview of Query Processing

A user or an application program initiates some action, using the data-manipulation
language. This command does not affect the schema of the database, but may affect
the content of the database or will extract data from the database. DML statements
are handled by two separate subsystems.

#### 1.2.1 Answering the Query

The query is parsed and optimized by a **query compiler**. The resulting
**query plan** is passed to the **execution engine**. The execution engine issues
a sequence of requests for small pieces of data, typically records or tuples of
a relation, to a resource manager that knows about data files, the format and
size of records in those files, and **index files**, which help find elements of
data files quickly.

The requests for data are passed to the **buffer manager**. The buffer manager's
task is to bring appropriate portions of the data from secondary storage where
it is kept permanently, to the main-memory buffers.

#### 1.2.2 Transaction Processing

Queries and other DML actions are grouped into **transactions**, which are units
that must be executed atomically and in isolation from one another. Any query or
modification action can be a transaction by itself. In addition, the execution
of transactions must be **durable**, meaning that effect of any completed
transaction must be preserved even if the system fails in some way right after
completion of the transaction. We divide the transaction processor into two major
parts:

+ A **concurrency-control manager**, or **scheduler**, responsible for assuring
atomicity and isolation of transactions.
+ A **logging and recovery manager**, responsible for the durability of transactions.

### 1.3 Storage and Buffer Management

To perform any useful operation on data, that data must be in main memory. It is
the job of the **storage manager** to control the placement of data on disk and
its movement between disk and main memory.

The **buffer manager** is responsible for partitioning the available main memory
into buffers, which are page-sized regions into which disk blocks can be transferred.

### 1.4 Transaction Processing

It is normal to group one or more database operations into a **transaction**,
which is a unit of work that must be executed atomically and in apparent isolation
from other transactions. In addition, a DBMS offers the guarantee of durability:
that the work of a completed transaction will never be lost. The **transaction**
**manager** therefore accepts **transaction commands** from an application, which
tell the transaction manager when transactions begin and end, as well as information
about the expectations of the application (some may not wish to require atomicity,
for example). The transaction processor performs the following tasks:

+ Logging
+ Concurrency control
+ Deadlock resolution
