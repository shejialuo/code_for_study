# 1. Why parallel computing

Simply adding more processors will not magically improve the performance
of the vast majority of **serial** programs. Programs are unaware
of the existence of multiple processors, and the performance
of such a program on a system with multiple processors will
be effectively the same as its performance on a single processor
of the multiprocessor system.

All of this raises a number of questions:

+ Why do we care? Aren't single-processor systems fast enough?
+ Why build parallel systems?
+ Why can't we write programs that will automatically convert serial
programs into parallel programs.

## 1.1 Why we need ever-increasing performance

As our computational power increases, the number of problems
that we can seriously consider solving also increases.

## 1.2 Why we're building parallel systems

Much of the tremendous increase in single-processor performance was
driven by the ever-increasing density of transistors. Most of this
power is dissipated as heat, and when an integrated circuit gets
too hot, it becomes unreliable. Therefore it is becoming impossible
to continue to increase the speed of integrated circuits.

How then, can we continue to build ever more powerful computers? The
answer is *parallelism*.

## 1.3 Why we need to write parallel programs

Most programs that have been written for conventional, single-core
systems cannot exploit the presence of multiple cores. We can
run multiple instances of a program on a multicore system, but this
is often of little help. For example, being able to run multiple instances
of our favorite game isn't really what we want.

## 1.4 How do we write parallel programs?

There are a number of possible answers to this question, but most
of them depend on the basic idea of *partitioning* the work to be
done among the cores. There are two widely used approaches:

+ **task-parallelism**: we partition the various tasks carried out in
solving the problem among the cores.
+ **data-parallelism**: we partition the data used in solving the problem
among the cores, and each core carries out more or less similar operations
on its part of the data.

## 1.5 What we'll be doing

Our purpose is to learn the basics of programming parallel computers
using the C language and four different API:

+ Message-Passing Interface(MPI).
+ POSIX threads(Pthreads).
+ OpenMP
+ CUDA

You may wonder why we're leaning about four different APIs instead of
just one. The answer has to do with both the extensions and
parallel systems. Currently, there are two main ways of classifying
the parallel systems: one is to consider the memory that the
different cores have access to, and the other is to consider
whether the cores can operate independently of each other.

In the memory classification, we'll be focusing on *shared-memory*
systems and *distributed-memory* systems. In a shared-memory
system, the cores can shared access to the computer's memory.
In a distributed-memory system, each core has its own, private memory,
and the cores can communicate explicitly by sending messages
across a network.

<!-- TODO: Add the picture -->
![A shared memory system and a distributed memory system](.)

The second classification divides parallel systems according to
the number of independent instruction streams and the number of
independent data streams. In one type of system, the cores can be
thought of as conventional processors, so they have their own
control units, and they are capable of operating independent of
each other. Each core can manage its own instruction stream and its
own data stream. So this type of system is called a MIMD
(Multiple-Instruction Multiple-Data) system.

An alternative is to have a parallel system with cores that are
not capable of managing their own instruction streams: they can
be thought of as cores with no control unit. Rather, the cores
