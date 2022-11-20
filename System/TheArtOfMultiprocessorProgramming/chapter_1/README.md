# Chapter 1 Introduction

This book focuses on how to program multiprocessors that communicate via a shared
memory. Such systems are often called *shared-memory multiprocessors*.

We approach multiprocessor programming from two complementary directions: principles
and practice. In the *principles* part, we focus on *computability*: figuring out
what can be computed in an asynchronous concurrent environment. We use an idealized
model of computation in which multiple concurrent *threads* manipulate a set of
shared objects.

An important step in the understanding of computability is the specification and
verification of what a given program actually does. This is perhaps the best described
as *program correctness*. The correctness of multiprocessor programs,by their very
nature, is more complex than that of their sequential counterparts, and requires a
different set of tools, even for the purpose of "informal reasoning". Sequential
correctness is mostly concerned with safety properties. A *safety* property states
that some "bad thing" never happens. Concurrent correctness is also concerned with
safety, but the problem is much harder because safety must be ensured despite the
vast number of ways that the steps of concurrent threads can be interleaved. Equally
important, concurrent correctness encompasses a variety of *liveness* properties that
have no counterparts in the sequential world. A *liveness* property states that a
particular good thing will happen.

The second part of the book deals with the *practice* of multiprocessor programming, and
focuses on performance.

## 1.1 Shared Objects and Synchronization

On the first day of your new job, your boss asks you to find all primes between 1 and $10^{10}$,
using a parallel machine that supports ten concurrent threads.

As a first attempt, you might consider giving each thread an equal share of the input domain. Each
thread might check $10^{9}$ numbers.

[staticPrimeFinder.cpp](./staticPrimeFinder.cpp)

This approach fail. Equal ranges of inputs do not necessarily produce equal amounts of work. Primes
do not occur uniformly: there  are many primes between 1 and $10^9$, but hardly any between
$9 \times 10^{9}$ and $10^{10}$. To make matters worse, the computation time per time is not the same
in all ranges: it usually takes longer to test whether a large number is prime than a small number.
In short, there is no reason to believe that the work will be divided equally among the threads, and
it is not clear even which threads will have the most work.

A more promising way to split the work among the threads is to assign each thread one integer at a time.
When a thread is finished with testing an integer, it asks for another. To this end, we introduce a
*shared counter*, an object that encapsulates an integer value, and that provides a `getAndIncrement()`
method that it increments its value, and returns the counter's prior value to the caller.

[dynamicPrimeFinder.cpp](./dynamicPrimeFinder.cpp)

## 1.2 A Fable

Alice and Bob are neighbors, and they share a yard. Alice owns a cat and Bob owns a dog. Both pets like
to run around in the yard, but they do not get along.

### 1.2.1 Properties of Mutual Exclusion

*Mutual exclusion* is only one of several properties of interest. Another property of compelling interest
is *starvation-freedom* (sometimes called *lockout-freedom*). The last property of interest concerns
*waiting*.

### 1.2.2 The Moral

Two kinds of communication occur naturally in concurrent systems:

+ *Transient* communications requires both parties to participate at the same time.
+ *Persistent* communication allows the sender and receiver to participate at different times.

Mutual exclusion requires persistent communication.

## 1.3 The Producer-Consumer Problem

Mutual exclusion is not the only problem worth investigating. Eventually, Alice and Bob fall in love and
marry. However, they divorce. ing?) The judge gives Alice custody of the pets, and tells Bob to feed them. The pets
now get along with one another, but they side with Alice, and attack Bob whenever they see him. As a result,
Alice and Bob need to devise a protocol for Bob to deliver food to the pets without Bob and the pets being in the yard together. Moreover, the protocol should not waste anyone’s time: Alice does not want to release her pets into the yard unless there is food there, and Bob does not want to enter the yard unless the pets have consumed all the food. This problem is known as the *producer–consumer* problem.

## 1.4 The Reader-Writers Problem

The Readers–Writers Problem Bob and Alice eventually decide they love their pets so much they need to
communicate simple messages about them. Bob puts up a billboard in front of his house. The billboard
holds a sequence of large tiles, each tile holding a single letter. Bob, at his leisure, posts a
message on the bulletin board by lifting one tile at a time. Alice, at her leisure, reads the message
by looking at the billboard through a telescope, one tile at a time.

## 1.5 The Harsh Realities of Parallelization

In an ideal world, upgrading from a uniprocessor to an $n$-way multiprocessor should provide about
an $n$-fold increase in computational power. In practice, sadly, this never happens. The primary
reason for this is that most real-world computational problems cannot be effectively parallelized
without incurring the costs of inter-processor communication and coordination.

The formula we need is called *Amdahl's Law*. It captures the notion that the extent to which we
can speed up any complex job is limited by how much of the job must be executed sequentially.

Define the *speedup* $S$ of a job to be the ratio between the time it takes one processor to
complete the job versus the time it takes $n$ concurrent processors to complete the same job.
*Amdahl's Law* characterizes the maximum speedup $S$ that can be achieved by $n$ processors
collaborating on an application, where $p$ is the fraction of the job that can be executed in
parallel. Assume, for simplicity, that it takes (normalized) time $1$ for a single processor
to complete the job. With $n$ concurrent processors, the parallel part takes time $p /n$ and
the sequential part takes time $1 - p$. Overall, the paralleled computation takes time:

$$
1 - p + \frac{p}{n}
$$

Amdahl's Law says that the speedup, that is, the ratio between the sequential time and the
parallel time is:

$$
S = \frac{1}{1 - p + \frac{p}{n}}
$$
