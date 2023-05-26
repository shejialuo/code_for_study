# Chapter 2 Mutual Exclusion

Mutual exclusion is perhaps the most prevalent form of coordination in multiprocessor
programming. This chapter covers classical mutual exclusion algorithms.

## 2.1 Time

Reasoning about concurrent computation is mostly reasoning about time. We need a simple
but unambiguous language to talk about events and durations in time.

A thread is a *state machine*, and its state transitions are called *events*.

Events are *instantaneous*: they occur at a single instant of time. It is convenient to
require that events are never simultaneous: distinct events occur at distinct times. (As
a practical matter, if we are unsure about the order of two events that happen very close
in time, then any order will do). A thread $A$ produces a sequence of events $a_{0},a_{1},\dots$
threads typically contain loops, so a single program statement can produce many events. We
denote the $j$-th occurrence of an event $a_{i}$ by $a_{i}^{j}$. One event a precedes another
event $b$, written $a \to b$, if $a$ occurs at an earlier time.

Let $a_{0}$ and $a_{1}$ be events such that $a_{0} \to a_{1}$. An *interval* $(a_{0},a_{1})$
is the duration between $a_{0}$ and $a_{1}$. Interval $I_{A} = (a_{0}, a_{1})$ precedes
$I_{B} = (b_{0}, b_{1})$, written $I_{A} \to I_{B}$, if $a_{1} \to B_{0}$. Intervals that are
unrelated by $\to$ are said to be *concurrent*. By analogy with events, we denote the $j$-th
execution of interval $I_{A}$ by $I_{A}^{j}$.

## 2.2 Critical Sections

We now formalize the properties that a good `Lock` algorithm should satisfy. Let $CS_{A}^{j}$ be
the interval during which $A$ executes the critical section for the $j$-th time. Let us assume,
for simplicity, that each thread acquires and releases the lock infinitely often, with other work
taking place in the meantime.

+ *Mutual Exclusion* Critical sections of different threads do not overlap. For threads $A$ and
$B$, and integers $j$ and $k$m either $CS_{A}^{k} \to CS_{B}^{j}$ or $CS_{B}^{j} \to CS_{A}^{k}$.
+ *Freedom from Deadlock* If some thread attempts to acquire the lock, then some thread will
succeed in acquiring the lock.
+ *Freedom from Starvation* Every thread that attempts to acquire the lock eventually succeeds.

[Lock Interface](./lock.hpp)

## 2.3 2-Thread Solutions

### 2.1.3 The LockOne Class

Below shows the `LockOne` algorithm. Our 2-thread lock algorithms follow the following conventions:
the threads have ids 0 and 1, the calling thread has $i$, and the other $j = i -1$. Each thread
acquires its index by calling `ThreadID.get()`.

[LockOne](./lock_one.cpp)

*Lemma* 2.3.1. The `LockOne` algorithm satisfies mutual exclusion.

*Proof*: Suppose not. Then there exist integers $j$ and $k$ such that
$CS_{A}^{j} \nrightarrow CS_{B}^{j}$
