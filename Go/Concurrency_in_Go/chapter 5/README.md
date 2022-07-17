# 5. Goroutines and the Go Runtime

## 5.1 Work Stealing

Go will handle multiplexing goroutines onto OS threads for you.
The algorithm it uses to do this is known as a *work stealing* strategy.

First, let's look at a naive strategy for sharing work across many
processors, something called *fair scheduling*. In an effort to ensure
all processors were equally utilized, we could evenly distributed
the load between all available processors. Imagine there are
$n$ processors and $x$ tasks to perform. In the fair  scheduling strategy,
each processor would get $x / n$ tasks.

Unfortunately, there are problems with this approach. o models concurrency
using a fork-join model. In a fork-join paradigm, tasks are likely
dependent on one another, and it turns out naively splitting them
among processors will likely cause one of the processors to be
underutilized. Not only that, but it can also lead to poor cache
locality as tasks that require the same data are scheduled on
other processors.

We could first use a FIFO queue: work tasks get scheduled into the
queue, and our processors dequeue tasks as they have capacity, or
block on joins.

The next leap we could make is to decentralize the work queues. We could
give each processor its own thread and a double-ended queue

We've solved problem with a central data structure under high contention
but what about the problems with cache locality and processor utilization?

In Go, Forks are when goroutines are started, and join points are
when two or more goroutines are synchronized through channels or types
in the `sync` package. The work stealing algorithm follows a few
basic rules. Given a thread of execution:

+ At a fork point, add tasks to the tail of the deque associated with the thread.
+ If the thread is idle, steal work from the head of the deque associated with
some other random thread.
+ At a join point that cannot be realized yet, pop work off the
tail of the thread's own deque.
+ If the thread's deque is empty, either:
  + Stall at a join.
  + Steal work from the head of a random thread's associated deque.
