# Chapter 2 asyncio basics

## 2.1 Introducing coroutines

### 2.1.1 Creating coroutines with the async keyword

[asyncio_run_example](./asyncio_run_example.py)

`asyncio.run` is doing a few important things in this scenario. First, it creates
a brand-new event. Once it successfully does so, it takes whichever coroutine we
pass into it and runs it until it completes, returning the result. This function
will also do some cleanup of anything that might be left running after the main
coroutine finishes. Once everything has finished, it shuts down and closes the
event loop.

### 2.1.2 Pausing execution with the await keyword

The real benefit of asyncio is being able to pause execution to let the event
loop run other tasks during a long-running operation. To pause execution, we use
`await` keyword. The `await` keyword is usually followed by a call to a coroutine.

Using the `await` keyword will cause the coroutine following it to be run, unlike
calling a coroutine directly, which produces a coroutine object. The `await` expression
will also pause the coroutine where it is contained in until the coroutine we awaited
finished and returns a result.

[asyncio_wait_example.py](./asyncio_wait_example.py)

## 2.2 Introducing long-running coroutines with sleep

We can use `asyncio.sleep` to make a coroutine "sleep" for a given number of seconds.
This will pause our coroutine for the time we give it.

[asyncio_two_coroutine_example.py](./asyncio_two_coroutine_example.py)

In this example, the two coroutines run sequentially. We want to run `add_one` concurrently
with `hello_world`. To achieve this, we'll need to introduce a concept called *tasks*.

## 2.3 Running concurrently with tasks

When we call a coroutine directly, we don't put it on the event loop to run. Instead,
we get a coroutine object that we then need to either use the `await` keyword on it or
pass it in to `asyncio.run` to run and get a value. *But, we can't run anything concurrently*.
To run coroutines concurrently, we'll need to introduce *tasks*.

Tasks are wrappers around a coroutine that schedule a coroutine to run on the event
loop as soon as possible. This scheduling and execution happen in a non-blocking
fashion, meaning that, once we create a task, we can execute other code instantly
while the task is running.

### 2.3.1 The basics of creating tasks

Creating a task is achieved by using the `asyncio.create_task` function. When we call
this function, we give it a coroutine to run, and it returns a task object instantly.
Once we have a task object, we can put it in an `await` expression that will extract
the return value once it is complete.

[asyncio_task_example.py](./asyncio_task_example.py)

### 2.3.2 Running multiple tasks concurrently

Given that tasks are created instantly and are scheduled to run as soon as possible,
this allows us to run many long-running tasks concurrently. We can do this by
sequentially starting multiple tasks with our long-running coroutine.

[asyncio_tasks_example.py](./asyncio_tasks_example.py)

## 2.4 Cancelling tasks and setting timeouts

### 2.4.1 Cancelling tasks

Canceling a task is straightforward. Each task object has a method named `cancel`,
which we can call whenever we'd like to stop a task. Canceling a task will cause
that task to raise a `CancelledError` when we `await` it, which we can then
handle as needed.

[asyncio_cancel_task_example](./asyncio_cancel_task_example.py)

### 2.4.2 Setting a timeout and cancelling with wait_for

asyncio provides the functionality through a function called `asyncio.wait_for`.
This function takes in a coroutine or task object, and a timeout specified in seconds.
It then returns a coroutine that we can `await`. If the task takes more time
to complete than the timeout we gave it, a `TimeoutException` will be raised.

[asyncio_waitfor_example](./asyncio_waitfor_example.py)

In certain circumstances we may want to keep our coroutine running. For example,
we may want to inform a user that something is taking longer than expected after
a certain amount of time but not cancel the task when the timeout is exceeded.

To do this we can wrap our task with the `asyncio.shield` function. This
function will prevent cancellation of the coroutine we pass in.

[asyncio_wait_shield_example](./asyncio_wait_shield_example.py)

## 2.5 Tasks, coroutines, futures, and awaitables

### 2.5.1 Introducing futures

A `future` is a Python object that contains a single value that you expect to
get at some point in the future but may not yet have value. Usually, when you
create a `future`, it does not have any value it wraps around because it doesn't
yet exist. In this state, it is considered incomplete, unresolved, or simply
not done. Then, once you get a result, you can set the value of the `future`.
This will complete the `future`; at that time, we can consider it finished and
extract the result from the `future`.

[simple_future_example](./simple_future_example.py)

Futures can also be used in `await` expressions.

[await_future_example](./await_future_example.py)

### 2.5.2 The relationship between futures, tasks, and coroutines

There is a strong relationship between tasks and futures. In fact, `task`
directly inherits rom `future`. A `task` can be thought as a combination
of both a coroutine and a `future`. When we create a `task`, we create
an empty `future` and running the coroutine. Then, when the coroutine
has completed with either an exception or a result, we set the result
or exception of the `future`.

All these types can be used in an `await` expression. The common thread between
these is the `Awaitable` abstract base class. This class defines one
abstract double underscore method `__await__`.

Coroutines inherit directly from `Awaitable`, as do `futures`. Tasks then extend
futures.

## 2.6 Accessing and manually managing the event loop

### 2.6.1 Creating an event loop manually

We can create an event loop by using the `asyncio.new_event_loop` method. This
will return an event loop instance. With this, we have access to all the
low-level methods that the event loop has to offer. With the event loop we have
access to a method named `run_until_complete`, which takes a coroutine and runs
it until it finishes.

[create_event_loop_example](./create_event_loop_example.py)

### 2.6.2 Accessing the event loop

From time to time, we may need to access the currently running event loop.
asyncio exposes the `asyncio.get_running_loop` function that allows us to
get the current event loop. As an example, let's look at `call_soon`, which
will schedule a function to run on the next iteration of the event loop.

[call_soon_example](./call_soon_example.py)
