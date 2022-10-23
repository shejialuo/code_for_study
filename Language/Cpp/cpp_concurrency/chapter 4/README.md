# Chapter 4 Synchronizing concurrent operations

## 4.1 Waiting for an event or other condition

+ Use spin to wait.
+ Use `std::this_thread:sleep_for`.
+ Use condition variable.

```c++
bool flag;
std::mutex m;

void wait_for_flag() {
  std::unique_lock<std::mutex> lk(m);
  while (!flag) {
    lk.unlock();
    std::this_thread:sleep_for(std::chromo::milliseconds(100));
    lk.lock();
  }
}
```

### 4.4.1 Waiting for a condition with condition variables

The C++ Standard Library provides not noe but *two* implementations of a
condition variable: `std::condition_variable` and `std::condition_variable_any`.
Both of these are declared in the `<condition_variable>` library header. In both
cases, they need to work with a mutex in order to provide appropriate synchronization;
the former is *limited* to working with `std::mutex`, whereas the latter can work
with anything that meets some minimal criterial for being mutex-like.

```c++
std::mutex mut;
std::queue<data_chunk> data_queue;
std::conditional_variable data_cond;

void data_preparation_thread() {
  while(more_data_to_prepare()) {
    data_chunk const data = prepare_data();
    std::lock_guard<std::mutex> lk(mut);
    data_queue.push(data);
    data_cond.notify_one();
  }
}

void data_processing_thread() {
  while (true) {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk, []() {
      return !data.queue.empty();
    });
    data_chunk data = data_queue.front();
    data_queue.pop();
    lk.unlock();
    process(data);
    if (is_last_chunk(data)) break;
  }
}
```

The implementation of `wait()` checks the condition and returns if it's satisfied. If
the condition isn't satisfied, `wait()` unlocks the mutex and puts the thread in a
blocked or waiting state.

### 4.1.2 Building a thread-safe queue with condition variables

[thread_safe_queue](./thread_safe_queue.cpp)

## 4.2 Waiting for one-off events with futures

The C++ Standard Library models one-off event with something called a *future*. If a thread
needs to wait for a specific one-off event, it somehow obtains a future representing this
event.

There are two sorts of futures in the C++ Standard Library, implemented as two class templates
declared in the `<future>` library header: *unique futures* (`std::future<>`) and
*shared futures* (`std::shared_future<>`). These are modeled after `std::unique_ptr` and
`std::shared_ptr`. An instance of `std::future` is the one and only instance that refers
to its associated event, whereas multiple instances of `std::shared_future` may refer to
the same event.

### 4.2.1 Returning values from background tasks

You use `std::async` to start an *asynchronous task* for which you don't need the result
right away. Rather than giving you back a `std::thread` object to wait on, `std::async`
returns a `std::future` object, which will eventually hold the return value of the
function. When you need the value, you just call `get()` on the future, and the thread
blocks until the future is *ready* and then returns the value.

```c++
#include <future>
#include <iostream>

int find_the_answer_to_ltuae();
void do_other_stuff();
int main() {
  std::future<int> the_answer = std::async(find_the_answer_to_ltuae);
  do_other_stuff();
  std::cout << "The answer is" << the_answer.get() << std::endl;
}
```

`std::async` allows you to pass additional arguments to the function by adding extra
arguments to the call, in the same way that `std::thread` does.

By default, it's up to the implementation where `std::async` starts a new thread, or
whether the task runs synchronously when the future is waited for. In most cases this
is what you want, but you can specify which to use with an additional parameter to
`std::async` before the function to call. This parameter is of the type `std::launch`,
and can either be `std::launch::deferred` to indicate that the function call is to
be deferred until either `wait()` or `get()` is called on the future, `std::launch::async`
to indicate that the function must be run on its own thread.

### 4.2.2 Associating a task with a future

`std::packaged_task<>` ties a future to a function or callable object. When the
`std::packaged_task<>` object is invoked, it calls the associated function or callable
object and makes the future *ready*, with the return value stored as the associated
data.

The template parameter for the `std::packaged_task<>` class template is a function
signature. When you construct an instance of `std::packaged_task`, you must pass
in a function or callable object that can accept the specified parameters and that
returns type convertible to the specified return type.

The return type of the specified function signature identifies the type of the
`std::future<>` returned from the `get_future()` member function.

The `std::packaged_task` object is thus a callable object, and it can be wrapped
in a `std::function` object, passed to a `std::thread` as the thread function,
passed to another function that requires a callable object, or even invoked directly.

When the `std::packaged_task` is invoked as a function object, the arguments supplied
to the function call operator are passed on to the contained function, and the return
value is stored as the asynchronous result in the `std::future` obtained from
`get_future()`.

We give an example here.

```c++
#include <deque>
#include <mutex>
#include <future>
#include <thread>
#include <utility>

std::mutex m;
std::deque<std::package_task<void()>> tasks;

bool gui_shutdown_message_received();
void get_and_process_gui_message();

void gui_thread() {
  while(!gui_shutdown_message_received()) {
    get_and_process_gui_message();
    std::package_task<void()> task;
    {
      std::lock_guard<std::mutex> lk(m);
      if (tasks.empty()) continue;
      task = std::move(tasks.front());
      tasks.pop_front();
    }
    task();
  }
}

std::thread gui_bg_thread(gui_thread);

template<typename Func>
std::future<void> post_task_for_gui_thread(Func f) {
  std::package_task<void> task(f);
  std::future<void> res = task.get_future();
  std::lock_guard<std::mutex> lk(m);
  tasks.push_back(std::move(task));
  return res;
}

```

### 4.2.3 Making promises

`std::promise<T>` provides a means of setting a value, which can be later read
through an associated `std::future<T>` object. You can obtain the `std::future`
object associated with the a given `std::promise` by calling the `get_future()`
member function, just like with `std::packaged_task`. When the value of promise
is set (using the `set_value()` member function), the future becomes *ready*
and can be used to retrieve the stored value.

### 4.2.4 Saving an exception for the future

Consider the following short snippet of code. If you pass in `-1` to the
`square_root()` function, it throws an exception, and this gets seen by the caller.

```c++
double square_root(double x) {
  if (x < 0) {
    throw std::out_of_range("x<0");
  }
  return sqrt(x);
}
```

Now suppose that instead of just invoking `square_root` from the current thread:

```c++
double y = square_root(-1);
```

You run the call as an asynchronous call:

```c++
std::future<double> f = std::async(square_root, -1);
double y = f.get();
```

If the function call invoked as part of `std::async` throws an exception, that
exception is stored in the future in place of a stored value, the future becomes
ready, and a call to `get()` rethrows that stored exception.

`std::promise` provides the same facility, with an explicit function call. If
you wish to store an exception rather than a value, you call the `set_exception()`
member function rather than `set_value()`

### 4.2.5 Waiting from multiple threads

Although `std::future` handles all the synchronization necessary to transfer data
from on thread to another, calls to the member functions of a particular
`std::future` instances are not synchronized with each other. If you access a
single `std::future` object from multiple threads without additional synchronization,
you have a *data race* and undefined behavior. This is by design.

If you want your concurrent code requires that multiple threads can wait for the same
event, you should use `std::shared_future` allows exactly that.

Now with `std::shared_future`, member functions on an individual object are still
unsynchronized, so to avoid data races when accessing a single object from multiple
threads, you must protect accesses with a lock.
