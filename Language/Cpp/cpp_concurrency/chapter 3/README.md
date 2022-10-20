# Chapter 3 Sharing data between threads

## 3.1 Problems with sharing data between threads

*If all shared data is read-only, there's no problem*, because the data
read by one thread is unaffected by whether or not another thread is
reading the same data. However, if data is shared between threads, and one
or more threads start modifying the data, there's a lot of potential for trouble.

### 3.1.1 Race conditions

The C++ Standard also defines the term *data race* to mean the specific type of
race condition that arises because of concurrent modification to a single object;
data races cause the dreaded *undefined behavior*.

### 3.1.2 Avoiding problematic race conditions

There are several ways to deal with problematic race conditions:

+ Wrap your data structure with a protection mechanism.
+ Modify the design of your data structure.
+ Handle the updates to the data structure as a *transaction*.

The most basic mechanism for protecting shared data provide by the C++ standard
is the *mutex*.

## 3.2 Protecting shared data with mutexes

### 3.2.1 Using mutexes in C++

In C++, you create a mutex by constructing an instance of `std::mutex`, lock it
with a call to the member function `lock()`, and unlock it with a call to the
member function `unlock()`. The C++ Library provides the `std::lock_guard` class
template, which implements that RAII for a mutex; it locks the supplied mutex
on construction and unlocks it on destruction.

```c++
#include <list>
#include <mutex>
#include <algorithm>

std::list<int> some_list;
std::mutex some_mutex;

void add_to_list(int new_value) {
  std::lock_guard<std::mutex> guard{some_mutex};
  some_list.push_back(new_value);
}

bool list_contains(int value_to_find) {
  std::lock_guard<std::mutex> guard{some_mutex};
  return std::find(some_list.cbegin(), some_list.cend(), value_to_find) != some_list.cend();
}
```

However, in the majority of cases it's common to group the mutex and the protected data
together in a class rather than use global variables. This is a standard application of
object-oriented design rules. If all the member functions of the class lock the mutex
before accessing any other data members and unlock it when done, the data is nicely
protected from all comers.

Well, that's not *quite* true. If one of the member functions returns a pointer or reference
to the protected data, the it doesn't matter that the member functions all lock the mutex in
a nice orderly fashion, because you've just blown a big hole in the protection:

+ *Any code that has access to that pointer or reference can now access the protected data*.

### 3.2.2 Structuring code for protecting shared data

Protecting data with a mutex is not quite as easy as just slapping a `std::lock_guard` object
in every member function; one stray pointer or reference, and all that protection is for nothing.
At one level, checking for stray pointers or references is easy; However, things won't such easy.
It's also important to check that *they don't pass such pointers or references to their callers*.

```c++
class some_data {
  int a;
  std::string b;
public:
  void do_something();
}

class data_wrapper {
private:
  some_data data;
  std::mutex m;
public:
  template<typename Function>
  void process_data(Function func) {
    std::lock_guard<std::mutex> l(m);
    func(data);
  }
}

some_data* unprotected;

void malicious_function(some_data &protected_data) {
  unprotected = &protected_data;
}

data_wrapper x;

void foo() {
  x.process_data(malicious_function);
  unprotected->do_something();
}

```

In this example, the code in `process_data` looks harmless enough, nicely protected with
`std::lock_guard`, but the call to the uer-supplied function `func` means that `foo` can
pass in `malicious_function` to bypass the protection and then call `do_something()` without
the mutex being locked.

We can get an important note:

+ Don't pass pointers and references to protected data outside the scope of the lock, whether
by returning them from a function, storing them in externally visible memory, or passing them
as arguments to user-supplied functions.s

### 3.2.3 Spotting race conditions inherent interfaces

Consider a stack data structure like the `std::stack` container adapter. If you change `top()`
so that it returns a copy rather than a reference and protect the internal data with a mutex,
this interface is still inherently subject to race conditions.

```c++
template<typename T, typename Container=std::deque<T>>
class stack {
public:
  explicit stack(const Container&);
  explicit stack(Container&& = Container());
  template <typename Alloc> explicit stack(const Alloc&);
  template <typename Alloc> explicit stack(const Container&, const Alloc&);
  template <typename Alloc> explicit stack(const Container&&, const Alloc&);
  template <typename Alloc> explicit stack(stack&&, const Alloc&);

  bool empty() const;
  size_t size() const;
  T& top();
  T const& top() const;
  void push(T const&);
  void push(T&&);
  void pop();
  void swap(stack&&);
}
```

The problem here is that the results of `empty()` and `size()` can't be relied on.
Although they might be correct at the time of the call, once they've returned,
other threads *are free to access the stack* and might `push()` new elements onto or
`pop()` the existing ones off of the stack before the thread that called `empty()`
or `size()` could use that information.

In particular, if the `stack` instance is *not shared*, it's safe to check for `empty()`
and then call `top()` to access the top element if the stack is not empty. However, when
the object is *shared*, this call sequence is no longer safe.

```c++
stack<int> s;
if (!s.empty()) {
  int const value = s.top();
  s.pop();
  do_something(value);
}
```

This problem happens as a consequence of the design of the interface, so the solution
is to change the interface.

### 3.2.4 Deadlock: the problem and a solution

The common advice for avoiding deadlock is to always lock the two mutexes in the same
order: if you always lock mutex A before mutex B, you'll never deadlock. Sometimes this
is straightforward, because the mutexes are serving different purposes, but other times
it's not so simple, such as when the mutexes are each protecting a separate instance
of the same class. Consider an operation that exchanges data between two instances of
the same class; in order to ensure that the data is exchanged correctly, without being
affected by concurrent modifications, the mutexes on both instances must be locked.

The C++ Standard Library has a cure for this in the form of `std::lock` a function that can
lock two or more mutexes at once without risk of deadlock.

```c++
class some_big_object;
void swap(some_big_object &lhs, some_big_object *rhs);

class X {
private:
  some_big_object some_detail;
  std::mutex m;
public:
  X(some_big_object const &sd): some_detail(sd) {}
  friend void swap(X &lhs, X&rhs) {
    if(&lhs == &rhs) return;
    std::lock(lhs.m, rhs.m);
    std::lock_guard<std::mutex> lock_a(lhs.m,std::adopt_lock);
    std::lock_guard<std::mutex> lock_b(rhs.m,std::adopt_lock);
    swap(lhs.some_detail,rhs.some_detail);
  }
}
```

First, the arguments are checked to ensure they are different instances, because
attempting to acquire a lock on `std::mutex` when you already hold it
is *undefined behavior*. Then, the call to `std::lock()` locks the two mutexes,
and two `std::lock_guard` instances are constructed, one for each mutex. The
`std::adopt_lock` parameter is supplied in addition to the mutex to indicate
to the `std::lock_guard` objects that the mutexes are already locked, and
they should just adopt the ownership of the existing lock on the mutex rather
than attempt to lock the mutex in the constructor.

### 3.2.5 Further guidelines for avoiding deadlock

Deadlock doesn't just occur with locks, although that's the most frequent cause;
you can create deadlock with two threads and no locks just by having each thread
call `join()` on the `std::thread` object for the other.

#### Avoid nested locks

The first idea is the simplest: don't acquire a lock if you already hold one. If
you need to acquire multiple locks, do it as a single action with `std::lock` in
order to acquire them without deadlock.

#### Avoid calling user-supplied code while holding a lock

Because the code is user supplied, you have no idea what it could do; it could
do anything, including acquiring a lock.

#### Acquire locks in a fixed order

If you absolutely must acquire two or more locks, and you can't acquire them as
a single operation with `std::lock`, the next-best thing is to acquire them in
the *same order* in every thread.

#### Use a lock hierarchy

A lock hierarchy can provide a means of checking that the convention is adhered to
at runtime. The idea is that you divide your application into layers and identify
all the mutexes that may be locked in any given layer. When code tries to lock a
mutex, it isn't permitted to lock that mutex if it already holds a lock from a
lower layer. For example,

```c++
hierarchical_mutex high_level_mutex(10000);
hierarchical_mutex low_level_mutex(5000);

int do_low_level_stuff();

int low_level_func() {
  std::lock_guard<hierarchical_mutex> lk(low_level_mutex);
  return do_low_level_stuff();
}

void high_level_stuff(int some_param);

void high_level_func() {
  std::lock_guard<hierarchical_mutex> lk(high_level_mutex);
  high_level_stuff(low_level_stuff());
}

void thread_a() {
  high_level_func();
}

hierarchical_mutex other_mutex(100);
void do_other_stuff();

void other_stuff() {
  high_level_func();
  do_other_stuff();
}

void thread_b() {
  std::lock_guard<hierarchical_mutex> lk(other_mutex);
  other_stuff();
}

```

`thread_a()` abides by the rules, so it runs fine. On the other hand, `thread_b()`
disregards the rules and therefore will fail at runtime.

This example also demonstrates another point, the use of the `std::lock_guard<>` template
with a user-defined mutex type. `hierarchical_mutex` is not part of the standard but is
easy to write; a simple implementation is below. Even though it's a user-defined type, it
can be used with `std::lock_guard<>` because it implements the three member functions
required to satisfy the mutex concept: `lock()`, `unlock()` and `try_lock()`.

```c++
class hierarchical_mutex {
  std::mutex internal_mutex;
  unsigned long const hierarchy_value;
  unsigned previous_hierarchy_value;
  static thread_local unsigned long this_thread_hierarchy_value;

  void check_for_hierarchy_violation() {
    if (this_thread_hierarchy_value <= hierarchy_value) {
      throw std::logic_error("mutex hierarchy violated");
    }
  }

  void update_hierarchy_value() {
    previous_hierarchy_value = this_thread_hierarchy_value;
    this_thread_hierarchy_value = hierarchy_value;
  }
public:
  explicit hierarchical_mutex(unsigned long value):
    hierarchy_value(value),
    previous_hierarchy_value(0) {}
  void lock() {
    check_for_hierarchy_violation();
    internal_mutex.lock();
    update_hierarchy_value();
  }
  void unlock() {
    this_thread_hierarchy_value = previous_hierarchy_value;
    internal_mutex.unlock();
  }
  bool try_lock() {
    check_for_hierarchy_violation();
    if (!internal.mutex.try_lock())
      return false;
    update_hierarchy_value();
    return true;
  }
}
```

The key here is the use of the `thread_local` value representing the hierarchy value
for the current thread: `this_thread_hierarchy_value`. It's initialized to the
maximum value, so initially any mutex can be locked.

### 3.2.6 Flexible locking with std::unique_lock

`std::unique_lock` provides a bit more flexibility than `std::lock_guard` by relaxing
the invariants; a `std::unique_lock` instance doesn't always own the mutex that it's
associated with.

Just as you can pass `std::adopt_lock` as a second argument to the constructor to have
the lock object manage the lock on a mutex, you can also pass `std::defer_lock` as the
second argument to indicate that the mutex should remain unlocked on construction.
The lock can then be acquired later by calling `lock()` on the `std::unique_lock` object
or by passing the `std::unique_lock` object itself to `std::lock()`.

```c++
class some_big_object;
void swap(some_big_object &lhs, some_big_object &rhs);
class X {
private:
  some_big_object some_detail;
  std::mutex m;
public:
  X(some_big_object const &sd): some_detail(sd) {}

  friend void swap(X &lhs, X&rhs) {
    if (&lhs==&rhs) return;
    std::unique_lock<std::mutex> lock_a(lhs.m, std::defer_lock);
    std::unique_lock<std::mutex> lock_b(lhs.m, std::defer_lock);
    std::lock(lock_a, lock_b);
    swap(lhs.some_detail, rhs.some_detail);
  }
}
```

The actual work for `std::lock()` is just to update a flag inside the `std::unique_lock`
instance to indicate whether the mutex is currently owned by that instance. This flag is
necessary in order to ensure that `unlock()` is called correctly in the destructor.

This flag has to be stored somewhere. Therefore, the size of a `std::unique_lock` object
is typically larger than that of a `std::lock_guard` object, and there's also a slight
performance penalty when using `std::unique_lock` over `std::lock_guard` because the flag
has to be updated or checked.

### 3.2.7 Transferring mutex ownership between scopes

Because `std::unique_lock` instances don't have to own their associated mutexes, the ownership
of a mutex can be transferred between instances by *moving* the instances around, which is
*moveable* but not *copyable*.

One possible use is to allow a function to lock a mutex and transfer ownership of that lock
to the caller, so the caller can then perform additional actions under the protection of the
same lock.

```c++
std::unique_lock<std::mutex> get_lock() {
  extern std::mutex some_mutex;
  std::unique_lock<std::mutex> lk(some_mutex);
  prepare_data();
  return lk;
}

void process_data() {
  std::unique_lock<std::mutex> lk(get_lock());
  do_something();
}
```

The flexibility of `std::unique_lock()` also allows instances to relinquish their locks
before they're destroyed. You can do this with the `unlock()` member function, just like
for a mutex.

## 3.3 Alternative facilities for protecting shared data

One particular extreme case is where the shared data needs protection only from concurrent
access while it's being initialized, but after that no explicit synchronization is required.

### 3.3.1 Protecting shared data during initialization

Suppose you have a shared resource that's so expensive to construct that tou want to do so
only if it's actually required.

```c++
std::shared_ptr<some_resource> resource_ptr;
void foo() {
  if (!resource_ptr) {
    resource_ptr.reset(new some_resource);
  }
  resource_ptr->do_something();
}
```

If the shared resource itself is safe for concurrent access, the only part that needs protecting
when converting this to multithreaded code is the initialization. Well, it is easy to write the
following code. However, each thread must wait on the mutex in order to check whether the resource
has already been initialized.

```c++
std::shared_ptr<some_resource> resource_ptr;
std::mutex resource_mutex;
void foo() {
  std::unique_lock<std::mutex> lk(resource_mutex);
  if (!resource_ptr) {
    resource_ptr.reset(new some_resource);
  }
  lk.unlock();
  resource_ptr->do_something();
}
```

Many people have tried to come up with a better way of doing this, including the infamous
*Double-Checked Locking* pattern: the pointer is first read without acquiring the lock,
and the lock is acquired only if the pointer is `NULL`.

```c++
void undefined_behavior_with_double_checked_locking() {
  if (!resource_ptr) {
    std::lock_guard<std::mutex> lk(resource_mutex);
    if (!resource_ptr) {
      resource_ptr.reset(new some_resource);
    }
  }
  resource_ptr->do_something();
}
```

This pattern is infamous for a reason: it has the potential for nasty race conditions,
because the read outside the lock isn't synchronized with the write done by another
thread inside the lock.

The C++ Standard Library provides `std::once_flag` and `std::call_once` to handle this
situation. Rather than locking a mutex and explicitly checking the pointer, every
thread can just use `std::call_once`, safe in knowledge that the pointer will have been
initialized by some thread by the time `std::call_once` returns.

```c++
std::shared_ptr<some_resource> resource_ptr;
std::once_flag resource_flag;

void init_resource() {
  resource_ptr.reset(new_resource);
}

void foo() {
  std::call_once(resource_flag, init_resource);
  resource_ptr->do_something();
}
```

We look at the another example:

```c++
class X {
private:
  connection_info connection_details;
  connection_handle connection;
  std::once_flag connection_init_flag;

  void open_connection() {
    connection = connection_manager.open(connection_details);
  }
public:
  X(connection_info const &connection_details_): connection_detail(connection_details) {}
  void send_data(data_packet const& data) {
    std::call_once(connection_init_flag, &X::open_connection, this);
  }
  data_packet receive_data() {
    std::call_once(connection_init_flag, &X::open_connection, this);
    return connection.receive_data();
  }
}
```

One scenario where there's a potential race condition over initialization is that of a
local variable defined with `static`. The initialization of such a variable is defined
to occur the first time control passes through its declaration; for multiple threads
calling the function, this means there's the potential for a race condition to define
first. In C++11 this problem is solved: the initialization is defined to happen on exactly
on the thread, and no other threads will proceed until that initialization is complete.

### 3.3.2 Protecting rarely updated data structures

Consider a table used to store a cache of DNS entries for resolving domain names to their
corresponding IP address. We need a *reader-writer* mutex, because it allows for two
different kinds of usage: exclusive access by a single "writer" thread or shared.

The C++17 there is a new class `std::shared_mutex` for this kind of synchronization.

### 3.3.3 Recursive locking

With `std::mutex`, it's an error for a thread to try to lock a mutex it already owns,
and attempting to do so will result in *undefined behavior*. However, in some circumstances it
would be desirable for a thread to reacquire the same mutex several times without having first
released it. For this purpose, the C++ Standard Library provides `std::recursive_mutex`. It
works just like `std::mutex`, except that you can acquire multiple locks on a single instance
from the same thread.
