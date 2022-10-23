# Chapter 5 The C++ memory model and operations and atomic types

## 5.1 Memory model basics

### 5.1.1 Objects and memory locations

There are four important things:

+ Every variable is an object, including those that are members of other objects.
+ Every object occupies *at least* memory location.
+ Variables of fundamental type such as `int` or `char` are *exactly one* memory
+ Adjacent bit fields are part of the same memory location;

### 5.1.2 Objects, memory locations, and concurrency

If two threads access *separate* memory locations, there's no problem: everything
works fine. On the other hand, if two threads access the *same* memory location,
then you have to be careful.

In order to avoid the race condition, there has to be an enforced ordering between
the accesses in the two threads. One way is mutex, another way is to use the
synchronization properties of *atomic* operations either on the same or other
memory locations to enforce an ordering between the accesses in the two threads.

### 5.1.3 Modification orders

Every object in a C++ program has a defined *modification order* composed of all
the writes to that object from all threads in the program, starting with the
object's initialization.

## 5.2 Atomic operations and types in C++

An *atomic operation* is an indivisible operation. You can't observe such an operation
half-done from nay thread in the system.

### 5.2.1 The standard atomic types

The standard *atomic types* can be found in the `<atomic>` header. All operations on such
types are atomic, and only operations on these types are atomic in the sense of the language
definition. The standard atomic types themselves might use such emulation: they all have
an `is_lock_free()` member function, which allows the user to determine whether operations
on a given type are done directly with atomic instructions or done by using a lock internal
to the compiler and library.

The only type that doesn't provide an `is_lock_free()` member function is `std::atomic_flag`.
This type is a really simple Boolean flag, and operations on this type are *required* to be
lock-free.

The remaining atomic types are all accessed through specializations of the `std::atomic<>`
class template an are a bit more full-featured but may not be lock-free.

The standard atomic types are not copyable or assignable in the conventional sense, in that
they have no copy constructor or copy assignment operators. They do support assignment from
and implicit conversion to the corresponding built-in types as well as direct `load()` and
`store()` member functions, `exchange()`, `compare_exchange_weak()`, and
`compare_exchange_strong()`. They also support the compound assignment assignment operators
where appropriate: `+=`, `-=`, `*=`, `|=`, and so on, and the integral types and
`std::atomic<>` specializations for pointers support `++` and `--`. These operators also
have corresponding named member functions with the same functionality: `fetch_add()`,
`fetch_or()`, and so on.

The `std::atomic<>` class template isn't just a set of specializations, though. It does
have a primary template that can be used to create an atomic variant of a user-defined
type. Because it's a generic class template, the operations are limited to `load()`,
`store()`, `exchange()`, `compare_exchange_weak()`, and `compare_exchange_strong()`.

Each of the operations on the atomic types has an optional memory-ordering argument
that can be used to specify the required memory-ordering semantics.

### 5.2.2 Operations on std::atomic_flag

`std::atomic_flag` is the simplest standard atomic type, which represents a Boolean
flag. Objects of this type can be in one of two states: set or clear. It's deliberately
basic and is intended as a building block only.

Objects of type `std::atomic_flag` *must* be initialized with `ATOMIC_FLAG_INIT`. This
initializes the flag to a *clear* state. There's no choice in the matter; the flag
always starts clear:

```c++
std::atomic_flag f = ATOMIC_FLAG_INIT;
```

This applies wherever the object is declared and whatever scope it has. It's the only
atomic type to require such special treatment of initialization, but it's also the
only type guaranteed to be lock-free. If the `std::atomic_flag` object has static
storage duration, it's guaranteed to be statically initialized, which means that there
are no initialization-order issues.

Once you have your flag object initialized, there are only three things you can do with
it: destroy it, clear it, or set it and query the previous value. These correspond to
the destructor, the `clear()` and `test_and_set()` member functions can have a memory
order specified.

The limited feature set makes `std::atomic_flag` ideally suited to use as a spin-lock
mutex.

```c++
class spinlock_mutex {
  std::atomic_flag flag;
public:
  spinlock_mutex() : flag{ATOMIC_FLAG_INIT} {}
  void lock() {
    while(flag.test_and_set(std::memory_order_acquire));
  }
  void unlock() {
    flag.clear(std::memory_order_release);
  }
};
```

### 5.2.3 Operations on std::atomic\<bool\>

The most basic of the atomic integral types is `std::atomic<bool>`. This is a more
full-featured Boolean flag than `std::atomic_flag`, as you might expect. You can
construct it from a non-atomic `bool`, so it can be initially `true` or `false`,
and you can also assign to instances of `std::atomic<bool>` from a non-atomic `bool`.

```c++
std::atomic<bool> b{true};
b = false;
```

Rather than using the restrictive `clear()` function of `std::atomic_flag`, writes are
done by calling `store()`, although the memory-order semantics can still be specified.
Similarly, `test_and_set()` has been replaced with the more general `exchange()`
member function that allows you to replace the stored value with a new one of your
choosing and atomically retrieve the original value. `std::atomic<bool>` also supports
a plain non-modifying query of the value with an implicit conversion to plain `bool`
or with an explicit call to `load()`.

```c++
std::atomic<bool> b;
bool x = b.load(std::memory_order_acquire);
b.store(true);
x = b.exchange(false, std::memory_order_ace_rel);
```

#### Storing a new value ot not depending on the current value

This new operation is called compare/exchange, and it comes in the from of the
`compare_exchange_weak()` and `compare_exchange_strong()` member functions. The
compare/exchange operation is the cornerstone of programming with atomic types;
it compares the value of the atomic variable with a supplied expected value
and stores the supplied desired value if they're equal. If the values aren't
equal, the expected value is updated with the actual value of the atomic
variable. The return type of the compare/exchange functions is a `bool`, which
is `true` if the store was performed and `false` otherwise.

For `compare_exchange_weak()`, the store might not be successful even if the
original value was equal to the expected value, in which case the value of
the variable is unchanged and the return value of `compare_exchange_weak()`
is `false`. This is most likely to happen on machines that lack a single
compare-and-exchange instruction.

On the other hand, `compare_exchange_strong()` is guaranteed to return `false`
only if the actual value wasn't equal to the `expected` value.

### 5.2.4 The other

There are some many details. I don't think it's a good idea to write a note. Skip.

## 5.3 Synchronizing operations and enforcing ordering

Suppose you have two threads, one of which is populating a data structure to be read
by the second. In order to avoid a problematic race condition, the first thread
sets a flag to indicate that the data is ready, and the second thread doesn't read
the data until the flag is set. The following listing shows such a scenario.

```c++
#include <vector>
#include <atomic>
#include <iostream>

std::vector<int> data;
std::atomic<bool> data_ready(false);

void reader_thread() {
  while(!data_ready.load()) {
    std::this_thread::slepp(std::milliseconds(1));
  }
  std::cout << "The answer=" << data[0] << "\n";
}

void writer_thread() {
  data.push_back(42);
  data_ready = true;
}
```

The required enforced ordering comes from the operations on the `std::atomic<bool>`
variable `data_ready`; they provide the necessary ordering by virtue of the
memory model relations *happen-before* and *synchronizes-with*.

### 5.3.1 The synchronizes-with relationship

The synchronizes-with relationship is something that you can get only between
operations on atomic types. The basic idea is this: a sutiably tagged atomic
read operation on `x` that reads the value stored by either that write (`W`) or a
sequence atomic write operation on `x` by the same thread that performed the
initial write `W`, or a sequence of atomic read-modify-write operations on `x` by
any thread, where the value read by the first thread in the sequence is the value
written by `W`.

### 5.3.2 The happens-before relationship

The *happens-before* relationship is the basic building block of operation ordering
in a program; it specifies which operations see the effects of which other operations.

If the operations occur in the same statement, in general, there's no happens-before
relationship between them, because they're unordered. This is just another way of
saying that the ordering is unspecified.

```c++
#include <iostream>

void foo(int a, int b) {
  std::cout << a << "," << "b" << std::endl;
}

int get_num() {
  static int i = 0;
  return ++i;
}

int main() {
  foo(get_num(), get_num());
}
```

### 5.3.3 Memory ordering for atomic operations

There are six memory ordering options that can be applied to operations on atomic
types:

+ `memory_order_relaxed`
+ `memory_order_consume`
+ `memory_order_acquire`
+ `memory_order_release`
+ `memory_order_acq_rel`
+ `memory_order_seq_cst`

Unless you specify otherwise for a particular operation, the memory-ordering option
for all operations on atomic type is `memory_order_seq_cst`, which is the most
stringent of the available options. Although there are six ordering options, they
represent three models:

+ *Sequentially consistent*: `memory_order_seq_cst`
+ *acquire-release*: `memory_order_consume`, `memory_order_acquire`, `memory_order_release`
and `memory_order_acq_rel`.
+ *relaxed*: `memory_order_relaxed`
