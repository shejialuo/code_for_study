# Chapter 2 Managing threads

## 2.1 Baisc thread management

### 2.1.1 Launching a thread

It doesn't matter what the thread is going to do or where it's launched from,
but starting a thread just using the C++ Thread Library always boils down
to constructing a `std::thread` object:

```c+
void do_some_work();
std::thread my_thread(do_some_work);
```

As with much of the C++ Standard Library, `std::thread` works with any *callable*
type, so you can pass an instance of a class with a function call operator to
the `std::thread` constructor instead:

```c++
class background_task {
public:
  void operator()() const {
    do_something();
    do_something_else();
  }
}
background_task f;
std::thread my_thread(f);
```

In this case, the supplied function object is *copied* into the storage belonging to
the newly created thread of execution and invoked from there. It's therefore essential
that *the copy behave equivalently to the original*.

Once you've started your thread, you need to explicitly decide whether to wait for it to
finish (by joining it) or leave it to run on its own (by detaching it). If you don't
decide before the `std::thread` object is destroyed, then your program is terminated.
(the `std::thread` destructor calls `std::terminate()`). It's therefore imperative that
you ensure that *the thread is correctly joined or detached*, even in the presence of
exceptions.

If you don't wait for your thread to finish, then you need to ensure that the data accessed
by the thread is valid until the thread has finished with it. It's a bad idea to create a
thread within a function that has access to the local variables.

### 2.1.2 Waiting for a thread to terminate

If you need to wait for a thread to complete, you can do this by calling `join()` on the
associated `std::thread` instance. `join()` is simple and brute force. The act of calling
`join()` also cleans up any storage associated with the thread, so the `std::thread` object
is no longer associated with the now-finished thread; it isn't associated with any thread.

### 2.1.3 Waiting in exceptional circumstances

If you''re intending to wait for the thread, you need to pick carefully the place in the
code where you call `join()`. THis means that the call to `join()` is liable to skipped if
an exception is thrown after the thread has been started but before the call to `join()`.

To avoid your application being terminated when an exception is thrown, you therefore need
to make a decision on what to do in this case.

```c++
struct func;
void f() {
  int some_local_state = 0;
  func my_func(some_local_state);
  std::thread t(my_func);
  try {
    do_something_in_current_thread();
  } catch(...) {
    t.join();
    throw;
  }
  t.join();
}
```

However, this is tedious. We should use RAII.

```c++
class thread_guard{
  std::thread& t;
public:
  explicit thread_guard(std::thread& t_): t(t_) {}
  ~thread_guard() {
    if (t.joinable()) {
      t.join();
    }
  }
  thread_guard(thread_guard const&) = delete;
  thread_guard& operator=(thread_guard const&) = delete;
};
```

### 2.1.4 Running threads in the background

Calling `detach()` on a `std::thread` object leaves the thread to run in the
background, with no direct means of communicating with it. Detached threads
run in the background; ownership and control are passed over to the C++
Runtime Library.

## 2.2 Passing arguments to a thread function

It's important to bear in mind that by default the arguments are *copied* into
internal storage, where they can be accessed by the newly created thread of
execution, even if the corresponding parameter in the function is expecting
a reference. Here's a simple example:

```c++
void f(int i, std::string const &s);
std::thread t(f, 3, "hello");
```

This creates a new thread of execution associated with `t`, which calls `f(3, "hello")`.
Here, the string literal is passed as a `char const*` and converted to a `std::string`
only in the context of the new thread. This is particularly important when the argument
supplied is a pointe to an automatic variable, as follows:

```c++
void f(int i, std::string const& s);

void oops(int some_param) {
  char buffer[1024];
  sprintf(buffer, "%i", some_param);
  std::thread t(f, 3, buffer);
  t.detach();
}
```

In this case, it's the pointer to the local variable `buffer` that's passed through to
the new thread, and there's a significant chance that the function `opps` will exit
before the buffer has been converted to a `std::string` on the new thread, thus leading
to undefined behavior. The solution is to cast to `std::string` *before* passing the
buffer to the `std::thread` constructor:

```c++
void f(int i, std::string const& s);

void oops(int some_param) {
  char buffer[1024];
  sprintf(buffer, "%i", some_param);
  std::thread t(f, 3, std::string(buffer));
  t.detach();
}
```

In this case, the problem is that you were relying on the implicit conversion
of the pointer to the buffer into the `std::string` object expected as a
function parameter, because the `std::thread` constructor copies the supplied
value as is, without converting to the expected argument type.

It is also possible to get the reverse scenario: the object is copied, and what
you wanted was a reference:

```c++
void update_data_for_widget(widget_id w, widget_data &data);
void oops_again(widget_id w) {
  widget_data data;
  std::thread t(update_data_for_widget, w, data);
  t.join();
}
```

Although `update_date_for_widget` expects the second parameter to be passed by
reference, the `std::thread` constructor doesn't know that; it's oblivious to
the types of the arguments expected by the function and *blindly* copies the
supplied values. When it calls `update_data_for_widget`, it will end up passing
a reference to the *internal copy* of `data` and not a reference to `data` itself.

In this case, if you change th thread invocation to be references in `std::ref`. It
will be ok:

```c++
std::thread t(update_data_for_widget, w, std::ref(data));
```

## 2.3 Transferring ownership of a thread

The move support in `std::thread` also allows for containers of `std::thread` objects,
if those containers are move aware. This means that you can write code like that:

```c++
void do_work(unsigned id);

void f() {
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < 20; ++i) {
    threads.push_back(std::thread(do_work, i));
  }
  std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
}
```

## 2.4 Choosing the number of threads at runtime

One feature of the C++ Standard Library that helps here is
`std::thread::hardware_concurrency()`. This function returns an indicator of the number
of threads that can truly run concurrently for a given execution of a program.

## 2.5 Identifying threads

Thread identifiers aer of type `std::thread::id` and can be retrieved in two ways. First,
the identifier for a thread can be obtained from its associated `std::thread` object
by calling the `get_id()` member function. If the `std::thread` object doesn't have an
associated thread of execution, the call to `get_id()` returns a default-constructed
`std::thread::id` object, which indicates ""not any thread`.

Alternatively, the identifier for the current thread can be obtained by calling
`std::this_thread::get_id()`.
