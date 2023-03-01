# 1. Fixes and Deprecation

## Removed Things

### Removing auto_ptr

C++98 added `auto_ptr` as a way to support basic RAII features for raw
pointers. However, due to the lack of move semantics in the language,
this smart pointer could be easily misused and cause runtime errors.

Here's an example where `auto_ptr` might cause a crash:

```c++
void doSomething(std::auto_ptr<int> myPtr) {
  *myPtr = 11;
}

void AutoPtrTest() {
  std::auto_ptr<int> myTest(new int(10));
  doSomething(myTest);
  *myTest = 12;
}
```

`doSomething()` takes the pointer by value, but since it's not a shared pointer,
it takes the unique ownership of the managed object. Later when the function is done,
the copy of the pointer goes out of scope, and the object is deleted.

### Removing the register keyword

The `register` keyword is being removed.

### Removing Deprecated operator++(bool)

This operator has been deprecated for a very long time. This time it is removed
from the language.

### Removing Deprecated Exception Specifications

In C++17, exception specification will be part of the type system.

## Fixes

### New auto rules for direct-list-initialization

Since C++11 there's been a strange problem where `auto x{1};` is deduced as
`std::initializer_list<int>`. With the new standard, we can fix this so that
it will deduce `int`. To make this happen, we need to understand two ways
of initialization: copy and direct.

```c++
auto x = foo(); // copy-initialization
auto x{foo()};  // direct-initialization, initializes an initializer_list (until c++17)

int x = foo(); // copy-initialization
int x{foo()};  // direct-initialization
```

For the direct initialization, C++17 introduces new rules:

+ For a braced-init-list with only a single element, auto deduction will deduce
from that entry;
+ For a braced-init-list with more than one element, auto deduction will be ill-formed.

For example:

```c++
auto x1 = {1, 2}; // std::initializer_list<int>
auto x2 = {3}; // std::initializer_list<int>;
auto x3{3};    // int
```

### static_assert With no Message

This feature adds a new overload for `static_assert`. It enables you to have the
condition inside `static_assert` without passing the message.
