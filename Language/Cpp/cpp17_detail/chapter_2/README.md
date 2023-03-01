# 2. Language Clarification

## Stricter Expression Evaluation Order

Until C++17 the language hasn't specified any evaluation order for function
parameters. For example, that's why in C++14 `make_unique` is not just a
syntactic sugar, but it guarantees memory safety.

Let's have a look at the following example:

```c++
foo(make_unique<T>(), otherFunction());

foo(unique_ptr<T>(new T), otherFunction());
```

In C++14, the above explicit code, we know that `new T` is guaranteed to happen
before `unique_ptr` construction, but that's all. For example, `new T` might
happen first, then `otherFunction()`, and then `unique_ptr` constructor. When
`otherFunction()` throws, the `new T` generated a leak.

C++17 addresses this issue. In an expression `f(a, b, c)`. The order of evaluation
of `a`, `b`, `c` is still unspecified, but any parameter is fully evaluated before
the next one is started. It's especially crucial for complex expression like:

```c++
f(a(x), b, c(y));
```

## Guaranteed Copy Elision

Copy Elision is a popular optimization that avoids creating unnecessary temporary
objects. For example:

[copyElision](./copyElision.cpp)

In the above call, compiler would use Return Value Optimization.

Currently, the standard allows eliding in cases like:

+ When a temporary object is used to initialized another object.
+ When a variable that is about to go out of scope is returned or thrown
+ When an exception is caught by value

## Dynamic Memory Allocation for Over-Aligned Data

When you work with SIMD or when you have some other memory layout requirements, you
might need to align objects specifically.

```c++
void* operator new(size_t, align_val_t);
void* operator new[](size_t, align_val_t);
void operator delete(void*, align_val_t);
void operator delete[](void*, align_val_t);
void operator delete(void*, size_t, align_val_t);
void operator delete[](void*, size_t, align_val_t);
```
