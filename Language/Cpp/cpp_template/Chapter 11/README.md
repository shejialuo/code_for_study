# Chapter 11 Generic Libraries

Templates are most effective when used to write generic libraries and framework.

## 11.1 Callables

Many libraries include interfaces to which client code passes an entity that must be "called". In C++,
there are several types that work well for callbacks because they can both be passed as function
call arguments and can be directly called with syntax `f(...)`:

+ Pointer-to-function types
+ Class types with an overloaded `operator()`, including lambdas
+ Class types with a conversion function yielding a pointer-to-function or reference-to-function.

Collectively, these types are called *function object types*, and a value of such a type is a *function object*.

### 11.1.1 Supporting Function Objects

Let's look at how the `for_each` algorithm of the standard library is implemented.

[foreach.hpp](./foreach.cpp)

[foreach.cpp](./foreach.cpp)
