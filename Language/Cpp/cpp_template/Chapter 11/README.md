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

### 11.1.2 Dealing with Member Functions and Additional Parameters

One possible entity to call was not used in the previous example: member functions which doesn't
match the usual pattern *function-object(...)*.

Fortunately, since C++17, the C++ standard library provides a utility `std::invoke()` that
conveniently unifies this case with the ordinary function-call syntax cases, thereby enabling
calls to *any* callable object with a single form. The following implementation of our
`foreach()` templates uses `std::invoke()`:

[foreachinvoke.hpp](./foreachinvoke.hpp)

Here, besides the callable parameter, we also accept an arbitrary number of additional parameters.
The `foreach()` template then calls `std::invoke()` with the given callable followed by the
additional given parameters along with the referenced element. `std::invoke()` handles this as follows:

+ If the callable is a pointer to member, it uses the first additional argument as the `this` object.
+ Otherwise, all additional parameters are just passed as arguments to the callable.

[foreachinvoke.cpp](./foreachinvoke.cpp)

### 11.1.3 Wrapping Function Calls

A common application of `std::invoke()` is to wrap single function call. Now we can support move
semantics by perfect forwarding both the callable and all passed arguments:

```c++
#include <utility>
#include <functional>

template<typename Callable, typename... Args>
decltype(auto) call(Callable&& op, Args&&... args) {
  return std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...);
}
```

The other interesting aspect is how to deal with the return value of a called function to
"perfectly forward" it back to the caller. To support returning references you have to
use `decltype(auto)` instead of just `auto`:

`decltype(auto)` is a *placeholder type* that determines the type of variable, return type,
or template argument from the type of the associated expression.

If you want to temporarily store the value returned by `std::invoke()` in a variable to
return it after doing something else, you also have to declare the temporary variable with
`decltype(auto)`:

```c++
decltype(auto) ret {std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...)};
...
return ret;
```

Note that declaring `ret` with `auto&&` is not correct. As a reference, `auto&&` extends
the lifetime of the returned value until the end of its scope.

## 11.2 Other Utilities to Implement Generic Libraries

See Appendix D.

## 11.3 Perfect Forwarding Temporaries

Sometimes we have to perfectly forward data in generic code that does not come through a
parameter. In that case, we can use `auto&&` to create a variable that can be forwarded.
Assume, we have chained calls to functions `get()` and `set()` where the return value
of `get()` should be perfectly forwarded to `set()`:

```c++
template<typename T>
void foo(T x) {
  set(get(x));
}
```

Suppose further that we need to update our code to perform some operation on the intermediate
value produced by `get()`. We do this by holding the value in a variable declared with
`auto&&`:

```c++
template<typename T>
void foo(T x) {
  auto&& val = get(x);

  set(std::forward<decltype(val)>(val));
}
```

This avoid extraneous copies of the intermediate value.

## 11.4 References as Template Parameter

Although it is not common, template type parameters can become reference types. For
example:

[tmplparamref.cpp](./tmplparamref.cpp)
