# Chapter 4 Variadic Templates

Since C++11, templates can have parameters that accept a variable number
of template arguments. This feature allows the use of templates in places
where you have to pass an arbitrary number of arguments of arbitrary types.

## 4.1 Variadic Templates

Template parameters can be defined to accept an unbounded number of template
arguments. Templates with this ability are called *variadic templates*.

### 4.1.1 Variadic Templates by Example

For example you can use the following code to call `print()` for a variable number
of arguments of different types.

[varprint1.hpp](./varprint1.hpp)

If one or more arguments are passed, the function template is used, which by
specifying the first argument separately allows printing of the first argument before
recursively calling `print()` for the remaining arguments named `args` are a
*function parameter pack*:

`void print(T firstArg, Types... args)` using different `Types` specified by a
*template parameter pack*. To end the recursion, the nontemplate overload of
`print()` is provided, which is issued when the parameter pack is empty.

### 4.1.2 Overloading Variadic and Nonvariadic Templates

Note that you can also implement the example above as follows:

[varprint2.hpp](./varprint2.hpp)

### 4.1.3 Operator sizeof...

C++11 also introduces a new form of the `sizeof` operator
for variadic templates: `sizeof...`.

```c++
template<typename T, typename... Types>
void print(T firstArg, Types... args) {
  std::cout << sizeof...(Types) << '\n'; // print number of remaining types.
  std::cout << sizeof...(args) << '\n'; // print number of remaining args.
}
```

This might lead us to think we can skip the function for the end of the
recursion by not calling it in case there are no more arguments:

```c++
template<typename T, typename... Types>
void print(T firstArg, Types... args) {
  std::cout << "firstArg" << '\n';
  if(sizeof...(args) > 0) {
    print(args);
  }
}
```

However, this approach doesn't work because in general both branches of
all *if* statement in function templates are instantiated. Whether the
instantiated code is useful is a *run-time* decision, while the
instantiation of the call is a *compile-time* decision.

## 4.2 Fold Expressions

Since C++17, there is a feature to compute the result of
using a binary operator over *all* the arguments of a
parameter pack.

```c++
template <typename... T>
auto foldSum(T... s) {
  return (... + s);
}
```

| Fold Expression     | Evaluation |
| ------------------- | --------------------- |
| (... op pack)       | (pack1 op pack2) op pack3 |
| (pack op ...)       | pack1 op (pack2 op pack3) |
| init op ... op pack | (init op pack1) op pack2 |
| pack op ... op init | pack1 op (pack2 op init) |

[fold.cpp](./fold.cpp)

[foldtraverse.cpp](./foldtraverse.cpp)

## 4.3 Application of Variadic Templates

Variadic templates play an important role when implementing
generic libraries.

One typical application is the forwarding of a variadic number
of arguments of arbitrary type. For example:

+ Passing arguments to the constructor of a new heap object owned by
a shared pointer.
+ Passing arguments to a thread, which is started by the library.
+ Passing arguments to the constructor of a new element pushed into a
vector

```c++
// 1
auto sp = std::make_shared<std::complex<float>>(4.2, 7.7);
// 2
std::thread t(foo, 42, "Hello");
// 3
std::vector<Customer> v;
v.emplace_back("Tim", "Jovi", 1962);
```

## 4.4 Variadic Class Templates and Variadic Expressions

### 4.4.1 Variadic Expressions

```c++
template <typename... T>
void printDoubled(const T&... args) {
  print(args + args...);
}

template <typename... T>
void addOne(const T&... args) {
  print(args + 1 ...);
}

template <typename T, typename... TN>
constexpr bool isHomogeneous(T1, TN...) {
  return (std::is_same<T1,TN>::value && ...);
}
```

### 4.4.2 Variadic Indices

```c++
template <typename C, typename... Idx>
void printElems(const C& coll, Idx... idx) {
  print(coll[idx]...);
}

template <std::size_t... Idx, typename C>
void printIdx(const C& coll) {
  print(coll[Idx]...)
}
```

### 4.4.3 Variadic Class Templates

Variadic templates can also be class templates. An important example is a class
where an arbitrary number of template parameters specify the types of corresponding
members:

```c++
template<typename... Elements>
class Tuple;

Tuple<int, std::string, char> t;
```

You can also define a class that *as a type* represents a list of indices:

```c++
template<std::size_t...>
struct Indices {};
```

This can be used to define a function that calls `print()` for elements of a
`std::array` or `std::tuple` using the compile-time access with `get<>()` for
the given indices:

```c++
template<typename T, std::size_t... Idx>
void printByIdx(T t, Indices<Idx...>) {
  print(std::get<Idx>(t)...);
}
```

## Summary

+ By using parameter packs, templates can be defined for an arbitrary
number of template parameter of arbitrary type.
+ To process the parameters, you need recursion or a matching nonvariadic
function.
+ Operator `sizeof...` yields the number of arguments provided for a
parameter pack.
+ A typical application of variadic templates is forwarding an arbitrary
number of arguments of arbitrary types.
+ By using fold expressions, you can apply operators to all arguments of
a parameter pack.
