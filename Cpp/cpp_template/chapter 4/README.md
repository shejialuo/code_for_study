# Chapter 4

## Operator sizeof...

C++11 also introduces a new form of the `sizeof` operator
for variadic templates: `sizeof...`.

```c++
template<typename T, typename... Types>
void print(T firstArg, Types... args) {
  std::cout << sizeof...(Types); << '\n'; // print number of remaining types.
  std::cout << sizeof...(args); << '\n'; // print number of remaining args.
}
```

## Fold Expressions

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

## Application of Variadic Templates

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

## Variadic Expressions

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

## Variadic Indices

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
