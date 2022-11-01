# Chapter 8 Compile-Time Programming

## 8.1 Template Metaprogramming

Templates are instantiated at compile time. It turns out that some of the
features of C++ templates can be combined with the instantiation process
to produce a sort of primitive recursive "programming language" within
the C++ language itself.

[isprime.cpp](./isprime.cpp)

## 8.2 Computing with constexpr

C++11 introduces a new feature, `constexpr`, that greatly simplifies various
forms of compile-time computation. In particular, given proper input, a
`constexpr` function can be evaluated at compile time. While in C++11 `constexpr`
functions were introduced with stringent limitations (each `constexpr` function
definition was essentially limited to consist of a `return` statement), most
of these restrictions were removed with C++14. Of course, successfully evaluating
a `constexpr` function still requires that all computational steps be possible
and valid at compile time.

Our example to test  whether a number is a prime number could be implemented as
follows in C++11:

[isprime11.cpp](./isprime11.cpp)

With C++14, `constexpr` functions can make use of most control structures available
in general C++ code. So, instead of writing unwieldy template code or somewhat
arcane one-liners, we can now just use a plain `for` loop:

[isprime14.cpp](./isprime14.cpp)

## 8.3 Execution Path Selection with Partial Specialization

An interesting application of a compile-time test such as `isPrime()` is to use
partial specialization to select at compile time between different implementations.

For example, we can choose between different implementations depending on whether
a template argument is a prime number:

```c++
template<int SZ, bool = isPrime(SZ)>

// implementation if SZ is not a prime number:
template<int SZ>
struct Helper<SZ, false> {};

// implementation if SZ is a prime number:
template<int SZ>
struct Helper<SZ, true> {};
```

Because function template do not support partial specialization, you have to use
other mechanisms to change function implementation based on certain constraints.
Our options include the following:

+ Use classes with static functions
+ Use `std::enable_if`.
+ Use the SFINAE feature.
+ Use the compile-time `if` feature, available since C++17.

## 8.4 SFINAE

Consider the following example:

```c++
template<typename T, unsigned N>
std::size_t len(T(&)[N]) {
  return N;
}

template<typename T>
typename T::size_type len(T const& t) {
  return t.size();
}
```

Here, we define two function templates `len()` taking one generic argument:

+ The first function template declares the parameter as `T(&)[N]`, which means that
the parameter has to be an array of `N` elements of type `T`.
+ The second function template declares the parameter simply as `T`, which places no
constraints on the parameter but returns type `T::size_type`, which requires that
the passed argument type has a corresponding member `size_type`.

When passing a raw array of string literals, only the function for raw arrays matches:

```c++
int a[10];
std::cout << len(a);
std::cout << len("tmp");
```

According to its signature, the second function template also matches when substituting
`int[10]` and `char const[4]` for `T`, but those substitutions lead to potential errors
in return type `T::size_type`. The second type is therefore ignored for these calls.

## 8.5 Compile-Time if

C++17 additionally introduces a compile-time `if` statement that allows is to enable or
disable specific statements based on compile-time conditions. With the syntax
`if constexpr(...)`, the compiler uses a compile-time expression to decide whether to
apply the *then* part or the *else* part.

```c++
template<typename T, typename... Types>
void print(T const& firstArg, Types const&... args) {
  std::cout << firstArg << '\n';
  if constexpr(sizeof...(args) > 0) {
    print(args...);
  }
}
```

Note that `if constexpr` can be used in any function, not only in templates. We only
need a compile-time expression that yields a Boolean value.

```c++
template<typename T, std::size_t SZ>
void foo (std::array<T,SZ> const& coll)
{
  if constexpr(!isPrime(SZ)) {}
```
