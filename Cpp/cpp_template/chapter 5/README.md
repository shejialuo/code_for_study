# Chapter 5

## Zero Initialization

For fundamental types such as `int`, `double`, or pointer types, there is no default
constructor that initializes them with a useful default value.

```c++
void foo() {
  int x;       // x has undefined value
  int *ptr;    // ptr points to anywhere (instead of nowhere)
}
```

Now if you write templates and want to have variables of a template type initialized
by a default value, you have the problem that a simple definition doesn't do this
for built-in types:

```c++
template <typename T>
void foo() {
  T x;     // x has undefined value if T is built-in type
}
```

For this reason, it is possible to call explicitly a default constructor for built-in
types that initializes them.

```c++
template <typename T>
void foo() {
  T x1{};
  T x2 = T(); // Before C++11
}
```

To ensure that a member of a class template, for which the type is parameterized,
gets initialized, you can define a default constructor that uses a braced
initializer to initialize the member:

```c++
template <typename T>
class MyClass {
private:
  T x1;
  T x2;
public:
  MyClass(): x1{},x2() {}
}
```

Since C++11, you can also provide a default initialization for a non-static
member, so that the following is also possible.

```c++
template <typename T>
class MyClass {
private:
  T x{};
}
```

## Using this->

For class templates with base classes that depend on template parameters, using
a name `x` by itself is not always equivalent to `this->x`, even though a number
`x` is inherited. For example:

```c++
template<typename T>
class Base {
public:
  void bar();
};

template<typename T>
class Derived: Base<T> {
public:
  void foinline std::shared_ptr<spdlog::logger> create(std::string logger_name, SinkArgs &&... sink_args)
{
    return default_factory::create<Sink>(std::move(logger_name), std::forward<SinkArgs>(sink_args)...);
}

## Generic Lambdas and Member Templates

Note that generic lambdas, introduced with C++14, are shortcuts for member
templates. A simple lambda computing the "sum" of two arguments of arbitrary
types:

```c++
[](auto x, auto y) {
  return x + y;
}
```

It is a shortcut for a default-constructed object of the following class:

```c++
class CompilerSpecificName {
public:
  CompilerSpecificName();
  template<typename T1, typename T2>
  auto operator()(T1 x, T2 y) const {
    return x + y;
  }
}
```

## Variable Templates

Since C++14, variables also can be parameterized by a specific type. Such
a thing is called a *variable template*. (Like Haskell kind)

```c++
template <typename T>
constexpr T pi{3.1415926535897932285};
```

To use a variable template, you have to specify its type:

```c++
std::cout << pi<double> << '\n';
std::cout << pi<float> << '\n';
```
