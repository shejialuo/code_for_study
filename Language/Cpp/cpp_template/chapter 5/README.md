# Chapter 5 Tricky Basics

## 5.1 Keyword typename

The keyword `typename` was introduced during the standardization of C++ to clarify
that an identifier inside a template is a type. In general, `typename` has to be
used whenever a name that depends on a template parameter is a type.

[printcoll.hpp](./printcoll.hpp)

## 5.2 Zero Initialization

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

## 5.3 Using this->

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
  void foo() {
    bar();
  }
};
```

## 5.4 Templates for Raw Arrays and String Literals

When passing raw arrays or string literals to templates, some care has to be
taken. First, if the template parameters are declared as references, the
arguments don't decay.

You can provide templates that specifically deal with raw arrays or string literals.

[lessarray.hpp](./lessarray.hpp)

If you only want to provide a function template for string literals, you can do
this as follows:

[lessstring.hpp](./lessstring.hpp)

Now that you can and sometimes have to overload or partially specialize for arrays
of unknown bounds.

[arrays.hpp](./arrays.hpp)

## 5.5 Member Templates

Class members can also be templates. This is possible for both nested classes and
member functions.

[stack5del.h](./stack5decl.h)

[stack5assign.cpp](./stack5assign.cpp)

[stack6decl.hpp](./stack6assign.cpp)

[stack6assign.cpp](./stack6assign.cpp)

### 5.5.1 Generic Lambdas and Member Templates

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

## 5.6 Variable Templates

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

### Variable Templates for Data Members

A useful application of variable templates is to define variables that represent
members of class templates. For example, if a class template is defined as
follows:

```c++
template<typename T>
class MyClass {
public:
  static constexpr int max = 1000;
};
```

You can define different values for different specializations of `MyClass<>`, then
you can define

```c++
template<typename T>
int myMax = MyClass<T>::max;
```

So the application can just write `myMax<std::string>` instead of `MyClass<std::string>::max`.

### Type Traits Suffix_v

Since C++17, the standard library uses the technique of variable templates to define
shortcuts for all type traits in the standard library that yield a (Boolean) value.
