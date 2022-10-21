# Chapter 1

## Item 1

```c++
template<typename T>
void f(ParamType param);
f(expr)
```

### Case1: ParamType is a Reference or Pointer, but not a Universal Reference

+ If `expr`'s type is a reference, ignore the reference part.
+ Then patter-match `expr`'s type against `ParamType` to determine `T`.


### Case2: ParamType is a Universal Reference

+ If `expr` is an lvalue, both `T` and `ParamType` are deduced to be
lvalue references.
+ If `expr` is an ravalue, the "normal" rule applys

### Case3: ParamType is neither a Pointer or Reference

+ As before, if `expr`'s type is a reference, ignore the reference part.
+ Ignore `const`, `volatile` and parameter decays.

[item1.cpp](./item1.cpp)

Things to remember:

+ During template type deduction, arguments that are reference are treated as non-references.
+ When deducing types for universal reference parameters, lvalue arguments get
special treatment.
+ When deducing types are by-value parameters, `const` and/or `volatile` arguments
are treated as non-`const` and non-`volatile`.
+ During template type deduction, arguments that are array or function names decay
to pointers, unless they're used to initialize references.

## Item 2

[item2.cpp](./item2.cpp)

Things to remember:

+ `auto` type deduction is usually the same as template type deduction, but `auto`
type deduction assumes that a braced initializer represents a `std::initializer_list`,
and template type deduction doesn't.
+ `auto` in a function return type or a lambda parameter implies template type
deduction, not `auto` type deduction.

## Item 3

Things to remember:

+ `decltype` almost always yields the type of a variable or expression without
any modifications
+ For lvalue expressions of type `T` other than names, `decltype` always reports
a type of `T&`.
+ C++14 supports `decltype(auto)`, which, like `auto`, deduces from its initializer,
but it performs the type deduction using the `decltype`.
