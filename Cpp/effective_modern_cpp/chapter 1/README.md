# Chapter 1

## Item 1

+ During template type deduction, arguments that are reference are treated as non-references.
+ When deducing types for universal reference parameters, lvalue arguments get
special treatment.
+ When deducing types are by-value parameters, `const` and/or `volatile` arguments
are treated as non-`const` and non-`volatile`.
+ During template type deduction, arguments that are array or function names decay
to pointers, unless they're used to initialize references.

## Item 2

+ `auto` type deduction is usually the same as template type deduction, but `auto`
type deduction assumes that a braced initializer represents a `std::initializer_list`,
and template type deduction doesn't.
+ `auto` in a function return type or a lambda parameter implies template type
deduction, not `auto` type deduction.

## Item 3

+ `decltype` almost always yields the type of a variable or expression without
any modifications
+ For lvalue expressions of type `T` other than names, `decltype` always reports
a type of `T&`.
+ C++14 supports `decltype(auto)`, which, like `auto`, deduces from its initializer,
but it performs the type deduction using the `decltype`.

