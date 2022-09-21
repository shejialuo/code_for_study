# Chapter 3

## Item 7

+ Braced initialization is the most widely usable initialization syntax, it prevents
narrowing conversions, and it's immune to C++'s most vexing parse.
+ During constructor overload resolution, braced initializers are matched to
`initializer_list` parameters if at all possible, even if other constructors
offer seemly better matches.
+ An example of where the choice between parentheses and braces can make a significant
difference is creating a `vector<numeric type>` with two arguments.
+ Choosing between parentheses and braces for object creation inside templates can
be challenging.

## Item 8

+ Prefer `nullptr` to 0 and `NULL`.
+ Avoid overloading on integral and pointer types.

## Item 9

+ `typedef` doesn't support templatization, but alias declarations do.
+ C++14 offers alias templates for all the C++11 type traits transformations.

## Item 10

+ C++98-style `enum` is known as unscoped `enum`.
+ Enumerators of scoped `enum`s are visible only within the `enum`. They convert
to other types only with a cast.
+ Both scoped and unscoped `enum`s support specification of the underlying
type. The default underlying type for scoped `enum`s is `int`. Unscoped `enum`s have
no default underlying type.
+ Scoped `enum`s may always be forward-declared.

## Item 11

+ Prefer deleted functions to private undefined ones.
+ Any function may be deleted, including non-member functions and template instantiations.

## Item 12

+ Declare overriding functions `override`.

## Item 13

+ Prefer `const_iterator`s to `iterator`s.
+ In maximally generic code, prefer non-member versions of `begin`, `end`, `rbegin`.

## Item 14

+ `noexcept` is part of a function's interface, and that means that callers
may depend on it.
+ `noexcept` functions are more optimizable than non-noexcept functions.
+ `noexcept` is particularly valuable for the move operations, `swap`, memory
deallocation functions, and desctructors.
+ Most functions are exception-neutral rather than `noexcept`.

## Item 15

+ `constexpr` objects are `const` and are initialized with values known during
compilation.
+ `constexpr` functions can produce compile-time results when called with
arguments whose values are known during compilation.
+ `constexpr` objects and functions may be used in a wider range of context than
non-`constexpr` objects and functions.
+ `constexpr` is part of an object's or function's interface.
