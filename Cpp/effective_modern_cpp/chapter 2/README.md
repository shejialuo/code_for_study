# Chapter 2

## Item 5

+ `auto` variable must be initialized, are generally immune to type mismatches that
can lead to portability or efficiency problems, can ease the process of refactoring
and typically require less typing than variables with explicitly specified types.
+ `auto`-typed variables are subject to the pitfalls in Item 6.

## Item 6

+ "Invisible" proxy types can cause `auto` to deduce the "wrong" type for an
initializing expression.
+ The explicitly type initializer idiom forces `auto` to deduce the type you want
it to have.
