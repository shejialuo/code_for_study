# Chapter 4 Basic datatypes

## 4.1 Numeric types

For most purposes, the types of numbers we need to be concerned
with are:

+ *Integral numbers*: These are whole numbers, positive and negative
  + `Int`: This type is a fixed-precision integer.
  + `Integral`: This type is also for integers, but his one supports
  arbitrarily large (or small) numbers.
+ *Fractional*: These are not integers. `Fractional` values include
the following four types:
  + `Float`: This is type used for single-precision floating point
  numbers.
  + `Double`: This is a double-precision floating point number type.
  + `Rational`: This is a factional number that represents a ratio
  of two integers. The value `1 / 2 :: Rational` will be a value
  carrying two `Integer` values.
  + `Scientific`: This is a space efficient and almost arbitrary
  precision scientific number type.

These numeric datatypes all have instances of a typeclass called
`Num`.

## 4.2 Names and variables

In Haskell there are seven categories of entities that have names:
functions, term-level variables, data constructors, type variables,
type constructors, typeclassses, and modules. Term-level variables
and data constructors exists in your terms. *Term level* is where your
values live and is the code that executes when your program is running.
At the *type level*, which is used during the static analysis and
verification of your program.
