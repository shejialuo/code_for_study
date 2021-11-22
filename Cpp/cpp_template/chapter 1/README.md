# Chapter 1

## Two-Phase Translation

Template are "compiled" into two phases:

+ Without instantiation at definition time, the template
code itself checked for correctness ignoring the template
parameters. This includes:
  + Syntax errors.
  + Using unknown names that don't depend on template parameters.
  + Static assertions that don't depend on template parameters.
+ At *instantiation time*, the template code is checked again to
ensure all code is valid.

## Template Argument Deduction

Automatic type conversion are limited during type deduction:

+ When declaring call parameters by reference, even trivial
conversions do not apply to type deduction.
+ When declaring call parameters by value, only
trivial conversions that *decay* are supported:
  + `const` or `volatile` are ignored.
  + references convert to the referenced type.
  + raw arrays or functions convert to pointer type.

## Summary

+ Function templates define a family of functions for different
template arguments.
+ When you pass arguments to function parameters depending on
template parameters, function templates deduce the template
parameters to be instantiated for the corresponding parameter
types.
+ You can explicitly qualify the leading template parameters.
+ You can define default arguments for template parameters.
+ You can overload function templates.
+ When overloading function templates with other function
templates, you should ensure that only one of them matches for
any call.
+ Ensure the compiler sees all overloaded versions of function
templates before you call them.
