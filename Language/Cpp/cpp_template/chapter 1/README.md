# Chapter 1 Function Templates

## 1.1 A First Look at Function Templates

### 1.1.1 Defining the template

The following is a function template that returns the maximum of two values:

[max1.hpp](./max1.hpp)

### 1.1.2 Using the Template

The following program shows how to use the `max()` function template:

[max1.cpp](./max1.cpp)

Templates aren't compiled into single entities that can handle any type. Instead, different
entities are generated from the template for every type for which the template is used.

The process of replacing template parameters by concrete types is called *instantiation*.
It results in an *instance* of a template.

### 1.1.3 Two-Phase Translation

Template are "compiled" into two phases:

+ Without instantiation at definition time, the template
code itself checked for correctness ignoring the template
parameters. This includes:
  + Syntax errors.
  + Using unknown names that don't depend on template parameters.
  + Static assertions that don't depend on template parameters.
+ At *instantiation time*, the template code is checked again to
ensure all code is valid.

#### Compiling and Linking

Two-phase translation leads to an important problem in the handling of templates in
practice: When a function template is used in a way that triggers its instantiation,
a compiler will need to see that template's definition. This breaks the usual compile
and link distinction for ordinary functions.

## 1.2 Template Argument Deduction

Automatic type conversion are limited during type deduction:

+ When declaring call parameters by reference, even trivial
conversions do not apply to type deduction.
+ When declaring call parameters by value, only
trivial conversions that *decay* are supported:
  + `const` or `volatile` are ignored.
  + references convert to the referenced type.
  + raw arrays or functions convert to pointer type.

For example:

```c++
template<typename T>
T max(T a, T b);

int i = 17;
int const c = 42;
max(i, c)    // T is deduced as int
max(c, c)    // T is deduced as int
int &ir = i;
max(i, ir)   // T is deduced as int
```

### Type Deduction for Default Arguments

Note also that type deduction does not work for default call arguments. For example:

```c++
template<typename T>
void f(T = "");

f(1); // OK: deduced T to be int
f(); // ERROR: cannot deduce T
```

To support this case, you also have to declare a default argument for the template
parameter.

```c++
template<typename T = std::string>
void f(T = "");

f(); // OK
```

## 1.3 Multiple Template Parameters

+ Template parameters, which are declared in angle brackets before the function template name:

  ```c++
  template<typename T>
  ```

+ Call parameters, which are declared in parentheses after the function template name:

  ```c++
  T max(T a, T b)
  ```

You may have as many template parameters as you like.

```c++
template<typename T1, typename T2>
T1 max(T1 a, T2 b) {
  return b < a ? a : b;
}
```

This raises a problem. If you use one of the parameter types as return type, the argument
for th other parameter might get converted to this type.

C++ provides different ways to deal with this problem:

+ Introduce a third template parameter for the return type.
+ Let the compiler find out the return type.
+ Declare the return type to be the "common type" of the two parameter types.

### 1.3.1 Template Parameters for Return Types

[return_type.hpp](./return_type.hpp)
[return_type.cpp](./return_type.cpp)

### 1.3.2 Deducing the Return Type

[max_auto.hpp](./max_auto.hpp)

[max_decltype.hpp](./max_decltype.hpp)

### 1.3.3 Return Type as Common Type

[max_common.hpp](./max_common.hpp)

## 1.4 Default Template Arguments

You can also define values for template parameters. These values are called
*default template arguments* and can be used with any kind of template.

[max_default.hpp](./max_default.hpp)

## 1.5 Overloading Function Templates

Like ordinary functions, function templates can be overloaded. That is, you can
have different definitions with the same function name so that when that name
is used in a function call, a C++ compiler must decide which one of the various
candidates to call.

[max2.cpp](./max2.cpp)

+ All other factors being equal, the overload resolution process prefers the nontemplate.
+ If the template can generate a function with a better match, the template is selected.
+ Automatic type conversion is not considered for deduced template arguments but is considered
for ordinary function parameters

A useful example would be to overload the maximum template for pointers and ordinary C-strings:

[max3val.cpp](./max3val.cpp)

This is only one example of code that might behave differently than expected as a result of
detailed overload resolution rules. In addition, ensure that all overloaded versions of a function
are declared before the function is called.

[max4.cpp](./max4.cpp)

## 1.7 Summary

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
