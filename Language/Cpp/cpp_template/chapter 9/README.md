# Chapter 9 Using Templates in Practice

Template code is a little different from ordinary code. In some ways templates lie
somewhere between macros and ordinary (nontemplate) declarations.

## 9.1 The Inclusion Model

There are several ways to organize template source code. This section presents the
most popular approach: the inclusion model.

### 9.1.1 Linker Errors

Most C and C++ programmers organize their nontemplate code largely as follows:

+ Classes and other type declarations are entirely placed in *header files*.
+ For global variables and functions, only a declaration is put in a header file,
and the definition goes into a file compiled as its own translation unit.

With theses conventions in mind, a common error about which beginning template
programmers complain is illustrated by the following little program.

[myfirst.hpp](./myfirst.hpp)

The implementation of the function is placed in a CPP file:

[myfirst.cpp](./myfirst.cpp)

Finally, we use the template in another CPP file, into which our template declaration
is `#included`:

[myfirstmain.cpp](./myfirstmain.cpp)

A C++ compiler will most likely accept this program without any problems, but the
linker will probably report an error, implying that there is no definition of the
function `printTypeof()`.

The reason for this error is that the definition of the function template `printTypeof()`
has not been instantiated. In order for a template to be instantiated, the compiler
must know which definition should be instantiated and for what template arguments it
should instantiated. Unfortunately, in the previous example, these two pieces of information
are in files that are *compiled separately*. Therefore, when our compiler sees the call
to `printTypeof()` but has no definition in sight to instantiate the function for `double`,
it just assumes that such a definition is provided elsewhere and creates a reference to that
definition.

### 9.1.2 Templates in Header Files

The common solution to be the previous problem is to use the same approach that we would take
with macros or with inline functions: We include the definitions of a template in the header
file that declares that tempaltes.

That is, instead of providing a file `myfirst.cpp`, we rewrite `myfirst.hpp` so that it contains
all template declarations and template definitions:

```c++
#ifndef MYFIRST_HPP
#define MYFIRST_HPP

#include <iostream>
#include <typeinfo>

template<typename T>
void printTypeof(T const&);

template<typename T>
void printTypeof(T const& x) {
  std::cout << typeid(x).name() << '\n';
}

#endif
```

This way of organizing templates is called the *inclusion model*. With this in place, you should
find that our program now correctly compiles, links, and executes.

However, this approach has considerably increased the cost of including the header file `myfirst.hpp`.
In this example, the cost is not the result of the size of the template definition itself but the
result of the fact that must also include the headers used by the definition of our template.

This is a real problem in practice because it considerably increases the time needed by the compiler
to compile significant programs. We will therefore examine some possible ways to approach this
problem:

+ precompiled headers.
+ explicit template instantiation.

Despite this build-time issue, we do recommend following this inclusion model to organize your
templates when possible until a better mechanism becomes available.

Another (more subtle) observation about the inclusion approach is that noninline function
templates are distinct from inline functions and macros in an important way: They are expanded
at the call site. Instead, when they are instantiated, they create a new copy of a function.
Because this is an automatic process, a compiler could end up creating two copies in two
different files, and some linker could issue errors when they find two distinct definitions
for the same function.

## 9.2 Templates and inline

Declaring functions to be inline is a common tool to improve the running time of programs.
The `inline` specifier was meant to be a hint for the implementation that inline substitution
of the function body at the point of the call is preferred over the usual function call mechanism.

However, an implementation may ignore the hint. Hence, the only guaranteed effect of
`inline` is to allow a function definition to appear multiple times in a program
(usually because it appears in a header file that is included in multiple places).

Like inline functions, function templates can be defined in multiple translation units.
This is usually achieved by placing the definition in a header file that is included by
multiple CPP files.

This doesn't mean that function templates use inline substitutions by default. It is entirely
up to the compiler whether and when inline substitution of a function template body
at the point of call is preferred over the usual function call mechanism.
