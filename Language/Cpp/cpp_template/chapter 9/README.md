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
