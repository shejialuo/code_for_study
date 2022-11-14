# Appendix A The One-Definition Rule

Affectionately known as the ORD, the *one-definition rule* is a cornerstone for
the well-formed structuring of C++ programs. The most common consequences of the
ODR are simple enough to remember and apply: Define non-inline functions or objects
exactly once across all files, and define classes, inline functions, and inline
variables at most once per translation unit, making sure that all definitions for
the same entity are identical.

## A.1 Translation Units

A translation unit is the result of applying the preprocessor to a file you feed to
your compiler.

Here, as far as the ODR is concerned, having the following two files:

Connections across translation unit boundaries are established by having corresponding
declarations with external linkage in two translation units.

## A.2 Declarations and Definitions

A *declaration* is a C++ construct that introduces or reintroduces a name in your program.
A declaration can also be a definition, depending on which entity it introduces and how
it introduces it:

+ *Namespaces and namespace aliases*: The declarations of namespaces and their aliases are
always also definitions.
+ *Classes, class templates, functions...*: The declaration is a definition if and only if
the declaration includes a brace-enclosed body associated with the name.
+ *Enumerations*: The declaration is a definition if and only if it includes the brace-enclosed
list of enumerators.
+ *Local variables and non-static data members*: These entities can always be treated as definitions.
+ *Global variables*: If the declaration is not directly preceded by a keyword `extern` or if it
has an initializer, the declaration of a global variable is also a definition of that variable.

## A.3 The One-Definition Rule in Detail

### A.3.1 One-per-Program Constraints

There can be at most one definition of the following items per program:

+ Non-inline functions and non-inline member functions
+ Non-inline variables
+ Non-inline static data members.

For example, a C++ program consisting of the following two translation units is invalid:

```c++
// == translation uint 1 :
int counter;

// == translation unit 2 :
int counter;
```

This rule does not apply to entities with *internal linkage* because even when two such
entities have the same name, they are considered distinct. In the same vein, entities
declared in unnamed namespace are consider distinct if they appear in distinct translation
units; in C++11 and later, such entities also have internal linkage by default, but prior
to C++11 they had external linkage by default.

```c++
// == translation unit 1 :
static int counter = 2;
namespace {
  void unique() {}
}

// == translation unit 2 :
static int counter = 0;
namespace {
  void unique() {
    ++counter;
  }
}

int main() {
  unique();
}
```

### A.3.2 One-per-Translation Unit Constraints

No entity can be defined more than once in a translation unit. So the following is
invalid in C++:

```c++
inline void f() {}
inline void f() {} // ERROR: duplicate definition
```

This is one of the main reasons for surrounding the code in header files with *guards*:

```c++
#ifndef GUARDDEMO_HPP
#define GUARDDEMO_HPP

#endif
```
