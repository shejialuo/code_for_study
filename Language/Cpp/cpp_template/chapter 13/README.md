# Chapter 13 Names in Templates

When a C++ compiler encounters a name, it must "look it up" to
identify the entity being referred.

## 13.1 Name Taxonomy

There are two major naming concepts:

1. A name is a *qualified name* if the scope to which it belongs is
explicitly denoted by using a scope-resolution operation (`::`) or
a member access operator (`.` or `->`).
2. A name is a *dependent name* if it depends in some way on a template
parameter.

## 13.2 Looking Up Names

Qualified names are looked up in the scope implied by the qualifying construct.
If that scope is a class, then base classes may also be searched. However,
enclosing scopes are not considered when looking up qualified names.

```c++
int x;

class B {
public:
  int i;
};

class D : public B {};

void f(D * pd) {
  pd->i = 3; // Finds B::i
  D::x = 2; // Error: does not find ::x in the closing scope.
}
```

In contrast, unqualified names are typically looked up in successively
more enclosing scopes. This is called *ordinary lookup*.

```c++
extern int count;

int lookup_example(int count) {
  if (count < 0) {
    int count = 1;
    lookup_example(count);
  }
  return count + ::count;
}
```

A more recent twist to the lookup of unqualified names is that
*argument-dependent lookup (ADL)*. Before proceeding with the
details of ADL, let's motivate the mechanism with our `max()`
template:

```c++
template<typename T>
T max (T a, T b) {
  return b < a ? a : b;
}
```

Suppose now that we need to apply this template to a type defined
in another namespace:

```c++
namespace BigMath {
class BigNumber {
  ...
};

bool operator<(BigNumber const &, BigNumber const &);
}

using BigMath::BigNumber;

void g(BigNumber const & a, BigNumber const & b) {
  ...
  BigNumber x = ::max(a, b);
  ...
}
```

The problem here is that the `max()` template is unaware of the
`BigMath` namespace, but ordinary lookup would not find the
`operator <` applicable to values of type `BigNumber`. Without some
special rules, this greatly reduces the applicability of templates in
the presence of C++ namespaces. ADL is the answer

### 13.2.1 Argument-Dependent Lookup


