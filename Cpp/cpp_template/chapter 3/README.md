# Chapter 3

## Restrictions for Nontype Template Parameters

In general, they can be only constant integral values,
pointers to objects/functions/members, lvalue references
to objects or functions, or `std::nullptr_t`.

When passing template arguments to pointers or references,
the objects must not be string literals, temporaries, or
data members. Because these restrictions were relaxed with
each and every C++ version before C++17, additional
constraints apply:

+ In C++11, the objects also had to have external linkage.
+ In C++14, the objects also had to have external linkage
or internal linkage.

```c++
template<const char* name>
class MyClass {};

extern const char s03[] = "hi";
const char s11[] = "hi";

int main() {
  MyClass<s03> m03; // OK all versions
  MyClass<s11> m11; // OK since C++11
  static const char s17[] = "hi";
  MyClass<s17> m17; // OK since C++14
}
```
