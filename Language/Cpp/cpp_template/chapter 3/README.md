# Chapter 3 Nontype Template Parameters

For function and class templates, template parameters don't have to be types.
They can also be ordinary values.

## 3.1 Nontype Class Template Parameters

[stack_notype.hpp](./stack_notype.hpp)

[stack_notype.cpp](./stack_notype.cpp)

## 3.2 Nontype Function Template Parameters

You can also define nontype parameters for function templates.

[addvalue.hpp](./addvalue.hpp)

## 3.3 Restrictions for Nontype Template Parameters

Note that nontype template parameters carry some restrictions. In general, they
can be only constant literal values, pointers to objects/functions/members,
lvalue references to objects or functions, or `std::nullptr_t`.

Floating-point numbers can class-type objects are not allowed as nontype template
parameters.

```c++
template<double VAT> // ERROR: floating-point
double process(double v) {
  return v * VAT;
}

template<std::string name> // Error: class-type objects
class MyClass{};
```

When passing template arguments to pointers or references, the objects must not
be string literals, temporaries, or data members and other subobjects.

+ In C++11, the objects also had to have external linkage.
+ In C++14, the objects also had to have external linkage or internal linkage.

Thus, the following is not possible:

```c++
template<char const* name>
class MyClass {};

MyClass<"hello"> x; //Error: string literal "hello" not allowed.
```

## 3.4 Template Parameter Type auto

Since C++17, you can define a nontype template parameter to generically accept any
type that is allowed for a nontype parameter. Using this feature, we can provide
an even more generic stack class with fixed size.

[stackauto.hpp](./stackauto.hpp)
[stackauto.cpp](./stackauto.cpp)
