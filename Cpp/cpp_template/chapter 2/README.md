# Chapter 2

## Type Aliases

To simply define a new name for a complete type, there are two ways
to do it:

+ By using the keyword `typedef`.
+ By using the keyword `using`.

```c++
typedef Stack<int> IntStack;
```

```c++
using IntStack = Stack<int>;
```

### Alias Templates

```c++
template<typename T>
using DequeStack = Stack<T, std::deque<T>>;
```

This is what C++14 does.

```c++
template<typename T>
using add_const_t = typename add_const<T>::type;
```

## Summary

+ A class template is a class that is implemented with one
or more type parameters left out.
+ To use a class template, you pass the open type as template
arguments. The class template is then instantiated.
+ For class templates, only those member functions that are called
are instantiated.
+ You can specialize class templates for certain types.
+ Since C++17, class template arguments can automatically deduced
from constructors.
+ You can define aggregate class templates.
+ Call parameters of a template decay if declared to be called by value.
+ Templates can only be declared and defined in global/namespace scope
or inside class declaration.
