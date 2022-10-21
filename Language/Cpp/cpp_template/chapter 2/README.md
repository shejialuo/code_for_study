# Chapter 2 Class Templates

Similar to functions, classes can also be parameterized with one ore more
types. In this chapter, we use a stack as an example of a class template.

## 2.1 Implementation of Class Template stack

[stack1.hpp](./stack1.hpp)

### 2.1.1 Declaration of Class Templates

Declaring class templates is similar to declaring function templates: Before
the declaration, you have to declared one or multiple identifiers as a type
parameters. Again, `T` is usually used as an identifier.

```c++
template<typename T>
class Stack {

};
```

The type of this class is `Stack<T>`, with `T` being a template parameter. Thus,
you have to use `Stack<T>` whenever you use the type of this class in a declaration
except in cases where the template arguments can be deduced. However, inside a class
template using the class name not followed by template arguments represents the class
with its template parameters as its arguments.

For example, you have to declare you own copy constructor and assignment operator, it
typically looks like this:

```c++
template <typename T>
class Stack {
  Stack (Stack const&);
  Stack& operator=(Stack const&);
}
```

Above is formally equivalent to:

```c++
template <typename T>
class Stack {
  Stack (Stack<T> const&);
  Stack<T>& operator=(Stack<T> const&);
}
```

### 2.1.2 Implementation of Member Functions

To define a member function of a class template, you have to specify that it is a
template, and you have to use the full type qualification of the class template.

## 2.2 Use of Class Template Stack

To use an object of a class template, until C++17 you must always specify the
template arguments explicitly.

[stack1test.cpp](./stack1test.cpp)

Note that code is instantiated only for *template member functions that are called*.
For class templates, member functions are instantiated only if they are used.

## 2.3 Partial Usage of Class Templates

Template arguments only have to provide all necessary operations that are needed
(instead of that could be needed).

For example, class `Stack<>` would provide a member function `printOn()` to print
the while stack content, which calls `operator<<` for each element:

```c++
template<typename T>
class Stack {
  void printOn(std::ostream& strm) const {
    strm << elem << ' ';
  }
};
```

You can still use this class for elements that don't have `operator<<` defined:

```c++
Stack<std::pair<int, int>> ps; // std::pair<> has no operator<< defined
ps.push({4, 5});
ps.push({6, 7});
std::cout << ps.top().first << '\n';
std::cout << ps.top().second << '\n';
```

Only if you call `printOn()` for such a stack, the code will produce an error,
because it can't instantiate the call of `operator<<` for this specific element type.

### 2.3.1 Concepts

This raises the question: How do we know which operations are required for a template
to be able to get instantiated? The term *concept* is often used to denote as a set
of constraints that is repeatedly required in a template library.

## 2.4 Friends

Instead of printing the stack contents with `printOn()` it is better to implement
`operator<<` for the stack. However, as usual `operator<<` has to be implemented
as nonmember function, which the could call `printOn()` inline:

```c++
template <typename T>
class Stack {
  void printOn(std::ostream& strm) const {}
  friend std::ostream& operator<<(std::ostream& strm, Stack<T> const &s) {
    s.printOn(strm);
    return strm;
  }
};
```

However, when trying to *declare* the friend function and *define* it afterwards,
things become more complicated. In fact we have tw options:

We can implicitly declare a new function template, which must use a different template
parameter, such as `U`:

```c++
template <typename T>
class Stack {
  void printOn(std::ostream& strm) const {}
  template<typename U>
  friend std::ostream& operator<<(std::ostream& strm, Stack<U> const &s);
};
```

We can forward declare the output operator for a `Stack<T>` to be a template, which,
however, means that we first have to forward declare `Stack<T>`:

```c++
template <typename T>
class Stack;
template <typename T>
std::ostream& operator<<(std::ostream& strm, Stack<T> const &s);
```

Then, we can declare this function as friend:

```c++
template<typename T>
class Stack {
  friend std::ostream& operator<< <T>(std::ostream& strm, Stack<U> const &s);
};
```

Note the `<T>` behind the "function name". Thus, we declare a specialization of the
nonmember function template as friends. Without `<T>` we could declare a new
nontemplate function.

## 2.5 Specializations of Class Templates

You can specialize a class template for certain template arguments. Similar to the
overloading of function templates, specializing class templates allows you to optimize
implementations for certain types or to fix a misbehavior of certain types for an
instantiation of the class template. However, if you specialize a class template,
*you must also specialize all member functions*.

To specialize a class template, you have to declare the class with a leading
`template<>` and a specification of the types for which the class template is
specialized. The types are used as a template argument and must be specified directly
following the name of the class:

```c++
template<>
class Stack<std::string> {};
```

[stack2.hpp](./stack2.hpp)

## 2.6 Partial Specialization

Class templates can be partially specialized. You can provide special implementations
for particular circumstances, but some template parameters must still be defined by
the user.

[stack_partspec](./stack_partspec.hpp)

## 2.7 Default Class Template Arguments

[stack3.hpp](./stack3.hpp)

[stack3test.cpp](./stack3test.cpp)

## 2.8 Type Aliases

### Typedefs and Alias Declarations

To simply define a new name for a complete type, there are two ways
to do it:

+ By using the keyword `typedef`. We call this a `typedef` and the resulting name is
called a *typedef-name*.
+ By using the keyword `using`. This is called an *alias declaration*.

```c++
typedef Stack<int> IntStack;
```

```c++
using IntStack = Stack<int>;
```

### Alias Templates

Unlike a `typedef`, an alias declaration can be template to provide a convenient
name for a family of type. This is also available since C++11 and is called
*alias template*.

```c++
template<typename T>
using DequeStack = Stack<T, std::deque<T>>;
```

### Alias Templates for Member Types

Alias templates are especially helpful to define shortcuts for types that are
members of class templates. After

```c++
template<typename T> struct MyType {
  typedef ... iterator;
};
```

Or

```c++
template<typename T> struct MyType {
  using iterator = ...;
};
```

A definition such as

```c++
template<typename T>
using MyTypeIterator = typename MyType<T>::iterator;
```

We can then allow to use `MyTypeIterator<int> pos` instead of `typename MyType<T>::iterator pos`.

## 2.9 Class Template Argument Deduction

Since C++17, the constraint that you always have to specify the template arguments explicitly
was relaxed. Instead, you can skip defining the templates arguments explicitly, if the
constructor is able to *deduce* all template parameters.

```c++
Stack<int> intStack1;
Stack<int> intStack2 = intStack1;
Stack intStack3 = intStack1; // ok since C++17
```

By providing constructors that pass some initial arguments, you can support deduction of the
element type of a stack.

```c++
template<typename T>
class Stack {
private:
  std::vector<T> elems;
public:
  Stack () = default;
  Stack (T const& elem): elems({elem}) {}
};
```

This allows you to declare a stack as follows:

```c++
Stack intStack = 0; // Stack<int> deduced since C++17
```

### Class Template Arguments Deduction with String Literals

In principle, you can even initialize the stack with a string literal:

```c++
Stack stringStack = "bottom" // Stack<char const[7]> deduced since C++17
```

But this causes a lot of trouble: In general, when passing arguments of a
template type `T` by reference, the parameter doesn't *decay*. So maybe
we should copy by value.

### Deduction Guides

Instead of declaring the constructor to be called by value, there is a different
solution: Because handling raw pointers in containers is a source of trouble, we
should disable automatically deducing raw character pointers for container classes.

You can define specific *deduction guides* to provide additional of fix existing
class template argument deductions.

```c++
Stack(char const*) -> Stack<std::string>;
```

## 2.10 Templatized Aggregates

Aggregate classes can also be templates. For example:

```c++
template<typename T>
struct ValueWithComment {
  T value;
  std::string comment;
}
```

Since C++17, you can even define deduction guides for aggregate class templates:

```c++
ValueWithComment(char const*, char const*) -> ValueWithComment<std::string>;
```

## 2.11 Summary

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
