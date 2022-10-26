# Chapter 6

## 6.1 Perfect Forwarding

Suppose you want to write generic code that forwards the basic property of passed
arguments:

+ Modifyable object should be forwarded so that they still can be modified.
+ Constant objects should be forwarded as read-only objects.
+ Movable objects should be forwarded as movable objects.

To achieve this functionality without templates, we have to program all these cases.
For example, to forward a call of `f()` to a corresponding function `g()`:

[move1.cpp](./move1.cpp)

Here, we see different implementations of `f()` forwarding its argument to `g()`:

```c++
void f(X& val) {
  g(val);
}

void f(X const& val) {
  g(val);
}

void f(X&& val) {
  g(std::move(val));
}
```

Note that the code for movable objects differs from the other code: It needs a
`std::move()` because according to language rules, move semantics is not passed
through. Although `val` in the third `f()` is declared as rvalue reference its
value category when used as expression is a non-constant lvalue.

If we want to combine all three cases in generic code, we have a problem:

```c++
template<typename T>
void f(T&& val) {
  g(val);
}
```

This works for the first two cases, but not for the third case when movable objects
are passed.

C++11 for this reason introduces special rules for *perfect forwarding* parameters.

```c++
template<typename T>
void f(T&& val) {
  g(std::forward<T>(val));
}
```

[move2.cpp](./move2.cpp)

## 6.2 Special Member Function Templates

Member function templates can also be used as special member functions, including a
constructor, which, however, might lead to surprising behavior.

Consider the following example:

[special_member_template1.cpp](./special_member_template1.cpp)

Now let's replace the two string constructors with one generic constructor perfect
forwarding the passed argument to the member `name`:

[special_member_template2.cpp](./special_member_template2.cpp)

Construction with passed string works fine, as expected:

```c++
std::string s = "sname";
Person p1{s};
Person p2{"tmp"};
```

Note how the construction of `p2` does not create a temporary string int this case:
The parameter `STR` is deduced to be of type `char const[4]`. Applying `std::forward<STR>`
to the pointer parameter has not much of an effect, and the `name` member is thus
constructed from a null-terminated string.

But when we attempt to call the copy constructor, we get an error. When initializing a new
`Person` by a moveable object still works fine:

```c++
Person p3{p1}; // error
Person p4{std::move([1])};
```

Note that also copying a constant `Person` works fine:

```c++
Person const p2c{"ctmp"};
Person p3c{p2c};
```

The problem is that, according to the overload resolution rules of C++, for a nonconstant
lvalue `Person p` the member template

```c++
template<typename STR>
Person(STR&& n)
```

is a better match than the copy constructor:

```c++
Person(Person const& p)
```

`STR` is just substituted with `Person&`, while for the copy constructor a conversion to
`const` is necessary.

You might think about solving this by also providing a nonconstant copy constructor. However,
this is only a partial solution because for objects of a derived class, the member template
is still a better match. What you really want to is to disable the member template for
the case that the passed argument is a `Person` or an expression that can be converted
to a `Person`. That can be done by using `std::enable_if<>`.

## 6.3 Disable Templates with enable_if\<\>

`std::enable_if<>` is a type trait that evaluates a give compile-time expression passed
as its first template argument and behave as follows:

+ If the expression yields `true`, its type member `type` yields a type:
  + The type is `void` if no second template argument is passed.
  + Otherwise, the type is the second template argument type.
+ If the expression yields `false`, the member `type` is not defined. Due to
a template feature called SFINAE (substitution failure is not an error), this has the
effect that the function template with the `enable_if` expression is ignored.

```c++
template<typename T>
typename std::enable_if<(sizeof(T) > 4)>::type
foo() {}

template<typename T>
typename std::enable_if<(sizeof(T) > 4), T>::type
foo() { return T{}};
```

Note that having the `enable_if<>` expression in the middle of a declaration is
pretty clumsy. For this reason, the common way to use `std::enable_if<>` is to
use an additional function template argument with a default value:

```c++
template<typename T,
         typename = std::enable_if_t<(sizeof(T) > 4)>>
void foo() {}
```

If that is still too clumsy, and you want to make the requirement/constraint
more explicit, you can define your own name for it using an alias template:

```c++
template<typename T>
using EnableIfSizeGreater4 = std::enable_if_t<(sizeof(T) > 4)>;

template<typename T,
         typename = EnableIfSizeGreater4<T>>
void foo() {}
```

## 6.4 Using enable_if/</>

We can use `enable_if<>` to solve our problem with the constructor template. The
problem we have to solve is to disable the declaration of the template constructor

```c++
template<typename STR>
Person(STR&& n);
```

If the passed argument `STR` has the right type (i.e., is a `std::string` or a type
convertible to `std::string`). For this we use another standard type trait,
`std::is_convertible<FROM,TO>`. With C++17, the corresponding declaration looks
as follows:

```c++
template<typename STR,
         typename = std::enable_if_t<std::is_convertible_v<STR, std::string>>>
person(STR&& n);
```

Thus, the whole class `Person` should look like as follows:

[special_member_template3.cpp](./special_member_template3.cpp)

Note that there is an alternative to using `std::is_convertible<>` because
it requires that the types are implicitly convertible. By using `std::is_constructible<>`,
we allow explicit conversions to be used for the initialization. However, the order of
the arguments is the opposite in this case:

```c++
template<typename T>
using EnableIfString = std::enable_if_t<std::is_constructible_v<std::string, T>>;
```

### Disabling Special Member Functions

Normally we can't use `enable_if<>` to disable the predefined copy/move constructors
and/or assignment operators. The reason is that member function templates never count
as special member functions and are ignored.

```c++
class C{
public:
  template<typename T>
  C (T const&) {
    std::cout << "tmpl copy constructor\n";
  }
};

C x;
C y{x}; // still uses the predefined copy constructor (not the member template)
```

Deleting the predefined copy constructor is no solution, because then the trial
to copy a `C` results in an error.

## 6.5 Using Concepts to Simplify enable_if/</> Expressions

Even when using alias templates, the `enable_if` syntax is pretty clumsy, because
it uses a workaround: To get the desired effect, we add an additional template
parameter and "abuse" that parameter to provide a specific requirement for the
function template to be available at all.

In principle, we just need a language feature that allows us to formulate
requirements or constants for a function in a way that causes the function to be
ignored if the requirements/constraints are not met. For example, like Haskell.

```hs
f :: (a -> Ord) => a -> a
f = undefined
```

This is an application of the long-awaited language feature *concepts*, which allow
us to formulate requirements/conditions for templates with its own simple syntax.

With concepts, as their use is proposed, we simple have to write the following:

```c++
template<typename STR>
requires std::is_convertible_v<STR, std::string>;
Person(STR&& n) : name{std::forward<STR>(n)} {}
```
