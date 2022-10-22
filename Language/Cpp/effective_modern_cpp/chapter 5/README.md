# Chapter 5

## Item 23

It's useful to approach `std::move` and `std::forward` in terms of what they *don't* do.
`std::move` doesn't move anything. `std::forward` doesn't forward anything. At runtime,
neither does anything at all. They generate no executable code. Not a single byte.

`std::move` and `std::forward` are merely function templates that perform cats.

To make the story more concrete, here's a sample implementation of `std::move` in
C++11.

```c++
template<typename T>
typename remove_reference<T>::type&&
move(T&& param) {
  using ReturnType = typename remove_reference<T>::type&&;
  return static_cast<ReturnType>(param);
}
```

The `&&` part of the function's return type implies that `std::move` returns an rvalue
reference, but if the type `T` happens to be an lavalue reference, `T&&` would become
and lvalue reference. TO prevent this from happening, the type trait `std::remove_reference`
is applied to `T`, thus ensuring that `&&` is applied to a type that isn't a reference.
That guarantees that `std::move` truly returns an rvalue reference,

Rvalues are only *usually* candidates for moving. Suppose you're writing a class
representing annotations. The class's constructor takes a `std::string` parameter
comprising the annotation, and it copies the parameter to a data member.

```c++
class Annotation {
public:
  explicit Annotation(std::string text);
};
```

But `Annotation`'s constructor needs only to read `text`'s value. It doesn't need to
modify it.

```c++
class Annotation {
public:
  explicit Annotation(const std::string text);
};
```

To avoid paying for a copy operation when copying `text` into a data member, you apply
`std::move` to `text`:

```c++
class Annotation {
public:
  explicit Annotation(const std::string text) : value(std::move(text))
};
```

However, the `text` is not moved into `value`, it's *copied*. Sure, `text` is cast to an
rvalue by `std::move`, but `text` is declared to be a `const std::string`, so the result
of the cast is an rvalue `const std::string`.

Consider the effect that has when compilers have to determine which `std::string` constructor
to call. There are two possibilities:

```c++
class string {
public:
  string(const string& rhs);
  string(string&& rhs);
}
```

Thus the rvalue can't be passed to `std::string`'s move constructor, because move constructor
takes an rvalue reference to a non-const `std::string`. The rvalue can, however, be passed
to the copy constructor, because an lvalue-reference-to-const is permitted to bind to a `const`
rvalue.

There are two points:

+ Don't declare objects `const` if you want to be able to move from them.
+ `std::move` not only does't actually move anything.

The story for `std::forward` is similar to that for `std::move`, but whereas `std::move`
*unconditionally* casts its argument to an rvalue, `std::forward` does it only under
certain conditions. `std::forward` is a *conditional* cast. The most common scenario
is a function template taking a universal reference parameter that is to be passed
to another function:

```c++
void process(const Widget& lvalArg);
void process(Widget&& rvalArg);

template<typename T>
void logAndProcess(T&& param) {
  auto now = std::chrono::system_clock::now();

  makeLogEntry("Calling 'process'", now);
  process(std::forward<T>(param));
}
```

Consider two calls to `logAndProcess`, one with an lvalue, the other with an rvalue:

```c++
Widget w;

logAndProcess(w); // call with lvalue
logAndProcess(std::move(w)); // call with rvalue
```

`param`, like all function parameters, is an lvalue. Every call to `process` inside
`logAndProcess` will thus want to invoke the lvalue overloaded for `process`. To
prevent this, we need mechanism for `param` to be cast to an rvalue if and only
of the argument which `param` was initialized. This is what `std::forward` does.
That' why `std::forward` is a conditional cast: it casts to an rvalue only if
its argument was initialized with an rvalue.

Things to remember:

+ `std::move` performs an unconditional cast to an rvalue. In end of itself, it
doesn't move anything.
+ `std::forward` casts its argument to an rvalue only if that argument is bound to
an rvalue.
+ Neither `std::move` nor `std::forward` do anything at runtime.

## Item 24

To declare an rvalue reference to some type $T$, you write `T&&`. You may think
that it's an rvalue reference.

```c++
void f(Widget&& param); // rvalue reference
Widget&& var1 = Widget(); // rvalue reference
auto&& var2 = var1; // not rvalue reference

template<typename T>
void f(std::vector<T>&& param); // rvalue reference

template<typename T>
void f(T&& param); // not rvalue reference
```

The `T&&` has two different meanings. One is a rvalue reference. The other meaning
is *either* rvalue reference or lvalue reference. They can behave as if they were
lvalue references. Their dual nature permits them to bind to rvalues as well as
lvalues. Furthermore, they can bind to `const` or non-`const` objects, to `volatile`
or non-`volatile` objects.

There are two contexts:

```c++
template<typename T>
void f(T&& param);

auto &&var2 = var1;
```

What these context have in common is the presence of *type deduction*. For example:

```c++
template<typename T>
void f(T&& param);

Widget w;
f(w); // lvalue passed to f. param's type is Widget&
f(std::move(w)) // rvalue passed to f; param's type is Widget&&
```

For a reference to be universal, type deduction is necessary, but it's not sufficient.
The *form* of the reference declaration must also be correct, and that form is quite
constrained. It must be precisely `T&&`.

Even the simple presence of a `const` qualifier is enough to disqualify a reference
from being universal.

```c++
template<typename T>
void (const T&& param); // param is an rvalue reference.
```

Things to remember:

+ If a function template parameter has type `T&&` for a deduced type `T`, or
if an object is declared using `auto&&`, the parameter or object is a universal
reference.
+ If the form of the type declaration isn't precisely `type&&`, or if type
deduction not occur, `type&&` denotes an rvalue reference.
+ Universal references correspond to rvalue references if they're initialized with
rvalues. They correspond to lvalue references if they're initialized with lvalues.

## Item 25

In short, rvalue references should be *unconditionally cast* to rvalues when forwarding
them to other functions, because they're always bound to rvalues, and universal references
should be *conditionally cast* to rvalues when forwarding them, because they're only
*sometimes* bound to rvalues.

The idea of using `std::move` with universal references, because that can have the effect
of unexpectedly modifying lvalues.

```c++
class Widget {
public:
  template<typename T>
  void setName(T&& newName) {name = std::move(newName);}
private:
  std::string name;
  std::shared_ptr<SomeDataStructure> p;

std::string getWidgetName();

Widget w;

auto n = getWidgetName();

w.setName(n); // n's value now unknown
}
```

If you're in a function that returns *by value*, and you're retuning an object bound to
an rvalue reference or a universal reference, you'll want to apply `std::move` or
`std::forward` when you return the reference.

```c++
Matrix operator+(Matrix&& lhs, const Matrix& rhs) {
  lhs += rhs;
  return std::move(lhs);
}
```

If `Matrix` does not support moving, casting it to an rvalue won't hurt, because the rvalue
will simply be copied by `Matrix`'s copy constructor.

+ Apply `std::move` to rvalue references and `std::forward` to universal references
the last time each is used.
+ Do the same thing for rvalue references and universal references being returned
from functions that return by value.
+ Never apply `std::move` or `std::forward` to local objects if they would otherwise
be eligible for the return value optimization.

## Item 26

Suppose you need to write a function that takes a name as a parameter, logs the current date
and time, then adds the name to the global data structure.

```c++
std::multiset<std::string> names;

void logAndAdd(const std::string& name) {
  auto now = std::chrono::system_clock::now();

  log(now, "logAndAdd");

  names.emplace(name);
}
```

This isn't unreasonable code, but it's not as efficient as it could be.

```c++
std::string petName("Darla");

logAndAdd(petName); // pass lvalue std::string
logAndAdd(std::string("Persephone")); // pass rvalue std::string
logAndAdd("Patty Dog"); // pass string literal
```

In the first call, `logAndAdd`'s parameter `name` is bound to the variable `petName`.
Within `logAndAdd`, `name` is ultimately passed to `names.emplace`. Because `name`
is an lvalue, it is copied into `names`. There's no way to avoid that copy.

In the second call, the parameter `name` is bound to an rvalue. `name` itself is an
lvalue. so it's copied into `names`, but we recognize its value could be moved into `names`.

In the third call, the parameter `name` is again bound to an rvalue, but this time it's to
a temporary `std::string` that's implicitly created from `Patty Dog`.

We can eliminate the inefficiencies in the second and third calls by rewriting `logAndAdd`
to take a universal reference.

```c++
template<typename T>
void logAndAdd(T&& name) {
  auto now = std::chrono::system_clock::now();
  log(now, "logAndAdd");
  names.emplace(std::forward<T>(name));
}
```

## Item 27

+ Universal reference parameters often have efficiency advantages, but they typically
have usability disadvantages.

## Item 28

The encoding mechanism is simple. When an lvalue is passed an argument, `T` is deduced
to be an lvalue reference. When an rvalue is passed, `T` is reduce to be a non-reference.

```c++
template<typename T>
void func(T&& param);

Widget widgetFactory();

Widget w;
func(w); // call with lvalue; T deduced to be Widget&
func(widgetFactory) // call func with rvalue; T deduced to be Widget
```

In both calls to `func`, different types are deduced for the template parameter `T`. This
determines whether universal references become rvalue references or lvalue references, and
it's also the mechanism through which `std::forward` does its work.

We must note that references to references are illegal in C++. But consider what happens when
an lvalue is passed to a function template taking a universal reference:

```c++
template<typename T>
void func(T&& param);

func(w);
```

If we take the type deduced for `T` and use it to instantiate the template, we get this:

```c++
void(Widget& && param);
```

A reference to reference! So compiler uses *reference collapsing*. Yes, you are forbidden from
declaring references to references, but *compilers* may produce them in particular contexts.
When compilers generate references to references, reference collapsing dictates what happens next.

There are two kinds of references, so there are four possible reference-reference combinations.
If a reference to reference arises in a context where this is permitted, then references collapse
to a single reference according to this rule:

+ If either reference is an lvalue reference, the result is an lvalue reference. Otherwise, the
result is an rvalue reference.

Here's how `std::forward` can be implemented to do that:

```c++
template<typename T>
T&& forward(typename remove_reference<T>::type &param) {
  return static_cast<T&&>(param);
}
```

Suppose that the argument passed to `f` is an lvalue of type `Widget`. `T` will be deduced as
`Widget&`, and the call to `std::forward` will yield this:

```c++
Widget& && forward(typename remove_reference<Widget&>::type &param) {
  return static_cast<Widget& &&>(param);
}
```

And for compiles it will do *reference collapsing*:

```c++
Widget& forward(Widget& param) {
  return static_cast<Widget&>(param);
}
```

If the argument passed to `f` is an rvalue of type `Widget`. `T` will be reduced as `Wdiget`.

And we ill get

```c++
Widget&& forward(Widget& param) {
  return static_cast<Widget&&>(param);
}
```

The third context is the generation and use of `typedef`s and alias declarations.
