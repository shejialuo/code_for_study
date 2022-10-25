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
