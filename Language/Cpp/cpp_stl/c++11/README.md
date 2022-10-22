# New Language Feature

## Move Semantics and Rvalue References

### Overloading Rules for Rvalue and Lvalue References

+ If you implement only `void foo(X&)` without `void foo(X&&)`, `foo()` can
be called for lvalues but not for rvalues.
+ If you implement `void foo(const X&)` without `void foo(X&&)`, `foo()` can
be called for lvalues and for rvalues.
+ If you implement `void foo(X&); void foo(X&&)` or `void foo(const X&); void foo(X&&)`,
you can distinguish between dealing with rvalues and lvalues. The version for
rvalues is allowed to and should provide move semantics. Thus, it can *steal* the
internal state and resources of the passed argument.
+ If you implement `void foo(X&&)` nor `void foo(const X&)`, `foo()` can be called on
rvalues, but trying to call it on an lvalue will trigger a compile error.

### Returning Rvalue References

You don't have to and should not `move()` return values. According to the language rules,
the standard specifies that for the following code:

```c++
X foo() {
  X x;
  return x;
}
```

The following behavior is guaranteed:

+ If `X` has an accessible copy or move constructor, the compiler may choose to elide the copy.
This is the so-called *return value optimization*.
+ Otherwise, if `X` has a move constructor, `x` is moved.
+ Otherwise, if `X` has a copy constructor, `x` is copied.
+ Otherwise, a compile-time error is emitted.

Note also that returning an rvalue reference is an error if the returned object is local nonstatic
object:

```c++
X&& foo() {
  X x;
  return x; // ERROR: returns reference to nonexisting object
}
```
