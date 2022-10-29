# Chapter 7 By Value or by Reference?

Deciding how to declare parameters with known concrete types is complicated
enough. In templates, types are not known, and therefore it becomes even
harder to decide which passing mechanism is appropriate.

Nevertheless, we recommend passing parameters in function templates by
value unless there are good reasons, such as the following:

+ Copying is not possible.
+ Parameters are used to return data.
+ Templates just forward the parameters to somewhere else by keeping all
the properties of the original arguments.
+ There are significant performance improvements.

## 7.1 Passing by Value

When passing arguments by value, each argument must in principle be copied. Thus,
each parameter becomes a copy of the passed argument. For classes, the object
created as a copy generally is initialized by the copy constructor.

Calling a copy constructor can become expensive. However, there are various way
to avoid expressive copying even when passing parameters by value: In fact,
compilers might optimize away copy operations copying objects and can become
cheap even for complex objects by using move semantics.

```c++
template<typename T>
void printV(T arg) {}
```

The potential copy constructor is not always called.

```c++
std::string returnString();
std::string s = "hi";
printV(s);                 // copy constructor
printV(std::string("hi")); // copying usually optimized away(if not, move constructor)
printV(returnString());    // copying usually optimized away(if not, move constructor)
printV(std::move(s));      // move constructor
```

In the first call, we pass an *lvalue*, which means that the copy constructor is used.
However, in the second and third calls, when directly calling the function template
for *prvalues*, compilers usually optimize passing the argument so that no copying
constructor is called at all. Not that since C++17, this optimization is required.
Before C++17, a compiler that doesn't optimize the copying away, must at least
have to try to use move semantics, which usually makes copying cheap. In the last
call, when passing an *xvalue*, we force to call the move constructor by signaling
that we no longer need the value of `S`.

## 7.2 Passing by Reference

Now let's discuss the difference flavors of passing by reference. In all cases, no
copy get created. Also, passing the argument never decays.

### 7.2.1 Passing by Constant Reference

To avoid any copying, when passing non-temporary objects, we can use constant references.
For example:

```c++
template<typename T>
void printR(T const& arg) {}
```

With this declaration, passing an object never creates a copy. Under the hood, passing an
argument by reference is implemented by passing the address of the argument. Addresses
are encoded compactly, and therefore transferring an address from the caller to the callee
is efficient in itself. However, passing an address can create uncertainties for the compiler
when it compiles the caller's code: What is the callee doing with that address? In theory,
the callee can change all the values that are "reachable" through that address. That means,
that the compiler has to assume that all the values it may have cached are invalid after
the call. Reloading all those values can be quite expensive.

This bad news is moderated by inlining: If the compiler can expand the call *inline*, it can
reason about the caller and the callee *together* and in many cases "see" that the address
is not used for anything about passing the underlying value. Function templates are often
very short and therefore likely candidates for inline expression.

### 7.2.2 Passing by Non-constant reference

When you want to return values through passed arguments, you have to use non-constant references.
Again, this means that when passing the arguments, no copy get created. The parameters of the
called function template just get direct access to the passed argument.

Consider the following:

```c++
template<typename T>
void outR(T& arg) {}
```

Note that calling `outR()` for a temporary or an existing object passed with `std::move()`
usually is not allowed.

```c++
std::string returnString();
std::string s = "hi";
outR(s);                 // OK: T deduced as std::string
outR(std::string("hi")); // ERROR: not allowed to pass a prvalue
outR(reduceString());    // ERROR: not allowed to pass a prvalue
outR(std::move(s));      // ERROR: not allowed to pass an xvalue
```

You can pass raw arrays of non-constant types, which again don't decay:

```c++
int arr[4];
outR(arr);               // OK: T deduced as int[4], arg is int(&)[4]
```

Thus, you can modify elements and, for example, deal with the size of the array.
For example:

```c++
void outR(T &arg) {
  if (std::is_array<T>::value) {
    std::cout << "got array of " << std::extent<T>::value << " elems\n";
  }
}
```

However, templates are a bit tricky here. If you pass a `const` argument, the
deduction might result in `arg` becoming a declaration of a constant reference,
which means that passing an rvalue is suddenly allowed.

```c++
std::string const c = "hi";
outR(c);                   // OK: T deduced as std::string const
outR(returnConstString()); // OK: same if returnConstString() returns const string
outR(std::move(c));        // OK: T deduced as std::string const
outR("hi");                // OK: T deduced as char const[3]
```

If you want to disable passing constant objects to non-constant references, you
can do the following:

+ Use a static assertion to trigger a compile-time error:

  ```c++
  template<typename T>
  void outR(T& arg) {
    static_assert(!std::is_const<T>::value, "out parameter of foo<T>(T&) is const");
  }
  ```

+ Disable the template for this case either by using `std::enable_if<>`:

  ```c++
  template<typename T, typename = std::enable_if_t<!std::is_const<T>::value>>
  void outR(T &arg) {}
  ```

+ Concepts once they are supported.

  ```c++
  template<typename T>
  requires !std::is_const_v<T>
  void outR(T& arg) {}
  ```

### 7.2.3 Passing by Forwarding Reference

One reason to use call-by-reference is to be able to perfect forward a parameter.

```c++
template<typename T>
void passR(T&& arg) {}
```

You can pass everything to a forwarding reference.

```c++
std::string s = "hi";
passR(s);                 // OK: T deduced as std::string& (also the type of arg)
passR(std::string("hi")); // OK: T deduced as std::string, arg is std::string&&
passR(std::move(s));      // OK: T deduced as std::string, arg is std::string&&
passR(arr);               // OK: T deduced as int(&)[4] (also the type of arg)
```

However, the special rules for type deduction may result in some surprises:

```c++
std::string const c = "hi";
passR(c);     // OK: T deduced as std::string const&
passR("hi");  // OK: T deduced as char const(&)[3]
int arr[4];
passR("hi");  // OK: T deduced as int (&)[4]
```

## 7.3 Using std::ref() and std::cref()

Since C++11, you can let the caller decide, for a function template argument, whether
to pass it by value or by reference. When a template is declared to take arguments
by value, the caller can use `std::cref()` and `std::ref()`, declared in header file
`<functional>`, to pass the argument by reference. For example:

```c++
template<typename T>
void printT(T arg) {}

std::string s = "hello";
printT(s);
printT(std::cref(s));
```

However, note that `std::cref()` dose not change the handling of parameters in templates.
Instead, it uses a trick; It wraps the passed argument `s` by an object that acts like
a reference. In fact, it creates an object of type `std::reference_wrapper<>` referring
to the original argument and passes this object *by value*. The wrapper more or less
supports only one operation: an implicit type conversion back to the original type, yielding
the original object.

[cref.cpp](./cref.cpp)

## 7.4 Dealing with String Literals and Raw Arrays

So far, we have seen the different effects for templates parameters when using string literals
and raw arrays:

+ Call-by-value decays so that they become pointers to the element type.
+ Any form of call-by-reference does not decay so that the arguments become references
that still refer to arrays.

When decaying arrays to pointers, you lose the ability to distinguish between handling pointers
to elements from handling passed arrays. On the other hand, when dealing with parameter types
where string literals may be passed, not decaying can become a problem. For example:

```c++
template<typename T>
void foo(T const& arg1, T const& arg2) {}

foo("hi", "guy"); // ERROR
```

Here, `foo("hi", "guy")` fails to compile, because `"hi"` has type `char const[3]`, while `"guy"`
has type `char const[4]`, but the template requires them to have the same type `T`.

By declaring the function template `foo()` to pass the argument by value the call is possible:

```c++
template<typename T>
void foo(T arg1, T arg2) {}

foo("hi", "guy");
```

**But**, that doesn't mean that all problems are gone. Even worse, compile-time may have become
run-time problems.

```c++
template<typename T>
void foo(T arg1, T arg2) {
  if (arg1 == arg2) {} // OOPS: compares addresses of passed arrays
}

foo("hi", "guy");
```

### 7.4.1 Special Implementations for String Literals and Raw Arrays

You might have to distinguish your implementation according to whether a pointer or an array
was passed. This, of course, requires that a passed array wasn't decayed yet.

To distinguish these cases, you have to detect whether arrays are passed. Basically, there
are two options:

+ You can declared template parameters so that they are only valid for arrays:

  ```c++
  template<typename T, std::size_t L1, std::size_t L2>
  void foo(T (&arg1)[L1], T (&arg2)[L2]) {
    T *pa = arg1;
    T *pb = arg2;
    if (compareArrays(pa, L1, pb, L2))
  }
  ```

+ You can use type traits to detect whether an array (or a pointer) was passed:

  ```c++
  template<typename T, typename = std::enable_if_t<std::is_array_v<T>>>
  void foo(T && arg1, T&& arg2) {}
  ```

## 7.5 Dealing with Return Values

For return values, you can also decide between returning by value or by reference.
However, returning references is potentially a source of trouble, because you
refer to something that is out of your control. There are a few cases where
returning references is common programming practice:

+ Returning elements of containers or strings.
+ Granting write access to class members.
+ Returning objects for chained calls.

However, return by reference would cause trouble. We should therefore ensure that the
function templates return their result by value. However, `T` is no guarantee that
it is not a reference, because `T` might sometimes implicitly be deduced as a reference

```c++
template<typename T>
T retR(T&& p) {
  return T{};  // OOPS: returns by reference when called for lvalues.
}
```

Even when `T` is a template parameter deduced from a call-by-value call, it might become
a reference type when explicitly specifying the template parameter to become a reference:

```c++
template<typename T>
T retV(T p) {
  return T{}
}

int x;
retV<int&>(x); // retV() instantiated for T as int&
```

To be safe, you have two options:

+ Use the type trait `std::remove_reference<>` to convert type `T` to a non-reference:

  ```c++
  template<typename T>
  typename std::remove_reference<T>::type retV(T p) {
    return T{}; // always return by value
  }
  ```

+ Let the compiler deduce the return type by just declaring the return type to be `auto`,
because `auto` always decays:

  ```c++
  template<typename T>
  auto retV(T p) {
    return T{}; // always return by values
  }
  ```

## 7.6 Recommended Template Parameter Declarations

As we learned in the previous sections, we have very different ways to declare parameters
that depend on template parameters:

+ Declare to pass the arguments by value:

  This approach is simple, it decays string literals and raw arrays, but it doesn't provide
  the bst performance for large objects. Still the caller can decide to pass by reference
  using `std::cref()` and `std::ref()`, but the caller must be careful that doing so is valid.

+ Declare to pass the arguments by reference

  This approach often provides better performance for somewhat large objects, especially when
  passing

  + existing objects (lvalues) to lvalue references
  + temporary objects (prvalues) or objects marked as movable (xvalue) to rvalue rerference.
  + or both to forwarding references

  Because in all these cases the arguments don't decay, you may need special care when passing
  string literals and other raw arrays.



