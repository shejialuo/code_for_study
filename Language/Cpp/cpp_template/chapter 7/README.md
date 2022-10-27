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
