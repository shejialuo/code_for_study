# std::string_view

In C++17 you get a new type called `string_view`. It allows you to create
a constant, non-owning *view* of a contiguous character sequence. You can
manipulate that view and pass it around without the need to copy the
reference data. Nevertheless, the feature comes at some cost: you need to
be careful at some cost: you need to be careful not to end up with
"dangling" views.

## The Basics

Let's try a little experiment, how many string copies are created in the below
example?

```c++
std::string startFromWordStr(const std::string &strArg, const std::string &word) {
  return strArg.substr(strArg.find(word));
}

std::string str {"Hello Amazing Programming Environment" };
auto subStr = StartFromWordStr(str, "Programming Environment");
std::cout << subStr << '\n';
```

The answer is 3 or 5 depending on the compiler, but usually, it should be 3.

+ The first one is for `str`.
+ The second one is for the second argument in `startFromWordStr` - the argument
is `const string&` so since we pass `const char*` it will create a new string.
+ The third one comes from `substr` which returns a new string.
+ Then we might also have another copy or two - as the object is returned from the function.
But usually, the compiler can optimize and elide the copies
+ If the string is short, then there might be no heap allocation as Small String Optimisation

A much better pattern to solve the problem with temporary copies is to use `std::string_view`.
Instead of using the original string, you'll only get a non-owning view of it.

Most of the time it will be a pointer to the contiguous character sequence and the length. You
can pass it around and use most of the conventional string operations.

Views work well with string operations like substring - `substr`. In a typical case, each
substring operation creates another, smaller copy of the string. With `string_view`, `substr`
will only map a different portion of the original buffer, without additional memory usage, or
dynamic allocation.

Here's the updated version of our code that accepts `string_view`:

```c++
std::string_view startFromWord(std::string_view str, std::string_view word) {
  return str.substr(str.find(word));
}

std::string str {"Hello Amazing Programming Environment"};
auto subView = StartFromWord(str, "Programming Environment");
std::cout << subView << '\n';
```

In the above case, we have only one allocation.

## When to use

+ Optimization: you can carefully review your code an replace various operations
with `string_view`.
+ As a possible statement for `const std::string&` parameter, especially in functions
that don't need the ownership and don't store the string.
+ Handling strings coming from other API everything that is placed in a contiguous
memory chunk and has a basic char-type.

In any case, it's important to remember that's only a *non-owning view*, so if the
original object is gone, the view becomes rubbish and you might get into trouble.

## std::string_view Creation

You can create a `string_view` in serval ways:

+ from `const char*`
+ from `const char*` with length
+ by using a conversion from `std::string`
+ by using `""sv` literal.

[string_view_creation.cpp](./string_view_creation.cpp)

## Risk Using string_view

### Taking Care of Not Null-Terminated Strings

`string_view` may not contain `\0` at the end of the string.

### References and Temporary Objects

`string_view` doesn't own the memory, so you have to be very careful when working
with temporary objects.

In general, the lifetime of a `string_view` must never exceed the lifetime of
the string-owning object. That might be important when:

+ Returning `string_view` from a function.
+ Storing `string_view` in objects or containers.

## Initializing string Members from string_view

Since `string_view` is a potential replacement for `const string&` when passing
in functions, we might consider a case of `string` member initialization. For example:

```c++
class UserName {
  std::string mName;

public:
  UserName(const std::string& str) : mName(str) {}
};
```

You could potentially replace a constant reference with `string_view`:

```c++
UserName(std::string_view sv) : mName(sv) {}
```

Let's compare those alternatives implementations in three cases: creating from
a string literal, creating from an `l-value` and creating from an `rvalue` reference.

```c++
// creation from a string literal
UserName u1{"John With Very Long Name"};

// creation from l-value:
std::string s2 {"Marc With Very Long name"};
UserName u2 { s2 };

// from r-value reference
std::string s3 {"Marc With Very Long Name"};
UserName u3 { std::move(s3) };

std::string GetString() { return "some string..."; }
UserName u4 { GetString(); }
```

For `const std::string&`:

+ `u1` two allocations: the first one creates a temp string and binds it to the
input parameter, and then there's a copy into `mName`.
+ `u2` one allocation: we have a no-cost binding to the reference.
+ `u3` one allocation: we have a no-cost binding to the reference.
+ You have to write a constructor taking rvalue reference to skip one allocation
for the `u1` case, and also that could skip one copy for the `u3` case.

For `std::string_view`:

+ `u1` one allocation: no copy/allocation for the input parameter, there’s only one allocation
when `mName` is created.
+ `u2` one allocation: there's a cheap creation of a `std::string_view` for the argument, and
then there's a copy into the member variable.
+ `u3` one allocation: there's a cheap creation of a `std::string_view` for the argument, and
then there's a copy into the member variable.
+ You also have to write a constructor taking rvalue reference if you want to save
one allocation in the `u3` case.

However, since the introduction of move semantics in C++11, it’s usually better, and safer to pass
string as a value and then move from it.

However, is `std::string` cheap to move? Although the C++ Standard doesn't specify that,
usually, strings are implemented with *Small String Optimization*. The string object contains
extra space to fit characters without additional memory allocation. That means that
moving a string is the same as copying it.

## Handling Non-Null Terminated Strings

If you get a `string_view` from a `string` then it will point to a
null-terminated chunk of memory:

```c++
std::string s = "Hello World\n";
std::cout << s.size() << '\n';
std::string_view sv = s;
std::cout << sv.size() << '\n';
```

But what if you have just a part of the `string`:

```c++
std::string s = "Hello World";
std::cout << s.size() << '\n';
std::string_view sv = s;
auto sv2 = sv.substr(0, 5);
std::cout << sv2.data() << '\n'; /// ooops?
```
