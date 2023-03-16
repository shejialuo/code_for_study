# 7. std::variant

Another handy wrapper type that we get in C++17 is `std::variant`. This is a
type-safe union. You can store different type variants with the proper object
lifetime guarantee. The new type offers a huge advantage over the C-style union.
You can store all of the types inside.

What's crucial is the fact that the new type enhances implementation of design
patterns.

## Basics

[variantDemo.cpp](./variantDemo.cpp)

## std::variant Creation

There are several ways you can create and initialize `std::variant`:

```c++
// default initialization (the first type has to have a default constructor)
std::variant<int, float> intFloat;

// monostate for default initialization
class NotSimple {
public:
  NotSimple(int, float) {}
};

// std::variant<NotSimple, int>
std::variant<std::monostate, NotSimple, int> okInit;
std::cout << okInit.index() << '\n';

// pass a value
std::variant<int, float, std::string> intFloatString { 10.5f };
std::cout << intFloatString.index()
          << ", value" << std::get<float>(intFloatString) << '\n';

// ambiguity
// double might convert to float or int, so the compiler cannot decide

// std::variant<int, float, std::string> intFloatString { 10.5 };

// ambiguity resolved by in_place
variant<long, float, std::string> longFloatString {
  std::in_place_index<1>, 7.6
};

std::cout << longFloatString.index() << ", value"
          << std::get<float>(longFloatString) << '\n';

// in_place for complex types
std::variant<std::vector<int>, std::string> vecStr {
  std::in_place_index<0> { 0, 1, 2, 3 }
};
std::cout << vecStr.index() << ", vector size "
          << std::get<std::vector<int>>(vecStr).size() << '\n';

// copy-initialize from other variant
std::variant<int, float> intFloatSecond { intFloat };
std::cout << intFloatSecond.index() << ", value "
          << std::get<int>(intFloatSecond) << '\n';

```

There is a special type called `std::monostate`. This is just an empty type
that can be used with variants to present an empty state. The type might be
handy when the first alternative doesn't have a default constructor.

`std::variant` has two `in_place` helpers that you can use:

+ `std::in_place_type`: used to specify which type you want to change/set
in the variant.
+ `std::in_place_index`: used to specify which index you want to change/set.

For variant, we need the helpers for at least two cases:

+ ambiguity
+ efficient complex type creation

## Changing the Values

There are four ways to change the current value of the variant:

+ the assignment operator
+ `emplace`
+ `get` and then assign a new value for the currently active type
+ a visitor

```c++
std::variant<int, float, std::string> intFloatString { "Hello" };

intFloatString = 10;

intFloatString.emplace<2>(std::string("Hello"));

std::get<std::string>(intFloatString) += std::string(" World");

intFloatString = 10.1f;
if (auto pFloat = std::get_if<float>(&intFloatString); pFloat) {
  *pFloat *= 2.0f;
}

```

### Object Lifetime

When you use `union`, you need to manage the internal state: call constructors
or destructors. THis is error-prone and it's easy to shoot yourself in the foot.
But `std::variant` handles object lifetime as you expect.

[variantLifetime.cpp](./variantLifetime.cpp)

## Accessing the Stored Value

Even if you know what the currently active type is you cannot do:

```c++
std::variant<int, float, std::string> intFloatString { "Hello" };
std::string s = intFloatString;
```

So you have to use helper functions to access the value. `std::get<Type|Index>`.

## Visitors for std::variant

A visitor is a "Callable that accepts every possible alternative from every variant`.

```c++
auto PrintVisitor = [](const auto& t) { std::cout << t << '\n'; }

std::variant<int, float, std::string> intFloatString { "Hello" };
std::visit(printVisitor, intFloatString);
```

In the above example, a generic lambda is used to generate all possible overloads. Since
all of the types in the variant support `<<` then we can print them.

Generic lambdas can work if our types share the same "interface", but int most cases, we'd
like to perform different actions based on an active type.

That's wht we can define a structure with several overloads for the `operator()`:

```c++
struct MultiplyVisitor {
  float mFactor;

  MultiplyVisitor(float factor): mFactor(factor) {}

  void operator()(int &i) const {
    i *= static_cast<int>(mFactor);
  }

  void operator()(float &f) const {
    f *= mFactor;
  }

  void operator()(std::string &) const {}
};
```

## Performance & Memory Considerations

`std::variant` uses the memory in a similar way to union: so it will take the max
size of the underlying types. But since we need something that will know what
the currently active alternative is, then we need to use some more space. Plus
everything needs to honour the alignment rules.

## Examples of std::variant

### Error Handling

The basic idea is to wrap the possible return value with some `ErrorCode`, and
that way allow functions to output more information about the errors. Without using
exceptions or output parameters.

[variantErrorHandling.cpp](./variantErrorHandling.cpp)

### Polymorphism

Most of the time in c++ we can safely use runtime polymorphism based on a `vtable`
approach. You have a collection of related types that share the same interface, and
you have a well defined virtual method that can be invoked.

But what if you have "unrelated" types that don't share the same base class? What if
you'd like to quickly add new functionality without changing the code of supported
types?

With `std::variant` and `std::visit` we can build the following example

[variantPolymorphism.cpp](./variantPolymorphism.cpp)
