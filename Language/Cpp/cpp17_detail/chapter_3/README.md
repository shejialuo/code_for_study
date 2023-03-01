# 3. General Language Features

## Structured Binding Declarations

When you work with `std::set::insert` which returns `std::pair`:

```c++
std::set<int> mySet;
std::set<int>::iterator iter;
bool inserted;

std::tie(iter, inserted) = mySet.insert(10);

if (inserted) {
  std::cout << "Value was inserted\n";
}
```

However, with C++17 the code can be more compact:

```c++
std::set<int> mySet;

auto [iter, inserted] = mySet.insert(10);
```

### The syntax

The basic syntax is as follows:

```c++
auto [a, b, c, ...] = expression;
auto [a, b, c, ...] { expression };
auto [a, b, c, ...] ( expression );
```

### Modifiers

Several modifiers can be used with structured bindings:

```c++
const auto [a, b, c, ...] = expression;
auto& [a, b, c, ...] = expression;
auto&& [a, b, c, ...] = expression;
```

### Binding

Structured Binding is not limited to tuples, we have three cases from which
we can bind from:

If the initializer is an array:

```c++
double myArray[3] = {1.0, 2.0, 3.0 };
auto [a, b, c] = myArray;
```

In this case, an array is copied into a temporary object and `a`, `b` and `c`
refers to copied elements from the array. And the number of identifier must
match the number of elements in the array.

If the initializer supports `std::tuple_size<>` and provides `get<N>()` and
`std::tuple_element` functions:

```c++
auto [a, b] = std::pair(0, 1.0f);
```

If the initializer'type contains only non static, public members:

```c++
struct Point {
  double x;
  double y;
};

Point GetStartPoint() {
  return {0.0, 0.0};
}

const auto [x, y] = GetStartPoint();
```

[mapStructureBinding.cpp](./mapStructureBinding.cpp)

### Providing Structured Binding Interface for Custom Class

You can provide Structured Binding support for a custom class. To do
that you have to define `get<N>`, `std::tuple_size` and `std::tuple_element`
specializations for your type.

```c++
class UserEntry {
public:
  void Load() {}

  std::string GetName() const { return name; }
  unsigned GetAge() const { return age; }

private:
  std::string name;
  unsigned age {0};
  size_t cacheEntry {0};

};

template <size_t I> auto get(const UserEntry& u) {
  if constexpr (I == 0) return u.GetName();
  else if constexpr (I == 1) return u.GetAge();
}

namespace std {
  template <> struct tuple_size<UserEntry> : std::integral_constant<size_t, 2> {};
  template <> struct tuple_element<0,UserEntry> { using type = std::string; };
  template <> struct tuple_element<1,UserEntry> { using type = unsigned; };
}

```

## Init Statement for if and switch

C++17 provides new versions of the if and switch statements:

```c++
if (init; condition) {}
switch (init; condition) {}
```

## Inline Variables

Previously, only methods/functions could be specified as `inline`, but now
you can do the same with variables, inside a header file. For example:

```c++
struct MyClass {
  static const int sValue;
};

// later in the same header file.
inline int const MyClass::sValue = 777;
```
