# 6. std::optional

C++17 adds a few wrapper types that make it possible to write more
expressive code.

## Introduction

You can achieve "null-ability" by using unique values (-1, infinity, `nullptr`), it's
not as clear as the separate wrapper type. Alternatively, you could even use
`std::unique_ptr<Type>` and treat the empty pointer as not initialized.

## std::optional Creation

There are several ways to create `std::optional`:

+ Initializes as empty.
+ Directly with a value.
+ With a value using deduction guides.
+ By using `make_optional`.
+ With `std::in_place`.
+ From other `optional`.

```c++
// empty
std::optional<int> oEmpty;
std::optional<float> oFloat = std::nullopt;

// direct;
std::optional<int> oInt(10);
std::optional oIntDeduced(10);

// make_optional
auto oDouble = std::make_optional(3.0);
auto oComplex = std::make_optional<std::complex<double>>(3.0, 4.0);

// in_place
std::optional<std::complex<double>> o7{std::in_place, 3.0, 4.0};

// will call vector with direct init of {1,2,3}
std::optional<std::vector<int>> oVec(std::in_place, {1, 2, 3});

// copy from other optional:
auto oIntCopy = oInt;
```

### in_place Construction

What's the advantage of using `std::in_place_t` in `std::optional`? There
are at least two important reasons:

+ Default constructor.
+ Efficient construction for constructors with many arguments.

[optionalInPlaceDefault.cpp](./optionalInPlaceDefault.cpp)

The reason is that if we do not use `std::in_place`, we may call move constructor.
It's bad for efficiency. And if the type doesn't support move or copy constructor.
`std::in_place` is the only way to work with such types.

### std::make_optional()

If you don't like `std::in_place` then you can look at `make_optional` factory
function. `make_optional` implements in place construction equivalent to

```c++
return std::optional<T>(std::in_place, std::forward<Args>(args)...);
```

## Returning std::optional

If you return an optional from a function, then it's very convenient to return just
`std::nullopt` or the computed value.

```c++
std::optional<std::string> TryParse(Input input) {
  if (input.valid()) {
    return input.asString();
  }
  return std::nullopt;
}
```

Of course, you can also declare an empty optional at the beginning of your function
and reassign if you have the computed value.

```c++
std::optional<std::string> TryParse(Input input) {
  std::optional<std::string> oOut;
  if (input.valid()) {
    oOut = input.asString();
  }
  return std::nullopt;
}
```

However, it's more optimal to use the first version to avoid creating temporaries.

### Be Careful With Braces when Returning

You might be surprised by the following code:

```c++
std::optional<std::string> CreateString() {
  std::string str {"Hello Super Awesome Long String"};
  return {str}; // this one will cause a copy
  // return str; // this one moves.
}
```

According to the Standard if you wrap a return value into braces `{}` then you
prevent more operations from happening. The returned object will be copied only.

[optionalReturn.cpp](./optionalReturn.cpp)

## Accessing The Stored Value

The most operation for optional is the way you can fetch the contained value. There
are several options:

+ `operator*` and `operator->`: if there's no value the behavior is undefined.
+ `value()`: returns the value, or throws `std::bad_optional_access`.
+ `value_or(defaultVal)`: returns the value if available, or `defaultVal` otherwise.

To check if the value is present you can use the `has_value()` method or just check
`if (optional)` as optional is contextually convertible to `bool`.

## std::optional Operations

If you have an existing optional object, then you can quickly change the contained
value by using several operations like `emplace`, `reset`, `swap`, assign. If you
assign or reset with a `nullopt` then if the optional contains a value its destructor
will be called.

[optionalRest.cpp](./optionalRest.cpp)

`std::optional` allows you to compare contained objects almost "normally", but with
a few exceptions whe the operands are `nullopt`.

## Performance & Memory Consideration

When you use `std::optional` you'll pay with an increase memory footprint. Conceptually
your version of the standard library might implement optional as:

```c++
template <typename T>
class optional {
  bool _initialized;
  std::aligned_storage_t<sizeof(t), alignof(T)> _storage;
public:
}
```
