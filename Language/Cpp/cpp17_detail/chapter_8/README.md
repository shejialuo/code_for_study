# std::any

## The Basics

In C++14 there weren't many options for holding variable types in a variable.
You could use `void*` of course, but this wasn't safe.

Potentially, `void*` could be wrapped in a class with some type discriminator.

```c++
class MyAny {
  void* _value;
  TypeInfo _typeInfo;
};
```

As you can see, we have some basic form of the type, but there's a bit of coding
required to make sure `MyAny` is type-safe. And this is what `std::any` from C++17
is in its basic form. It lets you store anything in an object and it reports errors
when you'd like to access a type that is not active.

```c++
std::any a{12};
a = std::string{"Hello"};
a = 16;

std::cout << std::any_cast<int>(a) << '\n';

try {
  std::cout << std::any_cast<std::string>(a) << '\n';
} catch (const std::bad_any_ast &e) {
  std::cout << e.what() << '\n';
}

// reset anc check if it contains any value
a.reset();
if (!a.has_value()) {
  std::cout << "a is empty!\n";
}

// you can use it in a container
std::map<std::string, std::any> m;
m["integer"] = 10;
m["string"] = std::string{"Hello World"};
m["float"] = 1.0f;
```

## std::any Creation

There are several ways you can create `std::any` object:

+ a default initialization
+ a direct initialization with a value/object
+ in place `std::in_place_type`.
+ via `std::make_any`.

[anyCreation.cpp](./anyCreation.cpp)

### In Place Construction

Following the style of `std::optional` and `std::variant`, `std::any` can use
`std::in_place_type` to efficiently create objects in place.

## Changing the Value

When you want to change the currently stored value in `std::any` then you have
two options: use `emplace` or the assignment:

```c++
std::any a;

a = std::string("Hello");

a.emplace<float>(100.5f);
a.emplace<std::vector<int>>({10, 11, 12, 13});
```

### Object Lifetime

The crucial part of being safe for `std::any` us not to leak any resources. To
achieve this behavior `std::any` will destroy any active object before assigning
a new value.

## Accessing The Stored Value

To read the currently active value in `std::any` you have mostly one option: `std::any_cast`.
This function returns the value of the requested type if it's in the object.

However, this function template is quite powerful, as it has three modes:

+ read access: returns a copy of the value, and throws `std::bad_any_cast` when it fails.
+ read/write access: returns a reference, and throws `std::bad_any_cast` when it fails.
+ read/write access - returns a pointer to the value (const or not) or `nullptr` on failure

[anyAccess.cpp](./anyAccess.cpp)

## Performance & Memory Considerations

`std::any` looks quite powerful, and you might use it to hold variables of variable types.
The main issue is extra dynamic memory allocations. `std::variant` and `std::optional` don't
require any extra memory allocations but this is because they know which type will be
stored in the object.
