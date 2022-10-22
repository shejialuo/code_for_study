# Utilities

## Pairs

### Piecewise Construction

Class `pair<>` provides piecewise construction which uses tuples to pass
their elements to the constructors of `first` and `second`. To force this
behavior, you have to pass `std::piecewise_construct` as an additional
first argument.

Only where `std::piecewise_construct` is passed as the first argument is to
use a constructor that takes the elements of the tuple rather than a tuple
as a whole.

Note that this form of initialization is required to `emplace()` a new element
into an map or multimap.

### Convenience Function make_pair()

Since C++11, the C++ standard library states that `make_pair()` is declared as:

```c++
namespace std {
  template<typename T1, typename T2>
  pair<V1, V2> make_pair(T1&& x, T2&& y) {
    return pair<T1, T2>(x, y);
  }
}
```

To force reference semantics, you have to use `ref()`, which forces a reference type,
or `cref()`, which forces a constant reference type.

```c++
#include <utility>
#include <functional>
#include <iostream>

int i = 0;
auto p = std::make_pair(std::ref(i), std::ref(i));
++p.first;
++p.second;
std::cout << "i: " << i << std::endl;
```

Since C++11, you can also use `tie()` interface, defined in `<tuple>` to extract values
out of a pair.

```c++
#include <utility>
#include <tuple>
#include <isotream>

std::pair<char, char> p = std::make_pair('x', 'y');

char c;
std::tie(std::ignore, c) = p;
```

## Tuples

In principle, the tuple interface is very straightforward:

+ You can create a tuple by declaring it either explicitly or implicitly with the convenience
`make_tuple`.
+ You can access elements with the `get<>()` function template.

### Tuples and Initializer Lists

The constructor taking a variable number of arguments to initailzie a tuple is
declared as `explicit`:

```c++
namespace std {
  template<typename... Types>
  class tuple {
    public:
      explicit tuple(const Types&...);
      template <typename... UTypes> explicit tuple(UTypes&&...);
  };
}
```

The reason is to avoid having single values implicitly converted into a tuple
with one element:

```c++
template<typename... Args>
void foo(const std::tuple<Args...> t);

foo(42); // ERROR
foo(make_tuple(42)); // OK
```

This situation, however, has consequences when using initializer lists to define values
of a tuple. For example, you can't use the assignment syntax to initialize a tuple.

```c++
std::tuple<int, double> t1(42, 3.14);
std::tuple<int, double> t2{42, 3.14};
std::tuple<int, double> t3 = {42, 3.14}; // ERROR
```

### Additional Tuple Features

For tuples, some additional helpers are declared, especially to support generic programming:

+ `tuple_size<tupletype>::value` yields the number of elements.
+ `tuple_element<idx, tupletype>::type` yields the type of the element with index `idx`.
+ `tuple_cat()` concatenates multiple tuples into one tuple

## Smart Pointers

### Class shared_ptr

Class `shared_ptr` provides this semantics of *shared ownership*. Thus, multiple `shared_ptr`s
are able to share, or "own", the same object. The last owner of the object is reponsible for
destroying it and cleaning up all resources associated with it.

By default, the cleanup is a call of `delete`, assuming that the object was created with `new`.

### Class weak_ptr

Under certain circumstances, this behavior doesn't work or is not what is intended:

+ Cyclic references
+ You want to share but not own an object.

For both cases, class `weak_ptr` is provided, which allows sharing but not owning an object. This
class requires a shared pointer to get created. Whenever the last shared pointer owning the object
loses its ownership, any weak pointer automatically becomes empty. Thus, besides default and copy
constructors, class `weak_ptr` provides only a constructor taking a `shared_ptr`.

You can't use operator `*` and `->` to access a referenced object of a `weak_ptr` directly. Instead,
you have to create a shared pointer out of it. This makes sense for two reasons:

1. Creating a shared pointer out of a weak pointer checks whether there is still an associated object.
If not, this operation will throw an exception or create an empty shared pointer.
2. While dealing with the referenced object, the shared pointer can't get released.

We uses `lock()` to yield a `shared_ptr`. If you are not sure that the object behind a weak pointer
still exists, you have several options:

+ You can call `expired()`, which returns `true` if the `weak_ptr` doesn't share an object any longer.
This option is equivalent to checking whether `use_count()` is equal to 0 but might be faster.
+ You can explicitly convert a `weak_ptr` into a `shared_ptr` by using a corresponding `shared_ptr`
constructor. If there is no valid referenced object, this constructor will throw a `bad_weak_ptr`
exception.
+ You can call `use_count` to ask for the number of owners the associated object has.

### Misusing Shared Pointers

You have to ensure that only one group of shared pointers owns an object. The following code will
not work:

```c++
int *p = new int;
shared_ptr<int> sp1(p);
shared_ptr<int> sp2(p);
```

The problem is that both `sp1` and `sp2` would release the associated resource when they lose
their ownership of `p`. For this reason, you should always directly intialized a smart pointer the
moment you created the object with its associated resource:

```c++
shared_ptr<int> sp1(new int);
shared_ptr<int> sp2(sp1);
```

The problem might also occur indirectly. For example:

```c++
shared_ptr<Person> mom(new Person(name + "'s mon"));
shared_ptr<Person> dad(new Person(name + "'s dad'"));
shared_ptr<Person> kid(new Person(name));
kid->setParentsAndTheirKids(mom, dad);

class Person {
public:
  void setParentsAndTheirKids(shared_ptr<Person> m = nullptr,
                              shared_ptr<Person> f = nullptr) {
    mother = m;
    father = f;
    if (m != nullptr) {
      m->kids.push_back(shared_ptr<Person>(this)); // ERROR
    }
    if (f != nullptr) {
      f->kids.push_back(shared_ptr<Person>(this)); // ERROR
    }
  }
}
```

The problem is that the creation of a shared pointer out of `this`. We do that because
we want to set `kids` of members `mother` and `father`. But to do that, we need a shared
pointer to the kid, which we don't have at hand. However, creating a new shared pointer
out of `this` doesn't solve the issue, because we the open a new group of owners.

One way to deal with this problem is to pass the shared pointer to the kid as a third
argument. But the C++ standard library provides another option: class
`std::enable_shared_from_this<>`.

You can use class `std::enable_shared_from_this<>` to derive your class, representing objects
managed by shared pointers, with your class name as template argument.

```c++
class Person : public std::enable_shared_from_this<Person> {
public:
  void setParentsAndTheirKids(shared_ptr<Person> m = nullptr,
                              shared_ptr<Person> f = nullptr) {
    mother = m;
    father = f;
    if (m != nullptr) {
      m->kids.push_back(shared_ptr<Person>(this)); // OK
    }
    if (f != nullptr) {
      f->kids.push_back(shared_ptr<Person>(this)); // OK
    }
  }
}
```

### Class unique_ptr

`unique_ptr` implements the concept of *exclusive ownership*, which means that it ensures
that an object and its associated resourecs are "owned" only by one pointer at a time.

Class `unique_ptr<>` does not allow you to initialize an object with an ordinary pointer
by using the assignment syntax.

```c++
std::unique_ptr<int> up = new int; // ERROR
std::unique_ptr<int> up(new int); // OK
```

You can call `release`, which yields the object a `unique_ptr` owned, and gives up ownership
so that teh caller is responsible for this object now.

By default, `unique_ptr`s call `delete` for an object they own if they lose ownership. Unfortunately,
due to the language rules derived from C, C++ can't differentiate between the type of a pointer to
one object and an array of objects.

Fortunately, the C++ standard library provides a partial specialization of class `unique_ptr` for
arrays, which calls `delete[]` for the referenced object.

```c++
std::unique_ptr<std::string[]> up(new std::string[10]);
```
