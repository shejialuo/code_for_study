# Chapter 1 Policy-Based Class Design

This chapter describe policies and policy classes, important class design techniques
that enable the creation of flexible, highly reusable libraries. In brief, policy-based
class design fosters assembling a class with complex behavior out of many little classes
(called *policies*), each of which take care of only one behavioral or structural
aspect. A policy establishes an interface pertaining to a specific issue.

## 1.1 The Multiplicity of Software Design

The multiplicity of the design space constantly confuses apprentice designers. Given
a software design problem, what's a good solution to it? Events? Objects? Observers?
Callbacks? Virtuals? Templates? Up to a certain scale and level of detail, many
different solutions seem to work equally well.

## 1.2 The Failure of the Do-It-All Interface

Implementing everything under the umbrella of do-it-all interface is not a good
solution, for several reasons.

+ Intellectual overhead, sheer size, and inefficiency.
+ Loss of static type safety.

In a large, all-encompassing interface, it is very hard to enforce such constraints.
Typically, once you have chosen a certain set of design constraints, only certain
subsets of the large interface remains semantically valid. A gap grows between
*syntactically valid* and *semantically valid* uses of the library. The programmer
can write an increasing number of constructs that are syntactically valid, but
semantically illegal.

What if the library implements different design choices as different, smaller classes?
Each class would represent specific canned design solution. In the smart pointer case,
you would except a battery of implementations: `SingleThreadedSmartPtr`,
`MultiThreadedSmartPtr`, `RefCountedSmartPtr`, `RefLinkedSmartPtr`, and so on.

The problem that emerges with this second approach is the combinatorial explosion of
the various design choices. The four classes just mentioned lead necessarily to
combinations such as `SingleThreadedRefCountedSmartPtr`.

## 1.3 Multiple Inheritance to the Rescue?

The problems with assembling separate features by using multiple inheritance are
as follows:

+ *Mechanics*. There is no boilerplate code to assemble the inherited components
in a controlled manner.
+ *Type information*. The base classes do not have enough type information to carry
out their tasks.
+ *State manipulation*. Various behavioral aspects implemented with base classes
must manipulate the same state.

## 1.4 The Benefit of Templates

Class templates are customizable in ways not supported by regular classes. If you
want to implement a special case, you can specialize any member functions of a
class template for a specific instantiation of the class template.

However, there are some drawbacks:

+ *You cannot specialize structure*.
+ *Partial specialization of member functions does not scale*.
+ *The library writer cannot provide multiple default values*.

## 1.5 Policies and Policy Classes

A *policy* defines a class interface or a class template interface. The interface
consists of one or all of the following: inner type definitions, member functions,
and member variables.

Policies have much in common with traits but differ in that they put less emphasis
on type and more emphasis on behavior.

For example, let's define a policy for creating objects. The `Creator` policy
prescribes a class template of type `T`. This class template must expose a
member function called `Create` that takes no arguments and returns a pointer to
`T`. Semantically, each call to `Create` could return a pointer to a new object
of type `T`.

Let' define some policy classes that implement the `Creator` policy. One
possible way is to use the `new` operator. Another way is to use `malloc` and
a call to the placement `new` operator. Yet another way would be to create new
objects by cloning a prototype object.

[creator_policy](./creator_policy.cpp)

For a given policy, there can be an unlimited number of implementations. The
implementations of a policy are called *policy classes*. Policy classes are not
intended for standalone use; instead, they are inherited, or contained within,
other classes.

An important aspect is that, unlike classic interfaces, policies's interfaces
are loosely defined. Policies are syntax oriented, not signature oriented. For
example, the `Creator` policy does not specify that `Create` must be static or
virtual, the only requirement is that the class template define a `Create`
member function.

Let's see how we can design a class that exploits the `Creator` policy. Such a
class will either contain or inherit one of the three classes defined previously,
as shown in the following:

```c++
// Library code
template <typename CreationPolicy>
class WidgetManager : public CreationPolicy {};
```

The classes that use one or more policies are called *hosts* or *host classes*. In
the example, `WidgetManager` is a host class with one policy.

When instantiating the `WidgetManager` template, the client passes the desired policy:

```c++
using MyWidgetMgr = WidgetManager<OpNewCreator<Widget>>;
```

Let's analyze the resulting context. Whenever an object of type `MyWidgetMgr` needs
to create a `Widget`, it invokes `Create()` for its `OpNewCreator<Widget>` policy
sub-object. However, it is the user of `WidgetManager` *who chooses the creation policy*.

### 1.5.1 Implementing Policy Classes with Template Template Parameters

Often, the policy's template argument is redundant. It is awkward that the
user must pass `OpNewCreator`'s template argument explicitly. Typically,
the host already knows.

In this case, library code can use *template template parameters* for specifying
policies, as shown in the following:

```c++
// Library code
template <template <typename> class CreationPolicy>
class WidgetManager : public CreationPolicy<Widget> {};
```

Application code now only needs to provide the name of the template in instantiating
`WidgetManager`:

```c++
// Application code
using MyWidgetMgr = WidgetManager<OpNewCreator>;
```

Using template template parameters with policy classes is not simply a matter of
convenience; sometimes, it is essential that the host class have access to the
template so that the host can instantiate it with a different type. For example,
assume `WidgetManager` also needs to create objects of type `Gadget` using the
same creation policy. Then the code would look like this:

```c++
// Library code
template <template <typename> class CreationPolicy>
class WidgetManager : public CreationPolicy<Widget> {
  void DoSomething() {
    Gadget *pw = CreationPolicy<Gadget>().Create();
  }
};
```

Using policies gives great flexibility to `WidgetManager`. First, you can change
policies *from the outside* as easily as changing a template argument when you
instantiate `WidgetManager`. Second, you can provide your own policies that are
specific to your concrete application. You can use `new`, `malloc`, prototypes,
or a peculiar memory application library that only your system uses.

To ease the lives of application developers, `WidgetManager`'s author might define
a battery of often-used policies, in addition, provide a default template argument
for the policy that's most commonly used:

```c++
template <template <typename> class CreationPolicy = OpNewCreator>
class WidgetManager {};
```

### 1.5.2 Implementing Policy Classes with Template Member Functions

An alternative to using template template parameters is to use template member
functions in conjunction with simple classes.

For example, we can redefine the `Creator` policy to prescribe a regular class
that exposes a template function `Create<T>`.

```c++
struct OpNewCreator {
  template <typename T>
  static T *Create() {
    return new T;
  }
}
```

## 1.6 Enriched Policies

The `Creator` policy prescribes only one member function, `Create`. However,
`PrototypeCreator` defines two more functions: `GetPrototype` and `SetPrototype`.

A user who uses a prototype-based `Creator` policy class can write the following code:

```c++
using MyWidgetManager = WidgetManager<PrototypeCreator>;

Widget* pPrototype = ...;
MyWidgetManager mgr;
mgr.SetPrototype(pPrototype);
```

## 1.7 Destructors of Policy Classes

Define a virtual destructor for a policy works against its static nature and
hurts performance. Many policies don't have any data members, but rather are
purely behavioral by nature. The first virtual function added incurs some size
overhead for the objects of that class, so the virtual destructor should be
avoided.

The lightweight, effective solution that policies should use is to define a
non-virtual protected destructor:

```c++
template <class T>
struct OpNewCreator {
  static T* Create() {
    return new T;
  }
protected:
  ~OpNewCreator() {}
};
```

Because the destructor is protected, only derived classes can destroy policy
objects, so it's impossible for outsiders to apply `delete` to a pointer to
a policy class.

## 1.8 Optional Functionality Through Incomplete Instantiation

In conjunction with policy classes, incomplete instantiation brings remarkable freedom
to you as a library designer. You can implement lean host classes that are able to use
additional features and degrade graciously.

## 1.9 Combining Policy Classes

The greatest usefulness of policies is apparent when you combine them. Typically,
a highly configurable class uses several policies for various aspects of its workings.
Then the library user selects the desired high-level behavior by combining several
policy classes.

For example, consider designing a generic smart pointer class. Say you identify two
design choices that you should establish with policies: threading model and
checking before dereferencing. Then you implement a `SmartPtr` class template that
use two polices, as shown:

```c++
template
<
  typename T,
  template <typename> class CheckingPolicy,
  template <typename> class ThreadingModel
>
class SmartPtr;
```

The two polices can defined as follows:

1. `Checking`: The `CheckingPolicy<T>` class template must expose a `Check` member
function, callable with an lvalue of type `T*`
2. `ThreadingModel`: The `ThreadingModel<T>` class template must expose an inner
type called `Lock`, whose constructor accepts a `T&`.

```c++
template <typename T>
struct NoChecking {
  static void Check(T *) {}
};
template <typename T>
struct EnforceNotNull {
  class NullPointerException : public std::exception {};
  static void Check(T *ptr) {
    if (!ptr) {
      throw NullPointerException();
    }
  }
};
```

`SmartPtr` uses the `Checking` policy this way:

```c++
template
<
  typename T,
  template <typename> class CheckingPolicy,
  template <typename> class ThreadingModel
>
class SmartPtr: public CheckingPolicy<T>, public ThreadingModel<T> {
  T* operator->() {
    typename ThreadingModel<SmartPtr>::Lock guard(*this);
    CheckingPolicy<T>::Check(pointee_);
    return pointee_;
  }
private:
  T *pointee_;
};
```

## 1.10 Customizing Structure with Policy Classes

Suppose that you want to support nonpointer representations for `SmartPtr`.
To solve this you might "indirect" the pointer access through a policy, say,
a `Structure` policy. The `Structure` policy abstracts the pointer storage.

```c++
template <typename T>
class DefaultSmartPtrStorage {
public:
  using PointerType = T*;
  using ReferenceType = T&;
protected:
  PointerType GetPointer() { return ptr_; }
  oid SetPointer(PointerType ptr) {
    ptr_ = ptr;
  }
private:
  PointerType ptr_;
};
```
