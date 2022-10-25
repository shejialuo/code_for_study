# Appendix B Value Categories

Expressions are a cornerstone of C++ language, providing the primary
mechanism by which it can express computations. Every expression has
a type, which describes the static type of the value that its computation
produces. Each expression also has a *value category*, which describes
something about how the value was formed and affects how the expression
behaves.

## B.1 Traditional Lvalues and Rvalues

Historically, there were only two value categories: lvalues and rvalues.
Lvalues are expressions that refer to actual values stored in memory
or in a machine register. These expressions may be modifiable, allowing
one to update the stored value.

The term lvalue is derived from the role these expressions could play
within an assignment: The letter "l" stand for left side because only
lvalues may occur on the left-hand side of the assignment. Conversely,
rvalues could occur only on the right-hand side of an assignment
expression.

However, things changed: while `int const` still was a value stored in
memory, it could not occur on the left-side of an assignment:

C++ changed things even further: Class rvalues can occur on the left-hand
side of assignments. Such assignments are actually function calls to
appropriate operator of the class rather than "simple" assignments for
scalar types, so they follow the rules of member function calls.

```c++
// See https://stackoverflow.com/questions/33784044/assigning-to-rvalue-why-does-this-compile
class A {
  private: double content;

  public:
  A():content(0) {}

  A operator+(const A& other) {
    content += other.content;
    return *this;
  }

 void operator=(const A& other) {
       content = other.content;
  }

};

int main() {
  A a, b, c;
  /*
    * Here, (a + b) is the rvalue. However, (a + b) = c
    * is a syntax sugar, which is actually (a+b).operator=(c)
    * it calls a function.
  */
  (a + b) = c;
}
```

Another class of expressions that are lvalues include
pointer dereference operations(`*p`),  and expressions that refer to a
member of a class object(`p->data`). Even calls to functions that return
values of "traditional" lvalue reference type declared with `&` are lvalues.

```c++
std::vector<int> v;
v.front() // yields an lvalue because the return type is an lvalue reference
```

Rvalues are pure mathematical values that don't necessarily have any
associated storage.

### B.1.1 Lvalue-to-Rvalue Conversions

The assignment `x = y` works becuase the expression on the right-hand side,
`y`, undergoes an implicit conversion called the *lvalue-to-rvalue conversion*,
which takes an lvalue and produces an rvalue of the same type by reading from
the storage or register associated with the lvalue.

This conversion accomplishes two things:

+ It ensures that an lvalue can be used wherever an rvalue is expected.
+ It identifies where in the program the compiler may emit a "load" instruction
to read a value from memory.

## B.2 Value Categories Since C++11

When rvalue references were introduced in C++11 in support of move semantics, the
traditional partitioning of expressions into lvalues and rvalues was no longer
sufficient to describe all the C++11 language behaviors. The C++ standardization
committee therefore redesigned the value category system based on three core and
two composite categories (see below figure). The core categories are: *lvalue*,
*prvalue*, and *xvalue*. The composite categories are: *glvalue* and *rvalue*.

![Value Categories since C++11](https://s2.loli.net/2022/03/31/6jbTKsVxcMy8GIr.png)

This C++11 categorization has remained in effect, but in C++17 the characterization
of the categories were reformulated as follows:

+ A *glvalue* is an expression whose evaluation determines the identity of the object,
bit-field, or function.
(i.e., an entity that has storage.)
+ A *prvalue* is an expression whose evaluation initializes an object or a bit-field,
or computes the value of operand of an operator.
+ An *xvalue* is a glvalue designating an object or bit-field whose resources can be
reused(usually because it is about to "expire").
+ A *lvalue* is glvalue that is not an xvalue.
+ A *rvalue* is an expression that is either a prvalue or a xvalue.

Examples of *lvalues* are:

+ Expressions that designate variables or functions
+ Applications of the built-in unary `*` operator("pointer indirection")
+ An expression that is just a string literal
+ A call to a function with a return type that is a lvalue reference

Examples of *prvalues* are:

+ Expressions that consist of a literal that is not a string literal or a user-defined
literal
+ Applications of the built-in unary `&` operator
+ Applications of built-in arithmetic operators
+ A call to a function with a return type that is *not* a reference type
+ Lambda expressions

Examples of *xvalues* are:

+ A call to a function with a return type that is an rvalue reference to an object
type(`std::move()`).
+ A cast to an rvalue reference to an object type.

It's worth emphasizing that glvalues, prvalues, xvalues, and so on, are
*expressions*, and *not* values or entities. For example, a variable is not a
lvalue even though an expression denoting a variable is an lvalue:

```c++
int x = 3; // x here is a variable, not a lvalue.3 is a prvalue initializing
           // the variable x
int y = x; // x here is a lvalue. The evaluation of that lvalue expression does
           // not produce the value 3, but a designation of an object containing
           // the value 3. That lvalue is the converted to a prvalue, which is
           // what initializes y.
```

### B.2.1 Temporary Materialization

We previously mentioned that lvalues often undergo an lvalue-to-rvalue conversion
because prvalues are the kinds of expressions that initialize objects.

In C++17, there is a dual to this conversion, known as *temporary materialization*:
any time a prvalue validly appears where glvalue (which includes the xvalue case) is
expected, a temporary object is created and initialized with the prvalue, and the
prvalue is replaced by an *xvalue* designating the temporary. For example:

```c++
int f(int const&);
int r = f(3);
```

More generally, a temporary is materialized to be initialized with a prvalue in the
following situations:

+ A prvalue is bound to a reference.
+ A member of a class prvalue is accessed.
+ An array prvalue is subscripted.
+ An array prvalue is converted to a pointer to its first element(array decay).
+ A prvalue appears in a braced initializer list.
+ The `sizeof` or `typeid` operator is applied to a prvalue.
+ A prvalue is the top-level expression in a statement of the form "expr;" or
an expression is cast to `void`.

Thus, in C++17, the object initialized by a prvalue is always determined by the
context, and, as a result, temporaries are created only when they are really needed.
Prior to C++17, prvalues always implied a temporary. Copies of those temporaries could
optionally be elided later on, but a compiler still had to enforce most semantics
constraints of the copy operation. The following example shows a consequence of the
C++17 version of the rules:

```c++
class N{
public:
  N();
  N(N const&) = delete;
  N(N&&) = delete;
};

N make_N() {
  return N{};    // Always create a conceptual temporary prior to C++17
}                // In C++17, no temporary is created at this point.

auto n = make_N();

```

Prior to C++17, the prvalue `N{}` produce a temporary of type `N`, but compilers
were allowed to elide copies and moves of that temporary. In this case, that
manes that the temporary result of calling `make_N()` can be constructed directly
in the storage of `N`; no copy or move operation is needed. Unfortunately, pre-C++17
compilers still have to check that a copy or move operation could be made.

## B.3 Checking Value Categories with decltype

With the keyword `decltype`, it is possible to check the value category of any C++
expression. For any *expression* `x`, `decltype((x))` yields:

+ *type* if `x` is prvalue.
+ *type*`&` if `x` is an lvalue.
+ *type*`&&` if `x` an xvalue.

The double parentheses in `decltype((x))` are needed to avoid producing the declared
type of a named entity in case where the expression `x` does indeed name an entity.

Thus, using type traits for any expression `e`, wec an check its value category:

```c++
if constexpr (std::is_lvalue_reference<decltype((e))>::value) {
  std::cout << "expression is lvalue\n";
} else if constexpr (std::is_rvalue_reference<decltype((e))>::value) {
  std::cout << "expression is rvalue\n";
} else {
  std::cout << "expression is prvalue\n";
}
```

## B.4 Reference Types

Reference Types in C++ such as `int&` interact with value categories in two important
ways. The first is that a reference may limit the value category of an expression it
can bind to. For example, a non-`const` lvalue reference of type `int&` can only
be initialized with an expression that is an lvalue of an lvalue of type `int`. An
rvalue reference of type `int&&` can only be initialized with an expression that is
an rvalue of type `int`.

The second way in which value categories interact with references is with the return
types of functions, where the use of a reference type as the return type affects
the value category of a call to that function. In particular:

+ A call to a function whose return type is an lvalue reference yields an lvalue.
+ A call to a function whose return type is an rvalue reference to an object
type yields an xvalue.
+ A call to a function that returns nonrefernce type yields a prvalue.

```c++
int& lvalue();
int&& rvalue();
int prvalue();

std::is_same_v<decltype(lvalue()), int&>
std::is_same_v<decltype(rvalue()), int&&>
std::is_same_v<decltype(prvalue()), int>
```
