# Chapter 10 Basic Template Terminology

Before we go into details, let's look at the terminology we use. This
is necessary because, inside the C++ community, there is sometimes
a lack of precision regarding terminology.

## 10.1 "Class Template" or "Template Class"

In C++, structs, classes and unions are collectively called *class type*.
Without additional qualification, the word "class" in plain text type
is meant to include class types introduced with either the keyword
`class` or the keyword `struct`.

There is some confusion about how a class that is a template is called:

+ The term *class template* states that the class is a template. That is,
it is a parameterized description of a family of classes.
+ The term *template class*, on the other hand, has been used
  + as a synonym for class template.
  + to refer to a classes generated from templates.
  + to refer to classes with a name that is a *template-id*.

We avoid the term *template class*. Similarly, we use *function template*,
*member function template* and *variable template*.

## 10.2 Substitution, Instantiation, and Specialization

When processing source code that uses templates, a C++ compiler must at
various times *substitute* concrete template arguments for the template
parameters for the template parameters in the template. The compiler
may need to check if the substitution could be valid.

The process of actually creating a *definition* for a regular class, type
alias, function, member function, or variable from a template by substituting
concrete arguments for the template parameters is called *template instantiation*.

The entity resulting from an instantiation or an incomplete instantiation is
generally called a *specialization*.

However, in C++ the instantiation process is not the only way to produce a specialization.
Alternative mechanisms allow the programmer to specify explicitly a declaration that is
tied to a special substitution of template parameters, such a specialization is introduced
with the prefix `template <>`:

```c++
template<typename T1, typename T2>
class MyClass {};

template<>
class MyClass<std::string, float> {};
```

Strictly speaking, this is called an *explicit specialization* as opposed to an *instantiated*
or *generated specialization*.

Specializations that still have template parameters are called *partial specialization*:

```c++
template<typename T, typename T>
class MyClass<T, T> {};

template<typename T>
class MyClass<bool, T> {};
```

## 10.3 Declaration versus Definition

A *declaration* is a C++ construct that introduces or reintroduces a name into a C++
scope. This introduction always includes a partial classification of that name, but the
details are not required to make a valid declaration. For example:

```c++
class C;
void f(int p);
extern int v;
```

Note that even though they have a "name", macro definitions and `goto` labels are not
considered declaration in C++.

Declarations become *definitions* when the details of their structure are made known or,
in the case of variables, when storage space must be allocated.

```c++
class C{}; // definition (and declaration) of class C

void f(int p) { // definition (and declaration) of function f()
  std::cout << p << '\n';
}

extern int v = 1; // an initializer makes this a definition of v

int w; // global variable declarations not preceded by extern are also definitions
```

### 10.3.1 Complete versus Incomplete Types

Types can be *complete* or *incomplete*, which is a notion closely related to the
distinction between *declaration* and a *definition*.

Incomplete types are one of the following:

+ A class type that has been declared but not yet defined.
+ An array type with an unspecified bound.
+ An array type with an incomplete element type.
+ `void`.
+ An enumeration type as long as the underlying type or the enumeration values are not defined.
+ Any type above to which `const` and/or `volatile` are applied.

## 10.4 The One-Definition Rule

See Appendix A.

## 10.5 Template Arguments versus Template Parameters

Comparing the following class template:

```c++
template<typename T, int N>
class ArrayInClass {
public:
  T array[N];
};

class DoubleArrayInClass {
public:
  double array[10];
};
```

The `DoubleArrayInClass` becomes essentially equivalent to the `ArrayInClass` if we
replace the parameters `T` and `N` by `double` and `10` respectively. In C++, the
name of this replacement is denoted as `ArrayInClass<double, 10>`. Note how the name
of the template is followed by *template arguments* in angle brackets.

Regardless of whether these arguments are themselves dependent on template parameters,
the combination of the template name, followed by the arguments in angle brackets, is
called a *template-id*.

It is essential to distinguish between *template parameters* and *template arguments*.

+ *Template parameters* are those names that are listed after the keyword `template`
in the template declaration or definition.
+ *Template arguments* are the items that are substituted for template parameters.
