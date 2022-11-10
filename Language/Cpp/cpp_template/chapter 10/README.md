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
