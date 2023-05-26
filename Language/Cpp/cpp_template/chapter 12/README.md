# Chapter 12 Fundamentals in Depth

In this chapter, we review some of the fundamentals *in depth*.

## 12.1 Parameterized Declarations

C++ currently supports four fundamental kinds of templates:

+ class templates
+ function templates
+ variable templates
+ alias templates

Each of these template kinds can appear in namespace scope, but also
in class scope. In class scope they become nested class templates,
member function templates, static data member templates, and
member alias templates.

First, some examples illustrate the four kinds of the templates.

[definitions1.hpp](./definitions1.hpp)

The following example shows the four kinds of template as class members
that are defined within their parent class:

<!-- TODO: ADD more -->

## 12.2 Template Parameters

There are three basic kinds of template parameters:

+ Type parameters
+ Nontype parameters
+ Template template parameters


