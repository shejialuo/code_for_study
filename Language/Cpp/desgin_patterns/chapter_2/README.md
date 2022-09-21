# Chapter 2 Builder

The Builder pattern is concerned with the creation of *complicated* objects,
that is, objects that cannot be built up in a single-line constructor call.
These types of objects may themselves be composed of other objects
and might involve less-than-obvious logic, necessitating a separate
component specifically dedicated to object construction.

## Scenario

Let's imagine that we are building a component that renders web pages.
To start with, we shall output a simple unordered list with two items
containing the words *hello* and *world*. A very simplistic implementation might
look as follows: (see `simpleWebRender.cpp`)

This does in fact give us what we want, but the approach is not
very flexible. How would we change this from a bulleted list to
a numbered list? How can we add another item *after* the list has been
created?

We might, therefore, go the OOP route and define an `HtmlElement` class
to store information about each tag: (see `htmlElement.cpp`)。

This works fine and gives us a more controllable, OOP-driven representation
of a list of items. But the process of building up each `HtmlElement` is not
very convenient, and we can improve it by implementing the
Builder pattern.

## Simple Builder

The Builder pattern simply tries to outsource the piecewise construction
of an object into a separate class.(see `simpleHtmlBuilder.cpp`)

You'll notice that, at the moment, the `addChild()` function is `void`-return.
There are many things we could use the return value for, but one of
the most common uses of the return value is to help us build a fluent interface.

## Fluent Builder

By returning a reference to the builder itself, the builder calls can
now be chained. That is what's called a *fluent interface* (see `fluentHtmlBuilder.cpp`)

## Communicating Intent

We have a dedicated Builder implemented for an HTML element, but how
will the users of our classes know how to use it? One idea is to simply
*force* them to use the builder whenever they are constructing an object.
(see ``)
