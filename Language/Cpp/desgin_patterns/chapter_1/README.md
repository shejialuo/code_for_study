# Chapter 1 Introduction

## Important Concepts

### Curiously Recurring Template Pattern

There is a pattern that an inheritor passes *itself* as a template
argument to its base calss:

```c++
struct Foo: SomeBase<Foo> {}
```

Now, you might be wondering *why* one would ever do that? Well, one
reason is to be able to access a typed *this* pointer inside a base
class implementation.
