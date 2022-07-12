# 5 Go's type system

This is an easy part. I write notes only for interface.

## 5.1 Interfaces

Polymorphism is the ability to write code that take on different
behavior through the implementation of types. Once a type implements
an interface, an entire word of functionality can be opened up
to values of that type.

### 5.1.1 Implementation

Interfaces are types that just declare behavior. This behavior is never
implemented by the interface type directly but instead by user-defined
types via methods. When a user-defined type implements the set
of methods declared by an interface type, values of the user-defined type
can be assigned to values of the interface type. This assignment
stores the value of the user-defined type into the interface type.

If a method call is made against an interface value, the equivalent method
fir the stored user-defined value is executed.

### 5.1.2 Method sets

Method sets define the rules around interface compliance.

```go
package main

import (
  "fmt"
)

type notifier interface {
  notify()
}

type user struct {
  name string
  email string
}

func (u *user) notify() {
  fmt.Printf("Sending user email to %s<%s>\n", u.name, u.email)
}

func main() {
  u := user{"Bill", "bill@email.com"}
  sendNotification(u)
}

func sendNotification(n notifier) {
  n.notify()
}
```

However, the compilation would be failed. Because `notify` method
has pointer receiver. Method sets define the set of methods that
are associated with values or pointers of a given type. The type
of receiver used will determine whether a method is associated with
a value, pointer or both.

Let's start with explaining the rules for method sets as it's documented
by the Go specification.

| **Values** | **Methods Receivers** |
|:----------:|:---------------------:|
|      T     |         (t T)         |
|     *T     |    (t T) and (t *T)   |

A value of type `T` only has methods declared that have a value
receiver, as part of its method set. But pointer of type `T` have
methods declared with both value and pointer receivers, as
part of its method set.

The question now is why the restriction? The answer comes from
the fact that it's not always possible to get the address of a value.
