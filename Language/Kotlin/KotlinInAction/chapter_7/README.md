# Chapter 7 Operator overloading and other conventions

As a running example in this chapter, we’ll use a simple `Point` class,
representing a point on a screen.

```kotlin
data class Point(val x: Int, val y: Int)
```

## 7.1 Overloading arithmetic operators

### 7.1.1 Overloading binary arithmetic operators

The first operation you’re going to support is adding two points together.

```kotlin
data class Point(val x: Int, val y: Int) {
    operator fun plus(other: Point) : Point {
        return Point(x + other.x, y + other.y)
    }
}
```

As an alternative to declaring the operator as a member, you can define
the operator as an extension function.

```kotlin
operator fun Point.plus(other: Point) : Point {
    return Point(x + other.x, y + other.y)
}
```

Compared to some other languages, defining and using overloaded operators in
Kotlin is simpler, because you can’t define your own operators. Kotlin has
a limited set of operators that you can overload, and each one corresponds
to the name of the function you need to define in your class.

+ `a * b`: `times`
+ `a / b`: `div`
+ `a % b`: `mod`
+ `a + b`: `plus`
+ `a - b`: `minus`

### 7.1.2 Overloading compound assignment operators

Normally, when you define an operator such as `plus`, Kotlin supports not
only the `+` operation but `+=` as well.

If you define a function named `plusAssign` with the `Unit` return type,
Kotlin will call it when the `+=` operator is used. Other binary arithmetic
operators have similarly named counterparts: `minusAssign`, `timesAssign`,
and so on.

### 7.1.3 Overloading unary operators

The procedure for overloading a unary operator is the same as you saw
previously: declare a function with a predefined name, and mark it with
the modifier `operator`.

```kotlin
operator fun Point.unaryMinus() : Point {
    return Point(-x, -y)
}
```

+ `+a`: `unaryPlus`
+ `-a`: `unaryMinus`
+ `!a`: `not`
+ `++a`, `a++`: `inc`
+ `--a`, `a--`: `dec`

## 7.2 Overloading comparison operators

### 7.2.1 Equality operators: "equals"

Using the `==` operator in Kotlin is translated into a call of the `equals` method.
Using the `!=` operator is also translated into a call of `equals`, with the obvious
difference that the result is inverted.

```kotlin
class Point(val x: Int, val y: Int) {
    override fun equals(obj: Any?): Boolean {
      if (obj === this) return true
      if (obj !is Point) return false
      return obj.x == x && obj.y == y
    }
}
```

You use the *identity equals* operator (`===`) to check whether the parameter to
`equals` is the same object as the one on which `equals` is called.

### 7.2.2 Ordering operators: compareTo

In Java, classes can implement the `Comparable` interface in order to be used
in algorithms that compare values. The `compareTo` method defined in that
interface is used to determine whether one object is larger than another.
But in Java, there's no shorthand syntax for calling this method. Only values
of primitive types can be compared using `<` and `>`; all other types require
you to write `element1.compareTo(element2)` explicitly.

Kotlin supports the same `Comparable` interface. But the `compareTo` method
defined in that interface can be called by convention, and uses of comparison
operators (`<`, `>`, `<=`, and `>=`) are translated into calls of `compareTo`.

```kotlin
class Person(val firstName: String, val lastName: String
) : Comparable<Person> {
    override fun compareTo(other: Person): Int {
        return compareValuesBy(this, other, Person::lastName, Person::firstName)
    }
}
```

You can use `compareValuesBy` function from the Kotlin standard library to
implement the `compareTo` method easily and concisely. This function receives
a list of callbacks that calculate values to be compared.

## 7.3 Conventions used for collections and ranges

### 7.3.1 Accessing elements by index: "get" and "set"

You can access the elements in a map similarly to how you access array in
Java:

```kotlin
val value = map[key]
```

You can use the same operator to change the value for a key in a mutable map:

```kotlin
mutableMap[key] = newValue
```

In Kotlin, the index operator is one more convention. Reading an element using
the index operator is translated into a call of the `get` operator method, and
writing an element becomes a call to `set`. The methods are already defined for
the `Map` and `MutableMap` interfaces.

You'll allow the use of square brackets to reference the coordinates of the
point: `p[0]` to access the X coordinate and `p[1]` to access the Y coordinate.
Here's how to implement and use it.

```kotlin
operator fun Point.get(index: Int) : Int {
    return when(index) {
        0 -> x
        1 -> y
        else ->
            throw IndexOutOfBoundsException("Invalid coordinate $index")
    }
}
```

Note that the parameter of `get` can be any type, not just `Int`.

### 7.3.2 The "in" convention

One other operator supported by collections is the `in` operator, which is used to
check whether an object belongs to a collection. The corresponding function is
called `contains`.

```kotlin
data class Rectangle(val upperLeft: Point, val lowerRight: Point)

operator fun Rectangle.contains(p: Point): Boolean {
    return p.x in upperLeft.x until lowerRight.x &&
            p.y in upperLeft.y until lowerRight.y
}
```

### 7.3.3 The rangeTo convention

To create a range, you use the `..` syntax: for instance, `1..10` enumerates all
the numbers from 1 to 10.

The `rangeTo` function returns a range. You can define this operator for your own
class. but if you class implements the `Comparable` interface, you don't need that:
you can create a range of any comparable elements by means of Kotlin standard

### 7.3.4 The "iterator" convention for the "for" loop

A statement such as `for (x in list) {...}` will be translated into a call of
`list.iterator()`, on which the `hasNext` and `next` methods are then repeatedly
called.

## 7.4 Destructuring declarations and component functions

A destructuring declaration looks like a regular variable declaration, but
it has multiple variables grouped in parentheses. Under the hood, the
destructuring declaration, a function named `componentN` is called, where
$N$ is the position of the variable in the declaration.

```kotlin
class Point(val x: Int, val y: Int) {
    operator fun component1() = x
    operator fun component2() = y
}
```
