# Chapter 4 Classes, objects, and interfaces

## 4.1 Defining class hierarchies

### 4.1.1 Interfaces in Kotlin

Kotlin interfaces are similar to those of Java 8: they can contain definitions
of abstract methods as well as implementations of non-abstract methods.

```kotlin
interface Clickable {
  fun click()
}
```

Here's how you implement the interface.

```kotlin
class Button : Clickable {
  override fun click() = println("I was clicked")
}
```

Kotlin uses the colon after the class name to replace both the `extends` and
`implements` keywords used in Java.

The `override` modifier, similar to the `@Override` annotation in Java, is used
to mark methods and properties that override those from the superclass or interface.
Unlike Java, *using* the `override` modifier is mandatory in Kotlin.

An interface method can have a default implementation. Unlike Java 8, which requires
you to mark such implementations with the `default` keyword, Kotlin has no special
annotation for such methods: you just provide a method body.

```kotlin
interface Clickable {
  fun click()
  fun showOff() = println("I'm clickable")
}
```

Let's suppose now that another interface also defines a `showOff` method and has
the following implementation for it.

```kotlin
interface Focusable {
    fun setFocus(b: Boolean) =
        println("I ${if (b) "got" else "lost"} focus.")
    fun showOff() = println("I'm focusable!")
}
```

What happens if you need to implement both interfaces in your class? Each of them
contains a `showOff` method with a default implementation; which implementation
wins? Neither one wins, you should provide your own implementation

```kotlin
class Button : Clickable, Focusable {
    override fun click() = println("I was clicked")
    override fun showOff() {
        super<Clickable>.showOff()
        super<Focusable>.showOff()
    }
}
```

### 4.1.2 Open, final, and abstract modifiers: final by default

As you know, Java allows you to create subclasses of any class, and to override
any method, unless it has been explicitly marked with the `final` keyword. Kotlin
classes and methods are `final` by default.

If you want to allow the creation of subclasses of a class, you need to mark
the class with the `open` modifier. In addition, you need to add the `open`
modifier to every property or method can be overridden.

```kotlin
open class RichButton : Clickable {
    fun disable() {}
    open fun animate() {}
    override fun click() {}
}
```

Note that if you override a number of a base class or interface, the overriding
member will also be `open` by default. If you want to change this and forbid
the subclasses of your class from overriding your implementation, you can explicitly
mark the overriding member as `final`.

```kotlin
open class RichButton : Clickable {
    final override fun click() {}
}
```

In kotlin, you may declare a class `abstract`. Abstract members are always open, so
you don't need to use an explicit `open` modifier.

```kotlin
abstract class Animated {
    abstract fun animate()

    open fun stopAnimating() {}

    fun animateTwice() {}
}
```

### 4.1.3 Visibility modifiers: public by default

Kotlin uses packages only as a way of organizing code in namespaces; it doesn't use them
for visibility control.

As an alternative, Kotlin offers a new visibility modifier, `internal`, which means
"visible inside a module." A *module* is a set of Kotlin files compiled together.

The advantage of `internal` visibility is that it provides real encapsulation for
the implementation details of your module.

Another difference is that Kotlin allows the use of `private` visibility for top-level
declarations, including classes, functions, and properties. Such declarations are
visible only in th file where they are declared.

### 4.1.4 Inner and nested classes: nested by default

In Kotlin you can declare a class in another class. Doing so can be useful for
encapsulating a helper class or placing the code closer to where it's used. The
difference is that Kotlin nested classes don't have access to the outer class instance,
unless you specifically requests that.

A nested class in Kotlin with no explicit modifiers is the same as a `static` nested
class in Java. To turn it into an inner class so that it contains a reference to
an outer class, you use the `inner` modifier.

The syntax to reference an instance of an outer class in Kotlin also differs from
Java. You write `this@outer` to access the `Outer` class from the `Inner` class.

### 4.1.5 Sealed classes: defining restricted class hierarchies

Kotlin provides `sealed` classes. You mark a superclass with the `sealed` modifier,
and that restricts the possibility of creating subclasses. All the direct subclasses
must be nested in the superclass.

```kotlin
sealed class Expr {
    class Num(val value: Int) : Expr()
    class Sum(val left: Expr, val right: Expr): Expr()
}

fun eval(e: Expr): Int =
    when (e) {
        is Expr.Num -> e.value
        is Expr.Sum -> eval(e.left) + eval(e.right)
    }
```

## 4.2 Declaring a class with nontrivial constructors or properties

Kotlin can declared one or more constructors and it makes a distinction between
a *primary* constructor and a *secondary* constructor. It also allows you to
put additional initialization logic in *initializer blocks*.

### 4.2.1 Initializing classes: primary constructor and initialize blocks

```kotlin
class User(val nickname: String)
```

Normally, all the declarations in a class go inside curly braces. Above block
of code surrounded by parentheses is called a *primary constructor*. It serves
two purposes: specifying constructor parameters and defining properties that
are initialized by those parameters. Let's unpack what happens here and look
at the most explicit code you can write that does the same thing.

```kotlin
class User constructor(_nickname: String) {
    val nickname: String

    init {
        nickname = _nickname
    }
}
```

The `constructor` keyword begins the declaration of a primary or a secondary
constructor. The `init` keyword introduces an *initializer block*. Such blocks
contain initialization code that's executed when the class is created, and
are intended to be used together with primary constructors.

However, you can omit the `constructor` keyword if there are no annotations
or visibility modifiers on the primary constructor. If you apply those changes,
you get the following:

```kotlin
class User(_nickname: String) {
  val nickname = _nickname
}
```

If your class has a superclass, the primary constructor also needs to initialize the
superclass. You can do so by providing the superclass constructor parameters after the
superclass reference in the base class list:

```kotlin
open class User(val nickname: String) { ... }
class TwitterUser(nickname: String) : User(nickname) { ... }
```

If you want to ensure that your class can't be instantiated by other code, you have to
make the constructor `private`. Here's how you make the primary constructor `private`:

```kotlin
class Secretive private constructor() {}
```

### 4.2.2 Secondary constructors: initializing the superclass in different ways

A secondary constructor is introduced using the `constructor` keyword. You can
declare as many secondary constructors as you need.

```kotlin
open class View {
    constructor(ctx: Context) {

    }

    constructor(ctx: Context, attr: AttributeSet) {

    }
}
```

### 4.2.3 Implementing properties declared in interfaces

In Kotlin, an interface can contain abstract property declarations.

```kotlin
interface User {
    val nickname: String
}
```

The interface doesn't specify whether the value should be stored in a backing
field or obtained through a getter. Therefore, the interface itself doesn't
contain any state, and only classes implementing the interface may store the
value if they need to.

```kotlin
class SubscribingUser(val email: String) : User {
    override val nickname: String
        get() = email.substringBefore('@')
}
```

In additional to abstract property declarations, an interface can contain
properties with getters and setters, as long as they don't reference a
backing field.

```kotlin
interface User {
    val email: String
    val nickname: String
        get() = email.substringBefore('@')
}
```

### 4.2.4 Accessing a backing field from a getter or setter

```kotlin
class User(val name: String) {
    var address : String = "unspecified"
        set(value: String) {
            println("""
                Address was changed for $name:
                "$field" -> "$value".
            """.trimIndent())
        }
}

fun main() {
    val user = User("Alice")
    user.address = "Elsenheimerstrasse 47, 80687 Muenchen"
}
```

In the body of the setter, you use the special identifier `field` to
access the value of the backing field.

The compiler will generate the backing field for the property if you either
reference it explicitly or use the default accessor implementation. If you
provide custom accessor implementations that don’t use `field` (for the
getter if the property is a val and for both accessors if it’s a mutable property), the
backing field won’t be present.

### 4.2.5 Changing accessor visibility

The accessor's visibility by default is the same as the property's. But you
can change this if you need to, by putting a visibility modifier before the
`get` or `set` keyword.

```kotlin
class LengthCounter {
    var counter: Int = 0
        private set

    fun addWord(word: String) {
        counter += word.length
    }
}
```

## 4.3 Compiler-generated methods: data classes and class delegation

### 4.3.1 Universal object methods

All kotlin classes have several methods you may want to override: `toString`,
`equals`, and `hashCode`.

### 4.3.2 Data classes: autogenerated implementations of universal methods

You don’t have to generate all of these methods in Kotlin. If you add the
modifier data to your class, the necessary methods are automatically
generated for you.

```kotlin
data class Client(val name: String, val postalCode: Int)
```

To make it even easier to use data classes as immutable objects, the Kotlin compiler
generates one more method for them: a method that allows you to *copy* the instances
of your classes, changing the values of some properties.

## 4.4 The "object" keyword

The `object` keyword comes up in Kotlin in a number of cases, but they all share
the same core idea: the keyword defines a class and creates an instance of that
class at the same time.

### 4.4.1 Object declarations: singletons made easy

A fairly common occurrence in the design of object-oriented systems is a class
for which you need only one instance.

### 4.4.2 Companion objects: a place for factory methods and static members

Classes in Kotlin can't have static members; Java's `static` keyword isn't part of
the Kotlin language. As a replacement, Kotlin relies on package-level functions
and object declarations.

One of the objects defined in a class can be marked with a special keyword:
`companion`. If you do that, you gain the ability to access the methods and
properties of that object directly through the name of the containing class,
without specifying the name of the object directly.

```kotlin
class A {
    companion object {
        fun bar() {
            println("Companion object called")
        }
    }
}

fun main(args : Array<String>) {
    A.bar()
}
```
