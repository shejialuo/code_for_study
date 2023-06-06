# Chapter 6 The Kotlin type system

## 6.1 Nullability

Nullability is a feature of the Kotlin type system that helps you avoid
`NullPointerException` errors. Kotlin converts these problems from runtime
errors into compile-time errors.

### 6.1.1 Nullable types

If you want to allow the use of this function with all arguments, including
those that can be `null`, you need to mark it explicitly by putting a
question mark after the type name:

```kotlin
fun strLenSafe(s: String?) = ...
```

### 6.1.2 Safe call operator "?."

One of the most useful tools in Kotlin's arsenal is the *safe-call* operator
`?.`, which allows you to combine a `null` check and a method call into a
single operation.

### 6.1.3 Elvis operator: "?:"

Kotlin has a handy operator to provide default values instead of `null`.
It's called the *Elvis operator*. It looks like this `?:`. Here's how it's used.

```kotlin
fun foo(s: String?) {
  val t: String = s ?: ""
}
```

### 6.1.4 Safe cases: "as?"

The `as?` operator tries to cast a value to the specified type and returns `null`
if the value doesn't have the proper type.

One common pattern of using a safe cast is combining it with the Elvis operator.
For example, this comes in handy for implementing the `equals` method.

```kotlin
class person(val firstName: String, val lastName: String) {
  override fun equals(o: Any?): Boolean {
    val otherPerson = o as? Person ?: return false

    return otherPerson.firstName == firstName && otherPerson.lastName == lastNam
  }

  override fun hashCode(): Int = firstName.hashCode() * 37 + lastName.hashCode()
}
```

### 6.1.5 Not-null assertions: "!!"

The *not-null assertion* is the simplest and bluntest tool Kotlin gives you for
dealing with a value of a nullable type. It's represented by a double exclamation
mark and converts any value to a non-`null` type. For `null` values, an exception
is thrown.

### 6.1.6 The "let" function

The `let` function makes it easier to deal with nullable expressions. Together with
a safe-call operator, it allows you to evaluate an expression, check the result
for `null` and store the result in a variable.

```kotlin
fun sendEmailTo(email: String) { /*...*/ }
```

You have to check explicitly whether this value isn't `null`:

```kotlin
if (email != null) sendEmailTo(email)
```

But you can go another way: use the `let` function, and call it via a safe call.
All the `let` function does is turn the object which it's called into a parameter
of the lambda.

```kotlin
email?.let{email -> sendEmailTo(email)}
email?.let{sendEmailTo(it)}
```

### 6.1.7 Late-initialized properties

Many frameworks initialize objects in dedicated methods called after the object
instance has been created. But you can't leave a non-`null` property without
an initializer in the constructor and only initialize it in a special method.

The method is simple. Define the field with `var`.

### 6.1.8 Extensions for nullable types

Define extension functions for nullable types is one more powerful way to deal
with `null` values. Rather than ensuring that a variable can't be `null` before
a method call, you can allow the calls with `null` as a receiver, and deal with
`null` in the function.

```kotlin
fun verifyUserInput(input: String?) {
    if (input.isNullOrBlank()) {
        println("Please fill in the required fields")
    }
}

fun String?.isNullOrBlank(): Boolean = this == null || this.isBlank()

fun main(args: Array<String>) {
    verifyUserInput(" ")
    verifyUserInput(null)
}
```

### 6.1.9 Nullability of type parameters

By default, all type parameters of functions and classes in Kotlin are nullable.
Any type, including a nullable type, can be substituted for a type parameter;
in this case, declarations using the type parameter as a type are allowed to be
`null`, even though the type parameter `T` doesn't end with a question mark.
Consider the following example.

```kotlin
fun <T> printHashCode(t: T) {
  println(t?.hashCode())
}
```

## 6.2 Primitive and other basic types

Unlike Java, Kotlin doesn't differentiate primitive types and wrappers.

### 6.2.1 Primitive types: Int, Boolean, and more

The full list of types that corresponding to Java primitive types is:

+ *Integer types*: `Byte`, `Short`, `Int`, `Long`.
+ *Floating-point number types*: `Float`, `Double`.
+ *Character type*: `Char`.
+ *Boolean type*: `Boolean`.

### 6.2.2 Nullable primitive types: Int?, Boolean?, and more

Nullable types in Kotlin can’t be represented by Java primitive types, because `null`
can only be stored in a variable of a Java reference type. That means whenever you use
a nullable version of a primitive type in Kotlin, it’s compiled to the corresponding
wrapper type.

### 6.2.3 Number conversions

One important difference between Kotlin and Java is the way they handle numeric
conversions. Kotlin doesn't automatically convert numbers from one type to
the other, even when the other type is larger. For example, the following code
won't compile in Kotlin:

```kotlin
val i = 1
val l: Long = i // Error: type mismatch
```

Instead, you need to apply the conversion explicitly:

```kotlin
val i = 1
val l: Long = i.toLong()
```

### 6.2.4 "Any" and "Any?": the root types

Similar to how `Object` is the root of the class hierarchy in Java, the
`Any` type is the supertype of all non-nullable types in kotlin. In Kotlin,
`Any` is a supertype of all types.

### 6.2.5 The Unit type: Kotlin's "void"

The `Unit` type in Kotlin fulfills the same function as `void` in Java.
It can be used as a return type of a function that has nothing interesting
to return.

### 6.2.6 The Nothing type: "This function never returns"

When analyzing code that calls such a function, it’s useful to know
that the function will never terminate normally. To express that,
Kotlin uses a special return type called `Nothing`:
