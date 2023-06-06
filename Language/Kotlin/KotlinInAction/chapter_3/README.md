# Chapter 3 Defining and calling functions

## 3.1 Creating collections in Kotlin

```kotlin
val set = hashSetOf(1, 7, 53)
val list = arrayListOf(1, 7, 53)
val map = hashMapOf(1 to "one", 7 to "seven", 53 to "fifty-three")
```

## 3.2 Making functions easier to call

```kotlin
fun <T> joinToString(
        collection: Collection<T>,
        separator: String,
        prefix: String,
        postfix: String
        ): String {
    val result = StringBuilder(prefix)
    for ((index, element) in collection.withIndex()) {
        if (index > 0) result.append(separator)
        result.append(element)
    }
    result.append(postfix)
    return result.toString()
}
```

### 3.2.1 Named arguments

With Kotlin, you can do better

```kotlin
joinToString(collection, separator = " ", prefix = " ", postfix = ".")
```

### 3.2.2 Default parameter values

In Kotlin, you can often avoid creating overloads because you can specify default
values for parameters in a function declaration.

```kotlin
fun <T> joinToString(
    collection: Collection<T>,
    separator: String = ", ",
    prefix: String = "",
    postfix: String = ""
): String
```

### 3.2.3 Getting rid of static utility classes: top-level functions and properties

In Kotlin, you don't need to create all those meaningless classes. Instead, you
can place functions directly at the top level of a source file, outside of any class.

## 3.3 Adding methods to other people's classes: extension functions and properties

An *extension function* is a simple thing: it's a function that can be called
as a member of a class but is defined outside of it.

```kotlin
fun String.lastChar(): Char = this.get(this.length - 1)A

val String.lastChar: Char
    get() = get(length - 1)
```

## 3.4 Working with collections: varargs, infix calls, and library support

### 3.4.1 Varargs: functions that accept an arbitrary number of arguments

Kotlin uses the `varargs` modifier on the parameter. And it could use the `*`
to unpack the `varargs` parameter.

```kotlin
fun main(args: Array<String>) {
    val list = listOf("args: ", *args)
    println(list)
}
```

### 3.4.2 Working with pairs: infix calls and destructuring declarations

To create maps, you use the `mapOf` function:

```kotlin
val map = mapOf(1 to "one", 7 to "seven", 53 to "fifty-three")
```

The `to` isn't a built-in construct, but rather a method invocation of
a special kind, called an *infix call*.

In an infix call, the method name is placed immediately between the target
object name and the parameter, with no extra separators.

```kotlin
1.to("one")
1 to "one
```

To allow a function to be called using the infix notation, you need to
mark it with the `infix` modifier.

```kotlin
infix fun Any.to(other: Any) = Pair(this, other)
```
