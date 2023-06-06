# Chapter 2 Kotlin basics

## 2.1 Basic elements: functions and variables

### 2.1.1 Hello, world

```kotlin
fun main(args: Array<String>) {
    println("Hello, world!")
}
```

### 2.1.2 Functions

```kotlin
fun max(a: Int, b: Int): Int {
    return if (a > b) a else b
}
```

```kotlin
fun max(a: Int, b: Int): Int = if (a > b) a else b
```

### 2.1.3 Variables

+ `val`: Immutable reference.
+ `var`: Mutable reference.

### 2.1.4 Easier string formatting: string templates

```kotlin
fun main(args: Array<String>) {
    val name = if (args.size > 0) args[0] else "Kotlin"
    println("Hello, $name!")
}
```

## 2.2 Classes and properties

```java
public class Person {
    private final String name;
    public Person(String name) {
        this.name = name;
    }
    public String getName() {
        return name;
    }
}
```

```kotlin
class Person(val name: String)
```

## 2.3 Representing and handling choices: enums and "when"

### 2.3.1 Declaring enum classes

```kotlin
enum class Color {
  RED, ORANGE, YELLOW, GREEN, BLUE, INDIGO, VIOLET
}
```

Just as in Java, enums aren’t lists of values: you can declare properties and methods
on enum classes. Here’s how it works.

```kotlin
enum class Color(val r: Int, val g: Int, val b: Int) {
    RED(255, 0, 0), ORANGE(255, 165, 0),
    YELLOW(255, 255, 0), GREEN(0, 255, 0), BLUE(0, 0, 255),
    INDIGO(75, 0, 130), VIOLET(238, 130, 238);

    fun rgb() = (r * 256 + g) * 256 + b
}
```

### 2.3.2 Using "when" to deal with enum classes

```kotlin
fun getMnemonic(color: Color) =
    when (color) {
        Color.RED -> "Richard"
        Color.ORANGE -> "Of"
        Color.YELLOW -> "York"
        Color.GREEN -> "Gave"
        Color.BLUE -> "Battle"
        Color.INDIGO -> "In"
        Color.VIOLET -> "Vain"
    }
```

```kotlin
fun getWarmth(color: Color) = when(color) {
    Color.RED, Color.ORANGE, Color.YELLOW -> "warm"
    Color.GREEN -> "neutral"
    Color.BLUE, Color.INDIGO, Color.VIOLET -> "cold"
}
```

### 2.3.3 Using "when" with arbitrary objects

The `when` construct allows any objects.

```kotlin
fun mix(c1: Color, c2: Color) =
    when (setOf(c1, c2)) {
        setOf(RED, YELLOW) -> ORANGE
        setOf(YELLOW, BLUE) -> GREEN
        setOf(BLUE, VIOLET) -> INDIGO
        else -> throw Exception("Dirty color")
    }
```

### 2.3.4 Using "when" without an argument

The above program is inefficient.

```kotlin
fun mixOptimized(c1: Color, c2: Color) =
when {
    (c1 == RED && c2 == YELLOW) ||
    (c1 == YELLOW && c2 == RED) ->
        ORANGE
    (c1 == YELLOW && c2 == BLUE) ||
    (c1 == BLUE && c2 == YELLOW) ->
        GREEN
    (c1 == BLUE && c2 == VIOLET) ||
    (c1 == VIOLET && c2 == BLUE) ->
        INDIGO
    else -> throw Exception("Dirty color")
}
```

### 2.3.5 Smart casts: combining type checks and casts

```kotlin
interface Expr
class Num(val value: Int) : Expr
class Sum(val left: Expr, val right: Expr) : Expr

fun eval(e: Expr): Int =
    when (e) {
        is Num -> e.value
        is Sum -> eval(e.right) + eval(e.left)
        else ->
        throw IllegalArgumentException("Unknown expression")
    }
```
