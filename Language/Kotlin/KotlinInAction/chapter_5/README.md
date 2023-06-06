# Chapter 5 Programming with lambdas

## 5.1 Lambda expressions and member references

### 5.1.1 Lambdas and collections

Let’s look at an example. You’ll use the `Person` class that contains information
about a person’s name and age.

```kotlin
data class Person(val name: String, val age: Int)
```

In Kotlin, there’s a better way to find the max age:

```kotlin
val people = listOf(Person("Alice", 29), Person("Bob", 31))
println(people.maxBy { it.age })
```

The code in curly braces `{ it.age }` is a lambda implementing that logic.
It receives a collection element as an argument (referred to using `it`) and
returns a value to compare.

If a lambda just delegates to a function or property, it can be replaced
by a member reference.

```kotlin
people.maxBy(Person::age)
```

### 5.1.2 Syntax for lambda expressions

A lambda expression in Kotlin is always surrounded by curly braces. Note that
there are no parentheses around the arguments. The arrow separates the
argument list from the body of the lambda.

```kotlin
val sum = {x: Int, y: Int -> x + y}
run { println(42) }
```

We rewrite the `Person` example:

```kotlin
people.maxBy({p: Person -> p.age})
```

In Kotlin, a syntactic convention lets you move a lambda expression out of
parentheses if it's the last argument in a function call. So:

```kotlin
people.maxBy() {p: Person -> p.age}
```

When the lambda is the only argument to a function, you can also remove the
empty parentheses from the call:

```kotlin
people.maxBy {p: Person -> p.age}
```

As with local variables, if the type of a lambda parameter can be inferred,
you don’t need to specify it explicitly.

```kotlin
people.maxBy {p -> p.age}
```

The last simplification you can make in this example is to replace a parameter with
the default parameter name: `it`. This name is generated if the context expects
a lambda with only one argument, and its type can be inferred.

### 5.1.3 Accessing variables in scope

If you use a lambda in a function, you can access the parameters of that function
as well as the local variables declared before the lambda

```kotlin
fun printMessageWithPrefix(messages: Collection<String>, prefix: String) {
    message.forEach {
      println("$prefix $it")
    }
}
```

However, you can modify the variable also.

```kotlin
fun printProblemCounts(responses: Collection<String>) {
    var clientErrors = 0
    var serverErrors = 0
    responses.forEach {
        if (it.startsWith("4")) {
            clientErrors++
        } else if (it.startsWith("5")) {
            serverErrors++
        }
    }
    println("$clientErrors client errors, $serverErrors server errors")
}
```

By default, the lifetime fo a local variable is constrained by the function
in which the variable is declared. But if it's captured by the lambda, the
code that uses this variable can be stored and executed later. When you
capture a final variable, its value is stored together with the lambda code
that uses it. For non-final variables, the value is enclosed in a special
wrapper that lets you change it, and the reference to the wrapper is stored
together with the lambda.

### 5.1.4 Member references

What if the code that you need to pass as a parameter is already defined
as a function? In Kotlin, you can do so if you convert the function to
a value. You use the `::` operator for that.

```kotlin
val getAge = Person::age
```

This expression is called *member inference*, and it provides a short
syntax for creating a function value that calls exactly one method or
access a property.

You can have a reference to a function that's declared at the top level
(`::`).

## 5.2 Functional APIs for collections

Omit detail here.

## 5.3 Lazy collection operations: sequences

The entry point for lazy collection operations in Kotlin is the `Sequence`
interface. The interface represents just that: a sequence of elements
that can be enumerated one by one. `Sequence` provides only one method,
`iterator`, that you can use to obtain the values from the sequence.

You can convert any collection to a sequence by calling the extension
function `asSequence`.

### 5.3.1 Executing sequence operations: intermediate and terminal operations

Operations on a sequence are divided into two categories: intermediate and terminal.
An *intermediate operation* returns another sequence, which knows how to transform
the elements of the original sequence. A *terminal operation* returns a result,
which may be a collection, an element, a number or any other object that's somehow
obtained by the sequence of transformations of the initial collection.

Intermediate operations are always lazy.

### 5.3.2 Creating sequences

Another possibility is to use the `generateSequence` function. This function calculates
the next element in a sequence given the previous one.

```kotlin
fun main(args : Array<String>) {
    val naturalNumbers = generateSequence (0) { it + 1  }
    val numbersTo100 = naturalNumbers.takeWhile { it <= 100 }
    println(numbersTo100.sum())
}
```

## 5.4 Using Java functional interfaces

Omit detail here.

## 5.5 Lambdas with receivers: "with" and "apply"

### 5.5.1 The "with" function

```kotlin
fun alphabet(): String {
    val result = StringBuilder()
    for (letter in 'A'..'Z') {
        result.append(letter)
    }
    result.append("\nNow I know the alphabet!")
    return result.toString()
}

fun main(args : Array<String>) {
    println(alphabet())
}
```

Here's how you can rewrite the code using `with`:

```kotlin
fun alphabet(): String {
    val stringBuilder = StringBuilder()
    return with(stringBuilder) {
        for (letter in 'A'..'Z') {
            this.append(letter)
        }
        append("\nNow I know the alphabet!")
        this.toString()
    }
}

fun main(args : Array<String>) {
    println(alphabet())
}
```

The `with` is a function takes two arguments: `stringBuilder`, in this case,
and a lambda. The `with` function converts its first argument into a *receiver*
of the lambda that's passed as a second argument. You can access this receiver
via an explicit `this` reference. Alternatively, as usual for a `this` reference,
you can omit it and access methods or properties of this value without any
additional qualifiers.

Let’s refactor the initial `alphabet` function even further and get rid of the extra
`stringBuilder` variable.

```kotlin
fun alphabet() = with(StringBuilder()) {
    for (letter in 'A'..'Z') {
        append(letter)
    }
    append("\nNow I know the alphabet!")
    toString()
}
```

### 5.5.2 The "apply" function

The `apply` function works almost exactly the same as `with`; the only difference
is that `apply` always returns the object passed to it as an argument. Let's
refactor the `alphabet` function again, this time using `apply`.

```kotlin
fun alphabet() = StringBuilder().apply {
    for (letter in 'A'..'Z') {
        append(letter)
    }
    append("\nNow I know the alphabet!")
}.toString()
```
