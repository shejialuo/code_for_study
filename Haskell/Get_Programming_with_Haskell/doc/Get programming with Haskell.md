# Get programming with Haskell

## Unit 1

### Function Principle

All functions in Haskell follow three rules that force them to behave like
functions in math:

+ All functions must take an argument
+ All functions must return a value.
+ Anytime a function is called with the same argument, it must return the same value.

The third rule is part of the basic mathematical definition of a function. When
the rule that the same argument must always produce the same result is applied
to function in a programming language, it's called **referential transparency**.

### Application of Variable

The key benefit of variables in programming is to clarify your code and avoid repetition.
For example:

```haskell
calcChange owed given = if given - owed > 0
                        then given - owed
                        else 0
```

Two things are wrong with this function:

+ Even for a tiny function, it's hard to read. Each time you see the
expression `given - owed`, you have to reason about what's happening.
+ You’re repeating your computation!

Haskell solves these problems by using a special `where` clause.

```haskell
calcChange owed given = if change > 0
                        then change
                        else 0
  where change = given - owed
```

### Lambda Function

One of the most foundational concepts in functional programming is a function
without a name, called a **lambda function**. To do this, you use Haskell's
lambda syntax.

```haskell
\x -> x
```

### Writing Own where Clause

It turns out the lambda function on its own is powerful enough to create
variables from nothing. To start, you'll look at a function that uses a `where` statement:

```haskell
--sumSquareOrSquareSum v.1

sumSquareOrSquareSum x y = if sumSquare > squareSum
                           then sumSquare
                           else squareSum
  where sumSquare = x^2 + y^2
        squareSum = (x+y)^2
```

Without a `where`, you could just replace the variables, but then you're doubling
computation and the code is ugly:

```haskell
sumSquareOrSquareSum x y = if (x^2 + y^2) > ((x+y)^2)
                           then x^2 + y^2
                           else (x+y)^2
```

One solution to not having variables is to split your function into two steps:

```haskell
body sumSquare squareSum = if sumSquare > squareSum
                           then sumSquare
                           else squareSum

sumSquareOrSquareSum x y = body (x^2 + y^2) ((x+y)^2)
```

This solves the problem but adds a lot of work, and you need to define a new
, intermediate function `body`. This is a such simple function that it'd be
nice if you didn't need an intermediate function:

```haskell
sumSquareOrSquareSum x y = (\sumSquare squareSum ->
        if sumSquare > squareSum
        then sumSquare
        else squareSum) (x^2 + y^2) ((x+y)^2)
```

### Let

Haskell has an alternative to `where` clauses called `let` expressions. A `let`
expression allows you to combine the readability of a `where` clause with
the power of your lambda function.

```haskell
sumSquareOrSquareSum x y = let sumSquare = (x^2 + y^2)
                               squareSum = (x+y)^2
                           in
                             if sumSquare > squareSum
                             then sumSquare
                             else squareSum
```

Also, you can use `let` to overwrite variables:

```haskell
overwrite x = let x = 2
              in
                let x = 3
                in
                  let x = 4
                  in
                    x
```

### List

To build a list, you need just one function and the infix operator `:`,
which is called cons. This term is short for construct and has
its origins in Lisp.

To make a list, you need to take a value and cons it with another list.
The simplest way to make a list is to cons a value with empty list:

```haskell
1:[]
```

Under the hood, all lists in Haskell are represented as a bunch of `:`
operations, and the `[...]` notation is syntactic sugar. An important
thing to remember is that in Haskell every element of the list must
be the same type.

### Recursion

The secret to writing recursive functions is to not think about the
recursion. The way to solve recursive functions is by following this
simple set of rules:

+ Identify the end goal.
+ Determine what happens when a goal is reached.
+ List all alternate possibilities.
+ Determine your "rinse and repeat" process.
+ Ensure that each alternative moves you toward your goal.

### Pattern Matching

First, let's see how to implement pattern matching by using `case`.

```haskell
sayAmount n = case n of
  1 -> "one"
  2 -> "two"
  3 -> "a bunch"
```

The pattern matching version of this looks like three separate
definitions, each for one of the possible arguments.

```haskell
sayAmount 1 = "one"
sayAmount 2 = "two"
sayAmount n = "bunch"
```

The important thing to realize about pattern matching is that it
can look only at arguments, but it can't do any computation on
them when matching. In Haskell, it's standard practice to use `_`
as wildcard for values you don't use.

### Higher-Order Functions

The `map` function takes another function and a list as arguments
and applies that function to each element in the list:

```haskell
map reverse ["dog", "cat", "moose"]
map head ["dog", "cat", "moose"]
map (take 4) ["pumpkin", "pie", "peanut butter"]
```

You can define your own `map`:

```haskell
myMap f [] = []
myMap f (x:xs) = (f x) : myMay xs
```

The `filter` function is used to filter just like `map`.

```haskell
filter even [1,2,3,4]
filter (\(x:xs) -> x == 'a') ["apple","banana","avocado"]
```

The function `foldl` takes a list and reduces it to a single value.
The function takes three arguments: a binary function, an initial value,
and a list. The most common use of `foldl` is to sum a list:

```haskell
foldl (+) 0 [1,2,3,4]
```

## Unit 2

### Type Basics

Haskell uses **type inference** to automatically determine the types of all
values at compile time based on the way they're used. Below shows a
variable that you'll give the `Int` type.

```haskell
x :: Int
```

All types in Haskell start with a capital letter to distinguish them from
functions. Haskell support all the types that you're likely familiar with
in other languages. Here are some examples.

```haskell
letter :: Char
letter = 'a'

interestRate :: Double
interestRate = 0.375

isFun :: Bool
isFun = True
```

Functions also have type signatures. In Haskell an `->` is used to separate
arguments and return values. The type signature for `double` looks like below:

```haskell
double :: Int -> Int
double n = n * 2

makeAddress :: Int -> String -> String -> (Int, String, String)
makeAddress number street town = (number, street, town)

ifEven :: (Int -> Int) -> Int -> Int
ifEven f n = if even n
             then f n
             else n
```

### Type Variable

Haskell has type variables. Any lowercase letter in a type signature indicates
that any type can be used in that place. The type definition for `simple` looks
like the following:

```haskell
simple :: a -> a
simple x = x
```

Type variables are literally variables for types. Type variables work exactly
like regular variables, but instead of representing a value, they represent a
type. When you use a function that has a type variable in its signature, you
can imagine Haskell substituting the variable that's needed.

Type signatures can contain more than one type of variable.

```haskell
makeTriple :: a -> b -> c -> (a,b,c)
makeTriple x y z = (x,y,z)
```

### Type Synonyms

When you have two names for the same type, it's referred to as a
**type synonyms**. Type synonyms are extremely useful, because they make reading
type signatures much easier. In Haskell, you can create new type synonyms by
using the `type` keyword. Here's the code to create the type synonyms you'd like.

```haskell
type FirstName = String
type LastName = String
type Age = Int
type Height = Int
```

### Creating New Types

Creating a new type can be done with the `data` keyword.

```haskell
data Sex = Male | Female
-- Sex is the type constructor
-- Male | Femal is The data constructors
```

In this new type, you define a key pieces. The `data` keyword tells Haskell
that you're defining a new type. The word `Sex` is the type constructor.
In this case, the type constructor is the name of the type, but the type
constructors can take arguments. A data constructor is used to create a
concrete instance of the type.

It turns out that `Bool` in Haskell is defined exactly the same way:

```haskell
data Bool = True | False
```

### Using Record Syntax

```haskell
data Sex = Male | Female
type Age = Int
type Height = Int
type Weight = Int

data Person = Person Sex Age Height Weight

showSexual :: Sex -> [Char]
showSexual Male = "Male"
showSexual Female = "Female"


getSexual :: Person -> [Char]
getSexual (Person sex _ _ _) = showSexual sex

getAge :: Person -> Age
getAge (Person _ age _ _) = age

getHeight :: Person -> Height
getHeight (Person _ _ height _) = height

getWeight :: Person -> Weight
getWeight (Person _ _ _ weight) = weight
```

As you can see, pattern matching makes these getters wonderfully easy to
write,but having to write out all six of them seems annoying. So to solve
this question, you can define data types by using **record syntax**.

```haskell
data Sex = Male | Female

data Person = Person { sex :: Sex,
                       age :: Int,
                       height :: Int,
                       weight :: Int}
```

The first victory for record syntax is that your type definition is much
easier to read and understand now. The next big win for record syntax is
that creating data is much easier.

```haskell
jackieSmith = Person { sex = Female,
                       age = 43,
                       height = 62,
                       weight = 115}
```

In addition, you don't have to write your getters; each field in the record
syntax automatically creates a function to access that value from the record:

```haskell
height jackieSmith
```

You can also set values in record syntax by passing the new value in curly
brackets to you data.

```haskell
jackieSmithUpdated = jackieSmith { age = 44 }
```

Because you're still in a purely functional world, a new type will be created
and must be assigned to a variable to be useful.

### Type Class

Type classes in Haskell are way of describing groups of types that all behave
in the same way. For example, `Num` is a type class generalizing the idea of a
number. All things of class `Num` must have a function `(+)` defined on them.

We can use `class` keyword to define a type class:

```haskell
class TypeName a where
  func1 :: a -> a
  func2 :: a -> String
  func3 :: a -> a -> Bool
-- TypeName is the name of the type class
-- a is the type variable as a placeholder
```

Haskell defines many type classes for your convenience:

```haskell
class Eq a => Ord a where
  compare :: a -> a -> Ordering 
  (<) :: a -> a -> Bool
  (<=) :: a -> a -> Bool
  (>) :: a -> a -> Bool
  (>=) :: a -> a -> Bool
  max :: a -> a -> a
  min :: a -> a -> a
```

Notice that right in the class definition there's another type class.

```haskell
class Eq a where
  (==) :: a -> a -> Bool
  (/=) :: a -> a -> Bool
```

This explains why the `ord` type class includes the `Eq` type class in its definition.

When you define a type, Haskell can do its best to automatically derive a type
class. For example:

```haskell
data Icecream = Chocolate | Vanilla deriving (Show)
```

### Using Type Class

```haskell
data SixSidedDie = S1 | S2 | S3 | S4 | S5 | S6

instance Show SixSidedDie where
  show S1 = "one"
  show S2 = "two"
  show S3 = "three"
  show S4 = "four"
  show S5 = "five"
  show S6 = "six"
```

One question that might come up is, why do you have to define `show` this way?
Why do you need to declare an instance of  a type class? Surprisingly, if you
remove your early instance declaration, the following code will compile just fine.

```haskell
show :: SixSidedDie -> String
show S1 = "one"
show S2 = "two"
show S3 = "three"
show S4 = "four"
show S5 = "five"
show S6 = "six"
```

However, if you use this way, there is no polymorphism.

## Unit 3

### Combining Functions

A special higher-order function that's just a period (called **compose**) takes two
functions as arguments. Using function composition is particularly helpful for
combining functions on the fly in a readable way. Here are some examples:

```haskell
myLast :: [a] -> a
myLast = head . reverse
myMin :: Ord a => [a] -> a
myMin = head . sort
myMax :: Ord a => [a] -> a
myMax = myLast . sort
```

### Using Guard

```haskell
howMuch :: Int -> String
howMuch n | n > 10 = "a whole bunch"
          | n > 0  = "not much"
          | otherwise = "we're in debt"
```

### Semigroup

The `Semigroup` class has only one important method you need, the `<>` operator.
For example:

```haskell
instance Semigroup Integer where
  (<>) x y = x + y
```

### Monoid

```haskell
class Monoid a where
 mempty :: a
 mappend :: a -> a -> a
 mconcat :: [a] -> a
```

The most common `Monoid` is a list.

```haskell
[1,2,3] ++ []
[1,2,3] <> []
[1,2,3] `mappend` mempty
```

### Parameterized Type

The most basic parameterized type you could make is a `Box` that serves a
container for any other type:

```haskell
data Box a = Box a deriving Show
```

The `Box` type is an abstract container that can hold any other type.

The most common parameterized type is a `List`. The `List` type is interesting
because it has a different constructor:

```haskell
data [] a = [] | a:[a]
```

### Kinds

Another thing that Haskell's types have in common with functions and data is
that they have their own types as well. The type of a type is called its **kind**.
The kind of a type indicates the number of parameters the type takes,
which are expressed using an asterisk (`*`).

+ Types that take no parameters have a kind of `*`.
+ Types  that take one parameter have the kind `* -> *`.
+ Types with two parameters have the kind `* -> * -> *`, and so forth.

### Maybe

The definition of `Maybe` is simple.

```haskell
data Maybe a = Nothing | Just a
```

## Unit 4

Haskell has a special parameterized type called `IO`. Any value in an `IO` context
must stay in this context. This prevents code that's pure and code that's
necessarily impure from mixing.

```haskell
helloPerson :: String -> String
helloPerson name = "Hello" ++ " " ++ name ++ "!"
main :: IO () 
main = do
 putStrLn "Hello! What's your name?"
 name <- getLine
 let statement = helloPerson name
 putStrLn statement
```

### IO Type

`IO` in Haskell is a parameterized type that's similar to `Maybe`. The first thing
they share in common is that they're parameterized types of the  same kind.
The other thing that `Maybe` and `IO` have in common is that they describe a context
for their parameters rather than a container.

For `main::IO()`, `()` may seem like a special symbol, but in reality, it's just
a tuple of zero elements. So actually the `main` returns nothing, it simply
performs an **action**.

### Do-notation

The **do-notation** allows you to treat `IO` types as if they were regular types.
This also explains why some variables use `let` and others use `<-`. Variables assigned
with `<-` allow you to act as though a type `IO a` is a just of type `a`.
You use `let` statements whenever you create variables that aren't `IO` types.

## Unit 5

In this unit, you'll take a look at three of Haskell's most powerful and often most
confusing type classes: `Functor`, `Applicative`, and `Monad`.

One way to understand functions is as a means of transforming one type into another.
Let's visualize two types as two shapes, a circle and a square, as shown in
Figure 1.

![A circle and square visually representing two types](https://i.loli.net/2021/09/19/LF6m4ZcIQWv9yCq.png)

These shapes can represent any two types. When you want to transform a circle into
a square, you use a function. You can visualize a function as a connector between
two shapes, as shown in Figure 2.

![A function can transform a circle to a square](https://i.loli.net/2021/09/19/ajbkMVh5tIFKRvJ.png)

When you want  to apply a transformation, you can visualize placing your connector
between the initial  shape (in this case, a circle) and the desired shape (a square);
see Figure 3.

![Visualizing a function as a way of connecting one shape to another](https://i.loli.net/2021/09/19/5R2vuahSxpHEd6s.png)

In this unit, you'll look at working with types in context. The two best examples
of types in context that you've seen are `Maybe` types and `IO` types. Keeping with
our visual language, you can imagine types in a context as shown in Figure 4.

![The shape around the shape  represents a context](https://i.loli.net/2021/09/19/ZgdXbMrsGSmkPY9.png)

Because these types are in a context, you can't simply use your old connector to
make the transformation. To perform the transformation of your types in a context,
you need a connector that looks like Figure 5.

![A function that  connects two types in a context](https://i.loli.net/2021/09/19/Rv84T1ontYO5MFC.png)

With this connector, you can easily transform types in a context, as shown in
Figure 6.

![Context Transformation](https://i.loli.net/2021/09/19/g81UkheafQLB5sV.png)

This may seem like a perfect  solution, but there's a problem. Let's look at a
function `halve`, which is of the type `Int -> Double`, and as expected halves
the `Int` argument.

```haskell
halve :: Int -> Double
halve n = fromIntegral n / 2.0

halveMaybe :: Maybe Int -> Maybe Double
halveMaybe (Just n) = Just (halve n)
halveMaybe Nothing = Nothing
```

For this one example, it's not a big deal to write a simple wrapper. But consider
the wide range of existing functions from `a -> b`. To use any of these with `Maybe`
types would require nearly identical wrappers. Even worse is that you have no way
of writing these wrappers for `IO` types!

This is where `Funcor`, `Applicative`, and `Monda` come in. You can think of these
type classes as adapters that allow you to work with different connectors so long
as the underlying types are the same.

### Functor

`Maybe` is a member of the `Functor` type class. The `Functor` type class
requires only one definition: `fmap`.

```haskell
fmap:: Functor f => (a -> b) -> f a -> f b
```

Going back to your visual language from the introduction, `fmap` provides an adapter,
as shown below. Notice we're using `<$>`, which is a synonym for `fmap`.

![fmap visualization](https://i.loli.net/2021/09/19/5FNUWPAMQeIjrK3.png)

You can define `fmap` as a generalization of your custom `incMaybe` function.

```haskell
instance Functor Maybe where
  fmap func (Just n) = Just (func n)
  fmap func Nothing = Nothing
```

### Applicative

We now give an example.

```haskell
import qualified Data.Map as Map

type LatLong = (Double, Double)

locationDB :: Map.Map String LatLong

locationDB = Map.fromList [("Arkham",(42.6054,-70.7829))
                          ,("Innsmouth",(42.8250,-70.8150))
                          ,("Carcosa",(29.9714,-90.7694))
                          ,("New York",(40.7776,-73.9691))]

toRadians :: Double -> Double
toRadians degrees = degrees * pi / 180

latLongToRads :: LatLong -> (Double, Double)
latLongToRads (lat, long) = (rlat, rlong)
  where rlat = toRadians lat
        rlong = toRadians long

haversine :: LatLong -> LatLong -> Double
haversine coords1 coords2 = earthRadius * c
  where (rlat1,rlong1) = latLongToRads coords1
        (rlat2,rlong2) = latLongToRads coords2
        dlat = rlat2 - rlat1
        dlong = rlong2 - rlong1
        a = (sin (dlat/2))^2 + cos rlat1 * cos rlat2 * (sin (dlong/2))^2
        c = 2 * atan2 (sqrt a) (sqrt (1-a))
        earthRadius = 3961.0

printDistance :: Maybe Double -> IO()
printDistance Nothing = putStrLn "Error, invalid city entered"
printDistance (Just distance) = putStrLn (show distance ++ " miles")
```

Now you have function `haversine`, which of this type:

```haskell
haversine :: LatLong -> LatLong -> Double
```

What you need is a function that looks like below:

```haskell
Maybe LatLong -> Maybe LatLong -> Maybe Double
```

However, you cannot use `Functor` type class. The problem you need to solve now is
generalizing `Functor`'s `fmap` to work with multiple arguments. A powerful type
class called `Applicative` contains a method that's the `<*>` operator, which allows
you to use functions that **are inside a context**.

```haskell
class Functor f => Applicative f where
  (<*>) :: Applicative f => f (a -> b) -> f a -> f b
  pure :: a -> fa
```

### Monad

The `Monad` type class will allow you to perform any arbitrary computation in a
context you'd like. You already saw this power with do-notation, which is
syntactic sugar for the methods of the `Monad` type class.

```haskell
class Applicative f => Monad f where
  (>>=)  :: m a -> (a -> m b) -> m b
  (>>)   :: m a -> m b -> m b
  return :: a -> m a
  fail   :: String -> m a
```

### Do-notation

```haskell
askForName :: IO ()
askForName = putStrLn "What is your name?"

nameStatement :: String -> String
nameStatement name = "Hello, " ++ name ++ "!"

helloName :: IO ()
helloName = askForName >>
            getLine >>=
            (\name -> return $ nameStatement name) >>=
            putStrLn
```

IO in Haskell is achieved by `Monad` type class. Unfortunately, this code is messy,
and is difficult to read and write. We can rewrite the code by using do-notation.

```haskell
helloNameDo :: IO ()
helloNameDo = do
   askForName
   name <- getLine 
   putStrLn (nameStatement name)
```

