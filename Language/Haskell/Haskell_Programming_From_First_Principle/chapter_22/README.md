# Chapter 22 Reader

## 22.1 Reader

When writing applications, programmers often need to pass around
some information that may be needed throughout an entire application.
We don't want to simply pass this information as arguments because
it would be present in the type of almost every function. This
can make the code harder to read and harder to maintain. To address
this, we use `Reader`.

## 22.2 A new beginning

Look at the following code snippet:

```hs
boop :: Integer -> Integer
boop = (*2)

doop :: Integer -> Integer
doop = (+10)

bip :: Integer -> Integer
big = boop . doop

bloop :: Integer -> Integer
bloop = fmap boop doop
```

In the following code, `bip` is the same as the `bloop`. It's easy to
understand the `big`. However, the `bloop` would be different, we first
show the type traverse.

```hs
fmap :: Functor f => (a -> b) -> f a -> f b
```

We could see that the `a` would be `Integer` and `b` would be Integer, thus
the `f` should be `Integer->` or `(->) Integer`. It's a wonderful idea that
we could lift a partially applied function to a fully applied function.

Let's look at the type of the composition:

```hs
(.) :: (b -> c) -> (a -> b) -> a -> c
```

Now we could simply see that the `f` is `(->) a`. So the lift of a function can
be explained as the following: a tool that translates the result of any
other function.

This is what the `Data.Functor` defined:

```hs
instance Functor ((->) r) where
    fmap = (.)
```

So what if the partially applied function can not be fully applied, so we need
to use the `<*>` operator.

```hs
bbop :: Integer -> Integer
bbop = (+) <$> boop <*> doop

duwop :: Intger -> Integer
duwop = liftA2 (+) boop doop
```

We still look at the type traverse of the `bbop`: for `<$>`, we could easily deduce
the `f` should be `(->) Integer`. However, to make it more concrete, I will
pose a `3` instead of `Integer`. Thus `f` should be `(->) 3`. And it would be
`(Integer->Integer->Integer) -> 3->Integer -> 3->Integer->Integer`. Now we have
passed the parameter to the first function. Now we use `<*>`. It could become
the following: `3 (Integer->Integer) -> (3 -> Integer) -> (3 -> Integer)`. We
have spread the parameter for each functions.

```hs
instance Applicative ((->) r) where
    pure = const
    (<*>) f g x = f x (g x)
    liftA2 q f g x = q (f x) (g x)
```

We'd use this when two functions would share the same input and we want
to apply some other function to the result of those to reach a final
result.

This is the idea of Reader. It is a way of stringing functions together
when all those functions are awaiting one input from a shared environment.

## 22.3 Reader

`Reader` is a newtype wrapper for the function type (`->`):

```hs
newtype Reader r a = Reader {runReader :: r -> a }
```

The `r` is the type we're reading in and `a` is the result type of our function.
And the `Reader` newtype has a handy `runReader` accessor to get the function out
of `Reader`. What does the `Functor` for this look like compared ot function
composition.

```hs
instance Functor (Reader r) where
  fmap :: (a -> b) -> Reader r a -> Reader r b
  fmap f (Reader ra) = Reader $ \r -> f (ra r)
```

We can easily change it to function composition:

```hs
instance Functor (Reader r) where
  fmap :: (a -> b) -> Reader r a -> Reader r b
  fmap f (Reader ra) = Reader $ (f . ra)
```

## 22.4 Functions have an Applicative too

We should notice how the types specialize:

```hs

pure :: a -> (r -> a)
(<*>) :: (r -> a -> b) -> (r -> a) -> (r -> b)
```

We have two arguments in this function, and both of them are functions
waiting for the `r` input. When that comes, both functions will be
applied to return a final result of `b`.

```hs
import Control.Applicative (Applicative (liftA2))

newtype HumanName
  = HumanName String
  deriving (Eq, Show)

newtype DogName
  = DogName String
  deriving (Eq, Show)

newtype Address
  = Address String
  deriving (Eq, Show)

data Person = Person
  { humanName :: HumanName,
    dogName :: DogName,
    address :: Address
  }
  deriving (Eq, Show)

data Dog = Dog
  { dogsName :: DogName,
    dogsAddress :: Address
  }
  deriving (Eq, Show)

pers :: Person
pers =
  Person
    (HumanName "Big Bird")
    (DogName "Barkley")
    (Address "Sesame Street")

chris :: Person
chris =
  Person
    (HumanName "Chris Allen")
    (DogName "Papu")
    (Address "Austin")

getDog :: Person -> Dog
getDog = Dog <$> dogName <*> address

getDog' :: Person -> Dog
getDog' = liftA2 Dog dogName address
```
