# Chapter 20 Foldable

## 20.1 The Foldable class

The Hackage documentation for the `Foldable` typeclass describes it
as being a "class of data structures that can be folded to a summary value".
The definition in the library begins:

```hs
class Foldable t where
  {-# MINIMAL foldmap | foldr #-}
```

## 20.2 Revenge of the monoids

Folding necessarily implies a binary associative operation that has an identity
value. The first two operations defined in `Foldable` make this explicit:

```hs
class Foldable (t :: * -> *) where
  fold :: Monoid m => t m -> m
  foldMap :: Monoid m => (a -> m) -> t a -> m
```

We can already see from the type of `fold` that it's not going to work the
same as `foldr`, because it doesn't take a function for its first argument.
But we also can't just fold up a list of numbers, because the `fold` function
doesn't have a `Monoid` specified:

```hs
fold (+) [1, 2, 3, 4, 5] -- error
fold [1, 2, 3, 4, 5]     -- error
```

We need to do to make `fold` work is specify a `Monoid` instance:

```hs
xs = map Sum [1..5]
fold xs
fold ["hello", " julie"]
```

Unlike `fold`, `foldMap` has a function as its first argument. Unlike `foldr`,
the first argument of `foldMap` must explicitly map each element of the
structure to a `Monoid`:

```hs
foldMap sum [1, 2, 3 ,4]
```
