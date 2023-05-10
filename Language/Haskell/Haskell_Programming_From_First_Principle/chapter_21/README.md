# Chapter 21 Traversable

`Traversable` allows you to transform elements inside the structure
like a functor, producing applicative effects along the way, and lift
those potentially multiple instances of applicative structure outside
of the traversable structures.

## 21.1 The Traversable typeclass definition

```hs
class (Functor t, Foldable t) => Traversable t where
  traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
  traverse f = sequenceA . fmap f

  sequenceA :: Applicative f => t (f a) -> f (t a)
  sequenceA = traverse id

```

`traverse` maps each element of a structure to an action, evaluates the
actions from left to right, and collects the result.

A minimal instance for this typeclass provides an implementation of either
`traverse` or `sequenceA`, because as you can see they can be defined in
terms of each other.

## 21.2 sequenceA

The effect of `sequenceA` is flipping two contexts or structures. It doesn't
by itself allow you to apply any function to the `a` value inside the
structure; it only flips the layers of structure around.

## 21.3 traverse

In a literal sense, anytime you need to flip two type constructors around,
or map something and then flip them around, that’s probably `Traversable`.
