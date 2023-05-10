# Chapter 5 Types

## 5.1 Polymorphism

In Haskell, polymorphism divides into two categories: *parametric polymorphism* and
*constrained polymorphism*.

### Working around constraints

```hs
6 / fromIntegral (length [1..3])
```

## 5.2 Type inference

Haskell will infer the most generally applicable type that is correct.
