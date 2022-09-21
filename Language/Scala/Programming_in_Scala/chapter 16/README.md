# List Operations

This chapter mainly talks about list. Scala is just like Haskell. So here, I don't write any code but summary.

```haskell
list :: [Int]
list = 1 : 2 : 3 : 4 : 5 : []
listAppend = list ++ [1,2,3,4,5]
listHead = head list
listMap = map (+1) list
listFilter = filter even list
listSum = foldl (+) 0 list
listRange = [1..5]
listConcat = mconcat ["ab","c"]
```

```scala
val list : List[String] = 1 :: 2 :: 3 :: 4 :: 5 :: Nil
val listAppend = list ::: List(1,2,3,4,5)
val listHead = list.head
val listMap = list map (_ + 1) // more functional way
val listMapAnother = list.map (_ + 1) // more imperative way
val listFilter = list filter (_ % 2 == 0)
val listSum = list.foldLeft(0)(_ + _)
val listRange = List.range(1,5)
val listConcat = List.concat("ab", "c")
```

Well, Haskell is more functional, because scala needs to
be consistent with imperative style.
