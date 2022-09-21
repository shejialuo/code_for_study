module CopyingProblem where

{-
  * A known disadvantage of functional programs
  * is that space can be considerably replicated
-}

insert :: Ord a => a -> [a] -> [a]
insert x [] = [x]
insert x (y:ys) | x > y = y : insert x ys
                | otherwise = x:y:ys

{-
  * If the item is inserted at the end of the list,
  * this list will have to replicated because other
  * pointers to the old list may exist.
-}

{-
  * We can introduce a label in the second equation
  * of the above definition as follows
-}

insert' :: Ord a => a -> [a] -> [a]
insert' x [] = [x]
insert' x l@(y:ys) | x > y = y : insert x ys
                   | otherwise = x : l
