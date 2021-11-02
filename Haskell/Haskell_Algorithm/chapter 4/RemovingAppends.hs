module RemovingAppends where

{-
  * Sometimes we need to use append operator
  * But this is not efficient, so we aim to remove
-}

myReverse :: [a] -> [a]
myReverse [] = []
myReverse (x:xs) = myReverse xs ++ [x]

{-
  * In this example, we follow the pattern
  * f(x1, x2, ..., xn) ++ [y].
  * We need to find the way to make it becomes
  * f'(x1, x2, ..., xn y), which means
  * f'(x1, x2, ..., xn y) = f(x1, x2, ..., xn) ++ [y]
  * To derive the function f', each definition of the
  * form "f(x1, ..., xn = e)" is replaced by
  * "f'(x1, ..., xn y) = e ++ y"
-}

{-
  * For above example, we need to find
  * myReverse' xs y = (myReverse xs) ++ y

  ! myReverse' [] y = myReverse [] ++ y = y

  ! myReverse' (x:xs) y = myReverse (x : xs) ++ y
  !                     = myReverse xs ++ [x] ++ y
  !                     = myReverse xs ++ (x:y)
  !                     = myReverse' xs (x:y) 
-}

myReverse' :: [a] -> [a] -> [a]
myReverse' [] y = y
myReverse' (x:xs) y = myReverse' xs (x:y)

