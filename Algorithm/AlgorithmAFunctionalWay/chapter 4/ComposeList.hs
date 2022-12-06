module ComposeList where

ldouble :: Num a => [a] -> [a]
ldouble = map (2*)

ltriple :: Num a => [a] -> [a]
ltriple = map (3*)

result :: [Integer]
result = (ldouble.ltriple) [1,2,3]

{-
  * (ldouble.ltriple) [1,2,3] => ldouble (ltriple [1 , 2 , 3] ) 
  *                           => ldouble (3:(ltriple [2,3]))
  *                           => ldouble 6 :(ldouble (ltriple [2,3]))
  *                           => 6 : 12 : (ldouble (ltriple [3]))
  *                           => [6, 12, 18]
-}

{-
  * Their composition operates in constant space that
  * is in O(1).
-}
