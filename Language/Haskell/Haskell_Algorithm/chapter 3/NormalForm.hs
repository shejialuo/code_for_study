module NormalForm where

add :: Integer -> Integer -> Integer 
add x y = x + y

double :: Integer -> Integer
double x = add x x

-- strict evaluation
result1 :: Integer
result1 = double $! (5 * 4)

-- lazy evaluation
result2 :: Integer
result2 = double (5 * 4)

{-
  * There are tow forms:
  
  * result1
  * double (5 * 4) => double 20
  *                => add 20 20
  *                => 20 + 20 => 40

  * result2
  * double (5 * 4) => add (5*4)(5*4)
  *                => (5*4) + (5*4)
  *                => 20 + 20
-}
