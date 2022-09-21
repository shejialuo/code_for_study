module Wholemeal where

fun1 :: [Integer] -> Integer
fun1 = foldl aux 1
  where aux = (*) . (\x -> if even x then x - 2 else 1)

fun2 :: Integer -> Integer
fun2  = sum . filter even . takeWhile (>1) . iterate aux
  where aux x = if even x then div x 2 else 3 * x + 1
