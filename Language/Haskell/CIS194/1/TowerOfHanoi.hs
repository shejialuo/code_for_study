module TowerOfHanoi where

type Peg = String
type Move = (Peg, Peg)

hanoi :: Integer -> Peg -> Peg -> Peg -> [Move]
hanoi n a b c
  | n > 0 = hanoi (n - 1) a b c <> [(a,c)] <> hanoi (n - 1) b a c
  | otherwise = []