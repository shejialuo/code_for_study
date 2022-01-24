module MoreFold where

xor :: [Bool] -> Bool
xor = foldr aux False
  where aux a b = (a || b) && not (a && b)

map' :: (a -> b) -> [a] -> [b]
map' f = foldr (\x y -> f x : y) []
