module Composition where

import Data.List ( sort )

-- Combine functions

myLast :: [a] -> a
myLast = head . reverse

myMin :: Ord a => [a] -> a
myMin = head . sort

myMax :: Ord a => [a] -> a
myMax = myLast . sort

-- Combing like types: Semigroups

data Color = Red |
             Yellow |
             Blue |
             Green |
             Purple |
             Orange |
             Brown deriving (Show, Eq)

instance Semigroup Color where
  (<>) Red Blue = Purple
  (<>) Blue Red = Purple
  (<>) Yellow Blue = Green
  (<>) Blue Yellow = Green
  (<>) Yellow Red = Orange
  (<>) Red Yellow = Orange
  (<>) a b = if a == b
              then a
              else Brown

-- Monoid

str :: [Char]
str = mconcat ["does", " this", " make", " sense?"]
