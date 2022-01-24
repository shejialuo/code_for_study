module Fibonacci where

fib :: Integer -> Integer
fib n | n >= 2 = fib (n - 1) + fib (n - 2)
      | n == 1 = 1
      | n == 0 = 0
      | otherwise = error "input must be positive"

fibs1 :: [Integer]
fibs1 = map fib [0..]
  
fibs2 :: [Integer]
fibs2 = fibIterate 0 1
  where fibIterate a b = a : fibIterate b (a + b)

data Stream a = Cons a (Stream a)

streamToList :: Stream a -> [a]
streamToList (Cons a stream) = a : streamToList stream

instance Show a => Show (Stream a) where 
  show = show . take 20 . streamToList

streamRepeat :: a -> Stream a
streamRepeat x = Cons x (streamRepeat x)

streamMap :: (a -> b) -> Stream a -> Stream b
streamMap f (Cons a stream) = Cons (f a) (streamMap f stream)

streamFromSeed :: (a -> a) -> a -> Stream a
streamFromSeed f x = Cons x (streamFromSeed f (f x))

nats :: Stream Integer
nats = streamFromSeed (+1) 0

interleaveStreams :: Stream a -> Stream a -> Stream a
interleaveStreams (Cons y ys) zs = Cons y (interleaveStreams zs ys)

ruler :: Stream Integer
ruler = startRuler 0

startRuler :: Integer -> Stream Integer
startRuler y = interleaveStreams (streamRepeat y) (startRuler (y+1))
