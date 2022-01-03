
module ValidateCardNumber where

toDigitsRev :: Integer -> [Integer]
toDigitsRev n | n > 0 = n `mod` 10 : toDigitsRev (n `div` 10)
           | otherwise = []

toDigits :: Integer -> [Integer]
toDigits = reverse . toDigitsRev

doubleEveryOther :: [Integer] -> [Integer]
doubleEveryOther l@(x: xs)
  | even $ length l = (x * 2) : doubleEveryOther xs
  | otherwise = x : doubleEveryOther xs
doubleEveryOther [] = []

sumDigits :: [Integer] -> Integer
sumDigits = foldr  ((+) . sum . toDigitsRev) 0

validate :: Integer -> Bool
validate cardNumber = (sumDigits. doubleEveryOther. toDigits) cardNumber `mod` 10 == 0