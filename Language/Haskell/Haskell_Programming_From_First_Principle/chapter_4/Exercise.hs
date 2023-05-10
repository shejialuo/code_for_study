module Exercise where

awesome = ["Papuchon", "curry", ":)"]
also = ["Quake", "The Simons"]
allAwesome = [awesome, also]

-- 1. length :: a -> Int
-- 2.
--    a) 5
--    b) 3
--    c) 2
--    d) 5
-- 3. 6 /3 will be ok, 6 / length [1, 2, 3] will not be ok.
-- 4. 6 `div` length [1, 2, 3]
-- 5. Bool, True
-- 6. False
-- 7.
--    a) work
--    b) list type should be homogenous
--    c) work
--    d) work
--    e) 9 is not Bool type

-- 8.
isPalindrome :: (Eq a) => [a] -> Bool
isPalindrome x = reverse x == x

-- 9.
myAbs :: Integer -> Integer
myAbs x = if x < 0 then -x else x

-- 10.
f :: (a, b) -> (c, d) -> ((b, d), (a, c))
f pair1 pair2 = ((snd pair1, snd pair2), (fst pair1, fst pair2))

-- Correcting syntax

-- 1.
x = (+)
f_ xs = w `x` 1
  where w = length xs

-- 2.
id_ = \x -> x

-- 3.

f__ (a, b) = a


-- Match the function names to their types

-- 1. c)
-- 2. b)
-- 3. a)
-- 4. d)
