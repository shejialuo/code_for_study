module ListExample where

powersOfTwoAndThree :: Int -> [(Int, Int)]
powersOfTwoAndThree n = do
  value <- [1 .. n]
  let powersOfTwo = 2 ^ value
  let powersOfThree = 3 ^ value
  return (powersOfTwo, powersOfThree)

allEvenOdds :: Int -> [(Int, Int)]
allEvenOdds n = [(evenValue, oddValue) | evenValue <- [2,4 .. n],
                                        oddValue <- [1,3 .. n]]
