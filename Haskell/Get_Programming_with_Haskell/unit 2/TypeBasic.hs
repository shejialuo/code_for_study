module TypeBasic where

-- Types

letter :: Char
letter = 'a'

interestRate :: Double
interestRate = 0.375

isFun :: Bool
isFun = True

values :: [Int]
values = [1,2,3]

testScores :: [Double]
testScores = [0.99, 0.7, 0.8]

aPet :: [Char]
aPet = "cat"

anotherPet :: String 
anotherPet = "dog"

ageAndHeight :: (Int, Int)
ageAndHeight = (34, 74)

firstLastMiddle :: (String , String, Char)
firstLastMiddle = ("Oscar", "Grouch", 'D')

-- Function Types

double :: Int -> Int
double n = n * 2

half :: Int -> Double
half n = fromIntegral n / 2

-- show

printDouble :: Int -> String 
printDouble value = show $ value * 2

-- read

aNumber :: Int 
aNumber = read "6"

anotherNumber :: Double
anotherNumber = read "6"

-- Types for first-class functions

ifEven :: (Int -> Int) -> Int -> Int
ifEven f n = if even n
             then f n
             else n

-- Type Variables

simple :: a -> a
simple x = x

makeTriple :: a -> b -> c -> (a, b, c)
makeTriple x y z = (x, y, z)
