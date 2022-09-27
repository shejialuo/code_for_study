module EqInstance where

data TisAnInteger = TisAn Integer

instance Eq TisAnInteger where
  (==) (TisAn x) (TisAn y) = x == y

-- For testing
-- TisAn 1 == TisAn 1
-- TisAn 2 == TisAn 1

data TwoIntegers = Two Integer Integer



instance Eq TwoIntegers where
  (==) (Two x1 y1) (Two x2 y2) = (x1 == x2) && (y1 == y2)

-- For testing
-- TisAn 1 2 == TisAn 1 2
-- TisAn 2 3 == TisAn 3 2

data StringOrInt = TisAnInt Int
                 | TisAString String

instance Eq StringOrInt where
  (==) (TisAnInt x) (TisAnInt y) = x == y
  (==) (TisAString xs) (TisAString ys) = xs == ys
  (==) _ _ = False

-- For testing
-- TisAnInt 3 == TisAnInt 4
-- TisAnInt 5 == TisAnInt 5
-- TisAString "123" == TisAnInt 5
-- TisAnInt 5 == TisAString "5"

data Pair a = Pair a a

instance Ord a => Eq (Pair a) where
  (==) (Pair x1 y1) (Pair x2 y2) = (x1 == x2) && (y1 == y2)

-- For testing
-- Pair 1 2 == Pair 1 2
-- Pair "1" 2 == Pair 1 "2"

data Tuple a b = Tuple a b

instance (Ord a, Ord b) => Eq (Tuple a b) where
  (==) (Tuple x1 y1) (Tuple x2 y2) = (x1 == x2) && (y1 == y2)


data Which a = ThisOne a
             | ThatOne a

instance Ord a => Eq (Which a) where
  (==) (ThisOne x) (ThisOne y) = x == y
  (==) (ThatOne x) (ThatOne y) = x == y
  (==) _ _ = False

data EitherOr a b = Hello a
                  | Goodbye b

instance (Ord a, Ord b) => Eq (EitherOr a b) where
  (==) (Hello x) (Hello y) = x == y
  (==) (Goodbye x) (Goodbye y) = x == y
  (==) _ _ = False
