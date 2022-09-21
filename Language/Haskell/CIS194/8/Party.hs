module Party where

import Employee
import Data.Monoid
import Data.Tree
import Data.Semigroup

glCons :: Employee -> GuestList -> GuestList
glCons e@(Emp name amount) (GL xs totalAmount) = GL (e:xs) (totalAmount + amount)

instance Monoid GuestList where
  mempty = GL [] 0
  mappend (GL al af) (GL bl bf) = GL (al ++ bl) (af + bf)

instance Semigroup GuestList where
  (<>) = mappend

moreFun :: GuestList -> GuestList -> GuestList
moreFun gl1 gl2 = if gl1 >= gl2 then gl1 else gl2

treeFold :: (a -> [b] -> b) -> b -> Tree a -> b
treeFold f init (Node {rootLabel = rl, subForest = sf})
  = f rl (map (treeFold f init) sf)

-- | First part of list is with boss.
nextLevel :: Employee -> [(GuestList, GuestList)] -> (GuestList, GuestList)
nextLevel boss bestLists = (maximumS withBossL, maximumS withoutBossL)
  where withoutBossL   = map fst bestLists
        -- ^ The new withoutBossList has sub bosses in it.

        withoutSubBoss = map snd bestLists
        withBossL      = map (glCons boss) withoutSubBoss
        -- ^ The new withBossList doesn't have sub bosses in it.

maximumS ::(Monoid a, Ord a) => [a] -> a
maximumS [] = mempty
maximumS lst = maximum lst

maxFun :: Tree Employee -> GuestList
maxFun tree = uncurry max res
  where res = treeFold nextLevel (mempty, mempty) tree

formatGL :: GuestList -> String
formatGL (GL lst fun) = "Total fun: " ++ show fun ++ "\n" ++ unlines employees
  where employees = map (\(Emp {empName = name}) -> name) lst

computeOutput :: String -> String
computeOutput = formatGL . maxFun . read

main :: IO ()
main = readFile "company.txt" >>= putStrLn . computeOutput
