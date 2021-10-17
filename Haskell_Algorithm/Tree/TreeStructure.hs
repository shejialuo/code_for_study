module TreeStructure where

{-
  Tree is itself defined by recursion
  So use haskell is simple
-}

-- To define the Tree type
data Tree a = Node a [Tree a] deriving Show

-- To define the binary tree BinaryTree type
data BinaryTree a = Empty | NodeBT a (BinaryTree a) (BinaryTree a)
  deriving Show
