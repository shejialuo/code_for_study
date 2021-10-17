import TreeStructure

{-
  Calculate the depth of Tree and BinaryTree.
-}

depth :: Tree a -> Int
depth (Node _ []) = 1
depth (Node _ successor) = 1 + maximum (map depth successor)

depth' :: BinaryTree a -> Int
depth' Empty = 1
depth' (NodeBT _ leftTree rightTree) = 1 + max (depth' leftTree) (depth' rightTree)