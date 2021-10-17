import TreeStructure

-- preOrder
preOrder :: BinaryTree a -> [a]
preOrder Empty = []
preOrder (NodeBT a leftTree rightTree) = [a] <> preOrder leftTree <> preOrder rightTree

--inOrder
inOrder :: BinaryTree a -> [a]
inOrder Empty = []
inOrder (NodeBT a leftTree rightTree) = inOrder leftTree <> [a] <> inOrder rightTree

--postOrder
postOrder :: BinaryTree a -> [a]
postOrder Empty = []
postOrder (NodeBT a leftTree rightTree) = postOrder leftTree <> postOrder rightTree <> [a]
