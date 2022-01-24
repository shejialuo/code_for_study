module JoinList where
import Sized

data JoinList m a = Empty
                  | Single m a
                  | Append m (JoinList m a) (JoinList m a)
  deriving(Eq, Show)

tag :: Monoid m => JoinList m a -> m
tag (Append m _ _ ) = m
tag (Single m a) = m
tag Empty = mempty 

(+++) :: Monoid m => JoinList m a -> JoinList m a -> JoinList m a
(+++) left right= Append (tag left <> tag right) left right

indexJ :: (Sized m, Monoid m) => Int -> JoinList m a -> Maybe a
indexJ index (Single _ a) | index == 0 = Just a
                          | otherwise = Nothing
indexJ index (Append m left right)
  | index < 0 || index > totalSize = Nothing
  | index > leftSize = indexJ (index - leftSize) right
  | otherwise = indexJ index left
    where totalSize = getSize . size $ m
          leftSize = getSize . size . tag $ left
indexJ _ _ = Nothing

dropJ :: (Sized m, Monoid m) => Int -> JoinList m a -> JoinList m a
dropJ index n@(Single _ _) | index <= 0 = n
dropJ index n@(Append m left right)
  | index > totalSize = Empty
  | index > leftSize = dropJ (index - leftSize) right
  | index > 0 = dropJ index left +++ right
  | otherwise = n
    where totalSize = getSize . size $ m
          leftSize = getSize . size . tag $ left
dropJ _ _ = Empty

takeJ :: (Sized m, Monoid m) => Int -> JoinList m a -> JoinList m a
takeJ index n@(Single _ _) | index > 0 = n
takeJ index n@(Append m left right)
  | index > totalSize = n
  | index > leftSize = left +++ takeJ (index - leftSize) right
  | index > 0 = takeJ index left
  | otherwise = Empty
    where totalSize = getSize . size $ m
          leftSize = getSize . size . tag $ left
takeJ _ _  = Empty
