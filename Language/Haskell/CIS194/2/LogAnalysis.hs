module LogAnalysis where

import Log

parseMessage :: String -> LogMessage
parseMessage logMessage = case wordLists of
  ("I": time : string) -> LogMessage Info (read time) (unwords string)
  ("W": time : string) -> LogMessage Warning (read time) (unwords string)
  ("E": num : time : string) -> LogMessage (Error $ read num ) (read time) (unwords string)
  _ -> Unknown logMessage
  where wordLists = words logMessage

parse :: String -> [LogMessage]
parse = map parseMessage . lines

{-
  * It is difficult to do a BST in Haskell
  * The most important thing here to notice is
  * When do bst insertion, when it moves to left tree,
  * the right tree doesn't matter, just put it into the
  * data constructor.
  * When it moves to right tree, the left tree doesn't
  * matter, just put it into the data constructor.
-}

insert :: LogMessage -> MessageTree -> MessageTree
insert l@(LogMessage _ time _) (Node left l'@(LogMessage _ time' _) right)
  | time < time' = Node (insert l left) l' right
  | otherwise = Node left l' (insert l right)
insert l leaf = Node leaf l leaf

build :: [LogMessage] -> MessageTree
build = foldr insert Leaf

inOrder :: MessageTree -> [LogMessage]
inOrder (Node left logMessage right) = inOrder left <> [logMessage] <> inOrder right
inOrder Leaf = []

{-
  * To test
  * testWhatWentWrong parse whatWentWrong "sample.log"
-}

whatWentWrong :: [LogMessage] -> [String]
whatWentWrong  = map extractString. filter filterAux . inOrder . build
  where filterAux (LogMessage (Error num) _ _ ) = num > 50
        filterAux _ = False
        extractString (LogMessage (Error num) _ string) = string
        extractString _ = ""
