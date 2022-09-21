module Main where

import System.Environment
import System.IO

getCounts :: String -> (Int, Int, Int)
getCounts input = (charCount, wordCount, lineCount)
  where charCount = length input
        wordCount = (length.words) input
        lineCount = (length.lines) input

countsText :: (Int, Int, Int) -> String
countsText (cc, wc, lc) = unwords ["chars: ",
                                   show cc,
                                   " words: ",
                                   show wc,
                                   " lines: ",
                                   show lc]

main :: IO()
main = do
  args <- getArgs
  let fileName = head args
  file <- openFile fileName ReadMode
  input <- hGetContents file
  -- ! You can't use hClose here
  -- ! Because of lazy evaluation
  let summary = (countsText . getCounts) input
  hClose file
  appendFile "assets/stats.dat" (mconcat [fileName, " ", summary, "\n"])
  putStrLn summary
