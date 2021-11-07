module Main where
import System.IO

main :: IO()
main = do
  myFile <- openFile "assets/hello.txt" ReadMode
  firstLine <- hGetLine myFile
  putStrLn firstLine
  secondLine <- hGetLine myFile
  putStrLn secondLine
  hClose myFile
  putStrLn "done!"