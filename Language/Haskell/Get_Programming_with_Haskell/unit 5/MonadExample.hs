module MonadExample where

askForName :: IO()
askForName = putStrLn "What is your name?"

nameStatement :: String -> String
nameStatement name = "Hello, " ++ name ++ "!"

helloName :: IO()
helloName = askForName >>
            getLine >>=
            return . nameStatement >>=
            putStrLn
