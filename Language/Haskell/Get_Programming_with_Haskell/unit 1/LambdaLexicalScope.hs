module LambdaLexicalScope where

-- Bad Way

sumSquareOrSquareSum1 x y = if x ^ 2 + y ^ 2 > (x + y) ^ 2
                           then x ^ 2 + y ^ 2
                           else (x + y) ^ 2

-- A better way

bodyNoLambda sumSquare squareSum = if sumSquare > squareSum
                           then sumSquare
                           else squareSum

sumSquareOrSquareSum2 x y = bodyNoLambda (x ^ 2 + y ^ 2) (x + y) ^ 2

-- A clean way

sumSquareOrSquareSum3 x y = if sumSquare > squareSum
                           then sumSquare
                           else squareSum
  where sumSquare = x ^ 2 + y ^ 2
        squareSum = (x + y) ^ 2

-- Also a clean way

sumSquareOrSquareSum4 x y = let sumSquare = x ^ 2 + y ^ 2
                                squareSum = (x + y) ^ 2
                            in
                              if sumSquare > squareSum
                              then sumSquare
                              else squareSum

overwriteUseLet x = let x = 2
              in
                let x = 3
                in
                  let x= 4
                  in
                    x
