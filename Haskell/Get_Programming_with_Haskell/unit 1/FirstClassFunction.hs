module FirstClassFunction where

-- Bad way
ifEvenIncBadWay n = if even n
              then n + 1
              else n

ifEvenDoubleBadWay n = if even n
                 then n ^ 2
                 else n

ifEvenSquareBadWay n = if even n
                 then n * 2
                 else n

-- A better way
ifEven myFunction x = if even x
                      then myFunction x
                      else x

inc n = n + 1
double n = n * 2
square n = n ^ 2

ifEvenIncGoodWay = ifEven inc
ifEvenDoubleGoodWay = ifEven double
ifEvenSquareGoodWay = ifEven square
