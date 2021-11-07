module CreateTypes where

-- Type synonym

type FirstName = String
type LastName = String
type Age = Int
type Height = Int

patientInfo :: FirstName -> LastName -> Age -> Height -> String
patientInfo fName lName age height = name ++ " " ++ ageHeight
  where name = lName ++ ", " ++ fName
        ageHeight = "(" ++ show age ++ "yrs. " ++ show height ++ "in.)"

-- Create new types

data Sex = Male | Female

sexInitial :: Sex -> Char
sexInitial Male = 'M'
sexInitial Female = 'F'

data RhType = Pos | Neg

data ABOType = A | B | AB | O

data BloodType = BloodType ABOType RhType

showRh :: RhType -> String
showRh Pos = "+"
showRh Neg = "-"

showABO :: ABOType -> String
showABO A = "A"
showABO B = "B"
showABO AB = "AB"
showABO O = "O"

showBloodType :: BloodType -> String
showBloodType (BloodType abo rh)  = showABO abo ++ showRh rh

type MiddleName = String
data Name = Name FirstName LastName 
          | NameWithMiddle FirstName MiddleName LastName

showName :: Name -> String
showName (Name f l) = f ++ " " ++ l
showName (NameWithMiddle f m l) = f ++ " " ++ m ++ " " ++ l

-- Record syntax

data Patient = Patient { name :: Name,
                         sex :: Sex,
                         age :: Int,
                         height :: Int,
                         weight :: Int,
                         bloodType :: BloodType}

jackieSmith :: Patient
jackieSmith = Patient {name = Name "Jackie" "Smith"
                      , age = 43
                      , sex = Female
                      , height = 62
                      , weight = 115
                      , bloodType = BloodType O Neg }

height1 = height jackieSmith
